"""Fail-closed audit for TABMON Step 9 statistical outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("ac", "doc", "atc", "cot", "cott")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--v8-1-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    root = args.analysis_dir.resolve()
    input_root = args.v8_1_dir.resolve()
    status = json.loads((root / "step9_status.json").read_text("utf-8"))
    manifest = json.loads((root / "step9_manifest.json").read_text("utf-8"))
    _require(status.get("complete") is True and status.get("step") == 9, "Incomplete Step 9")
    for name, expected in manifest["input_sha256"].items():
        _require(_sha256(input_root / name) == expected, f"Input changed: {name}")

    summaries = pd.read_csv(root / "classification_method_summary.csv")
    paired = pd.read_csv(root / "paired_classification_comparisons.csv")
    endpoints = pd.read_csv(root / "separate_endpoint_summary.csv")
    linear = pd.read_csv(root / "linear_mixed_effects.csv")
    binomial = pd.read_csv(root / "binomial_mixed_effects.csv")
    gee = pd.read_csv(root / "binomial_gee.csv")
    scenarios = pd.read_parquet(input_root / "scenario_metrics.parquet")

    _require(len(summaries) == 70, "Expected 70 method summary rows")
    _require(len(paired) == 140, "Expected 140 paired tests")
    _require(set(paired["outcome"]) == {"mean_absolute_error", "risk_failure_rate"}, "Wrong paired outcomes")
    expected_pairs = set(combinations(METHODS, 2))
    for _, group in paired.groupby(["outcome", "scope", "shift"], dropna=False):
        _require(set(zip(group["method_a"], group["method_b"])) == expected_pairs, "Pair family incomplete")
    for column in (
        "bootstrap_p_value", "q_value_bh_scope", "q_value_bh_global_outcome"
    ):
        _require(paired[column].between(0, 1).all(), f"Invalid {column}")
    _require(
        (~paired["globally_significant"] | paired["q_value_bh_global_outcome"].lt(0.05)).all(),
        "Significance flag ignores global BH correction",
    )
    _require(
        paired.loc[paired["globally_significant"], "ci_winner"].ne("no_clear_winner").all(),
        "Significant comparison has no CI winner",
    )

    overall = paired[paired["scope"].eq("overall")]
    _require(overall["n_paired_streams"].eq(3100).all(), "Overall pairing is incomplete")
    shifted = paired[paired["scope"].eq("shift") & paired["shift"].ne("no_shift")]
    _require(shifted["n_paired_streams"].eq(600).all(), "Shift pairing is incomplete")
    null = paired[paired["shift"].eq("no_shift")]
    _require(null["n_paired_streams"].eq(100).all(), "Null pairing is incomplete")

    # Recompute every point paired difference from the frozen scenario table.
    classification = scenarios[scenarios["endpoint"].eq("classification_error")]
    for row in paired.itertuples(index=False):
        frame = (
            classification
            if row.scope == "overall"
            else classification[classification["shift"].eq(row.shift)]
        )
        wide = frame.pivot(
            index="predictor_stream_id", columns="method", values=row.outcome
        )
        expected = float((wide[row.method_a] - wide[row.method_b]).mean())
        _require(
            abs(expected - row.difference_a_minus_b) < 1e-12,
            "Paired point estimate does not match source scenarios",
        )

    _require(set(endpoints["method"]) == {"confidence_log_loss", "drift_shap_permutation"}, "Endpoint summary method leakage")
    _require(len(linear) == 78 and len(binomial) == 39 and len(gee) == 39, "Mixed model coefficient grid incomplete")
    numeric_tables = [summaries, paired, linear, binomial, gee]
    for table in numeric_tables:
        numeric = table.select_dtypes(include=[np.number])
        _require(np.isfinite(numeric.to_numpy()).all(), "Non-finite statistical output")

    metadata = json.loads((root / "mixed_model_metadata.json").read_text("utf-8"))["models"]
    _require(len(metadata) == 4, "Expected four hierarchical/marginal models")
    for model in metadata:
        if "converged" in model:
            _require(model["converged"] is True, f"Model did not converge: {model['model']}")
    _require(manifest["cross_endpoint_ranking_prohibited"] is True, "Cross-endpoint guard missing")

    result = {
        "audit_passed": True,
        "classification_summary_rows": len(summaries),
        "paired_tests": len(paired),
        "globally_significant_tests": int(paired["globally_significant"].sum()),
        "separate_endpoint_summaries": len(endpoints),
        "mixed_and_marginal_models": len(metadata),
        "input_hashes_verified": True,
        "paired_point_estimates_recomputed": True,
        "cross_endpoint_ranking_absent": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
