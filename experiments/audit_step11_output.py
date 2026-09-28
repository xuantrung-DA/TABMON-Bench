"""Fail-closed audit for Step 11 natural-shift external validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.performance_estimators import METHODS
from src.evaluation.natural_shift_protocol import validate_monitor_output_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step11-dir", type=Path, required=True)
    parser.add_argument("--acs-file", type=Path, required=True)
    parser.add_argument("--bank-file", type=Path, required=True)
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
    root = args.step11_dir.resolve()
    status = json.loads((root / "step11_status.json").read_text("utf-8"))
    manifest = json.loads((root / "step11_manifest.json").read_text("utf-8"))
    _require(status.get("complete") is True and status.get("step") == 11, "Incomplete Step 11")
    _require(manifest.get("uses_synthetic_shift_generator") is False, "Synthetic generator leaked into external validation")
    _require(manifest.get("target_labels_available_to_monitors") is False, "Target-label boundary missing")
    _require(manifest.get("domain_selection_uses_outcomes") is False, "Outcome-selected external domain")
    _require(manifest.get("endpoints_kept_separate") is True, "Endpoints were mixed")
    input_paths = {
        "acs_income_raw": args.acs_file.resolve(),
        "bank_marketing_raw": args.bank_file.resolve(),
    }
    for name, path in input_paths.items():
        _require(_sha256(path) == manifest["input_sha256"][name], f"Input changed: {name}")

    assignments = pd.read_parquet(root / "natural_domain_assignments.parquet")
    guard = pd.read_parquet(root / "natural_schema_guard.parquet")
    monitor = pd.read_parquet(root / "natural_monitor_outputs.parquet")
    oracle = pd.read_parquet(root / "natural_oracle_metrics.parquet")
    offline = pd.read_parquet(root / "natural_offline_evaluation.parquet")
    summary = pd.read_csv(root / "natural_method_summary.csv")
    paired = pd.read_csv(root / "natural_paired_comparisons.csv")

    _require(len(assignments) == 240_876, "Natural split assignment count changed")
    _require(not assignments.duplicated(["dataset", "row_id"]).any(), "Natural partitions overlap")
    _require(assignments["dataset"].nunique() == 2, "Expected two natural datasets")
    _require(assignments.loc[assignments["partition"].eq("external_target"), "domain_id"].nunique() == 10, "Expected ten external domains")
    bank_external = assignments[
        assignments["dataset"].eq("bank_marketing")
        & assignments["partition"].eq("external_target")
    ].sort_values("row_id")
    _require(bank_external["row_id"].is_monotonic_increasing, "Bank target is not ordered")
    _require(int(bank_external["row_id"].min()) == int(np.floor(45_211 * 0.70)), "Bank target does not start at frozen temporal boundary")

    _require(len(guard) == 12 and guard["domain_id"].nunique() == 11, "Guard domain grid incomplete")
    _require(len(monitor) == 288, "Monitor-domain grid incomplete")
    validate_monitor_output_columns(monitor.columns)
    _require(_sha256(root / "natural_monitor_outputs.parquet") == manifest["monitor_output_sha256_before_oracle_access"], "Observable monitor output changed after oracle access")
    _require(len(oracle) == len(monitor) == len(offline), "Offline join is incomplete")
    _require(set(monitor["method"]) == {*METHODS, "confidence_log_loss"}, "Method set incomplete")
    _require(
        set(monitor.loc[monitor["method"].eq("confidence_log_loss"), "endpoint"])
        == {"excess_log_loss"},
        "Confidence endpoint changed",
    )
    _require(
        set(monitor.loc[~monitor["method"].eq("confidence_log_loss"), "endpoint"])
        == {"classification_error"},
        "Classification endpoint changed",
    )
    natural = monitor[monitor["domain_type"].str.startswith("external_")]
    _require(len(natural) == 240, "Expected 10 domains x 4 models x 6 methods")
    source = monitor[monitor["domain_type"].eq("source_holdout")]
    _require(len(source) == 48, "Expected two source holdouts x 4 models x 6 methods")
    _require(
        np.isfinite(
            monitor.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all(),
        "Non-finite observable output",
    )
    _require(
        np.allclose(offline["signed_error"], offline["estimated_value"] - offline["true_value"]),
        "Signed error does not match endpoints",
    )
    _require(
        np.allclose(offline["absolute_error"], np.abs(offline["signed_error"])),
        "Absolute error does not match endpoints",
    )
    _require(
        offline["failure"].eq(offline["absolute_error"].gt(0.05).astype(float)).all(),
        "Failure threshold changed",
    )

    _require(len(paired) == 20, "Expected 20 paired classification tests")
    _require(set(paired["outcome"]) == {"absolute_error", "failure"}, "Wrong paired outcomes")
    _require(paired["natural_domains"].eq(10).all(), "Natural-domain pairing incomplete")
    _require(paired["predictor_domains"].eq(40).all(), "Model-domain pairing incomplete")
    _require(paired["exact_cluster_sign_flip_p"].between(0, 1).all(), "Invalid exact p-value")
    _require(paired["q_value_bh_global"].between(0, 1).all(), "Invalid global BH q-value")
    _require(not paired["method_a"].eq("confidence_log_loss").any(), "Cross-endpoint comparison present")
    _require(not paired["method_b"].eq("confidence_log_loss").any(), "Cross-endpoint comparison present")
    _require(len(summary) > 0, "Natural summary is empty")

    result = {
        "audit_passed": True,
        "input_hashes_verified": True,
        "synthetic_generator_absent": True,
        "outcome_independent_domain_selection": True,
        "target_label_monitor_boundary_verified": True,
        "observable_output_hash_verified": True,
        "external_natural_domains": 10,
        "monitor_domain_evaluations": len(monitor),
        "paired_tests": len(paired),
        "cross_endpoint_ranking_absent": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
