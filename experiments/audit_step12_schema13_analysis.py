"""Fail-closed audit for the final Schema-13 statistical analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {"ac", "doc", "atc", "cot", "cott"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.analysis_dir.resolve()
    manifest = json.loads(
        (root / "schema13_analysis_manifest.json").read_text("utf-8")
    )
    if manifest.get("complete") is not True:
        raise ValueError("Schema-13 analysis is incomplete")
    summaries = pd.read_parquet(root / "performance_method_summaries.parquet")
    pairs = pd.read_parquet(root / "paired_method_comparisons.parquet")
    endpoints = pd.read_csv(root / "separate_endpoint_summaries.csv")
    xpe = pd.read_csv(root / "xpe_summaries.csv")
    applicability = pd.read_csv(root / "shd_applicability.csv")
    if set(summaries["task"]) != {"binary", "multiclass"}:
        raise ValueError("Both binary and multiclass tasks must be summarized")
    if set(summaries["method"]) != METHODS:
        raise ValueError("Performance-estimator method set is incomplete")
    if set(summaries["outcome"]) != {
        "mean_absolute_error",
        "risk_failure_rate",
    }:
        raise ValueError("Unexpected performance endpoint")
    if summaries.duplicated(
        ["task", "scope", "shift", "method", "outcome"]
    ).any():
        raise ValueError("Duplicate method summary")
    if not np.isfinite(
        summaries[["estimate", "ci_lower", "ci_upper"]]
    ).all().all():
        raise ValueError("Non-finite method interval")
    if (summaries["ci_lower"] > summaries["ci_upper"]).any():
        raise ValueError("Inverted method interval")
    if pairs.duplicated(
        ["task", "scope", "shift", "outcome", "method_a", "method_b"]
    ).any():
        raise ValueError("Duplicate paired comparison")
    if not pairs["method_a"].isin(METHODS).all() or not pairs[
        "method_b"
    ].isin(METHODS).all():
        raise ValueError("Cross-endpoint method entered paired comparisons")
    if set(endpoints["method"]) - {
        "confidence_log_loss",
        "shd_quantile_phi2_pm_eb",
    }:
        raise ValueError("Unexpected separate-endpoint method")
    if set(xpe["task"]) != {"binary", "multiclass"}:
        raise ValueError("XPE summaries must cover both tasks")
    if set(applicability["task"]) != {"binary", "multiclass"}:
        raise ValueError("SHD applicability must cover both tasks")
    if manifest.get("endpoint_separation") is not True:
        raise ValueError("Endpoint-separation declaration missing")
    if manifest.get("mixed_models_run") is True:
        mixed_files = [
            "mixed_linear_mae.csv",
            "mixed_linear_failure_rate.csv",
            "mixed_binomial_fixed.csv",
            "mixed_binomial_variance.csv",
            "gee_majority_failure.csv",
        ]
        for name in mixed_files:
            table = pd.read_csv(root / name)
            numeric = table.select_dtypes(include=[np.number])
            if numeric.empty or not np.isfinite(numeric).all().all():
                raise ValueError(f"Non-finite mixed/GEE output: {name}")
        mixed_metadata = json.loads(
            (root / "mixed_model_metadata.json").read_text("utf-8")
        )
        if len(mixed_metadata) != 4:
            raise ValueError("Mixed-model metadata is incomplete")
        gee_metadata = next(
            (
                item
                for item in mixed_metadata
                if item.get("model") == "binomial_gee_majority_failure"
            ),
            None,
        )
        if gee_metadata is None or gee_metadata.get("converged") is not True:
            raise ValueError("GEE metadata is missing or unconverged")
    audit = {
        "pass": True,
        "complete": True,
        "performance_summaries": len(summaries),
        "paired_comparisons": len(pairs),
        "separate_endpoint_summaries": len(endpoints),
        "xpe_summaries": len(xpe),
        "shd_tasks": len(applicability),
        "mixed_models_run": bool(manifest["mixed_models_run"]),
        "mixed_model_outputs_finite": bool(manifest["mixed_models_run"]),
        "endpoint_separation": True,
    }
    (root / "schema13_analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
