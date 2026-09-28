"""Integrity audit for TABMON Phase 5 performance-estimator output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {"ac", "doc", "atc", "cot", "cott"}
ENDPOINT = "classification_error"


def audit_output(output_dir: Path, require_complete: bool = True) -> dict:
    output_dir = output_dir.resolve()
    status = json.loads(
        (output_dir / "phase5_status.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "phase5_manifest.json").read_text(encoding="utf-8")
    )
    batches = pd.read_parquet(output_dir / "batch_metrics.parquet")
    scenarios = pd.read_parquet(output_dir / "scenario_metrics.parquet")
    paired = pd.read_csv(output_dir / "paired_method_comparisons.csv")
    calibration = pd.read_parquet(output_dir / "calibration_summary.parquet")

    if require_complete and not status.get("complete"):
        raise ValueError("Phase 5 output is incomplete")
    if status.get("endpoint") != ENDPOINT:
        raise ValueError("Status endpoint is not classification error")
    if manifest["endpoint"].get("log_loss_comparison_permitted") is not False:
        raise ValueError("Manifest does not prohibit log-loss comparison")
    if set(batches["method"]) != METHODS or set(scenarios["method"]) != METHODS:
        raise ValueError("Method set is incomplete")
    if set(batches["endpoint"]) != {ENDPOINT} or set(scenarios["endpoint"]) != {
        ENDPOINT
    }:
        raise ValueError("Mixed or incorrect endpoint")
    if any("log_loss" in column or column == "risk" for column in batches):
        raise ValueError("Log-loss field found in classification endpoint output")

    batch_keys = ["predictor_stream_id", "_tabmon_batch_index", "method"]
    scenario_keys = ["predictor_stream_id", "method"]
    if batches.duplicated(batch_keys).any() or scenarios.duplicated(scenario_keys).any():
        raise ValueError("Duplicate paired evaluation records")
    if not (batches.groupby("predictor_stream_id")["method"].nunique() == 5).all():
        raise ValueError("A predictor stream is missing a method")
    if not (scenarios.groupby("predictor_stream_id")["method"].nunique() == 5).all():
        raise ValueError("A scenario is missing a method")

    for column in (
        "true_error",
        "true_accuracy",
        "estimated_error",
        "estimated_accuracy",
    ):
        if not batches[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} falls outside [0, 1]")
    if not np.allclose(batches["true_accuracy"], 1.0 - batches["true_error"]):
        raise ValueError("True accuracy/error identity failed")
    if not np.allclose(
        batches["estimated_accuracy"], 1.0 - batches["estimated_error"]
    ):
        raise ValueError("Estimated accuracy/error identity failed")

    oracle_consistency = batches.groupby(
        ["predictor_stream_id", "_tabmon_batch_index"]
    ).agg(true_error_values=("true_error", "nunique"))
    if not (oracle_consistency["true_error_values"] == 1).all():
        raise ValueError("Methods do not share the same paired oracle endpoint")

    requested = int(status["requested_predictor_streams"])
    evaluated = int(status["completed_predictor_streams"])
    expected_scenarios = evaluated * len(METHODS)
    if len(scenarios) != expected_scenarios:
        raise ValueError(
            f"Expected {expected_scenarios} scenario-method rows; found {len(scenarios)}"
        )
    expected_batches = int(batches["_tabmon_batch_index"].nunique())
    if len(batches) != evaluated * expected_batches * len(METHODS):
        raise ValueError("Batch grid is incomplete")
    if len(paired) != 10 or not (paired["paired_streams"] == evaluated).all():
        raise ValueError("Paired method comparison grid is incomplete")
    if not np.allclose(calibration["atc_fitted_rate"], calibration["source_error"]):
        raise ValueError("ATC source calibration rate is inconsistent")
    if not np.allclose(calibration["cott_fitted_rate"], calibration["source_error"]):
        raise ValueError("COTT source calibration rate is inconsistent")

    return {
        "complete": bool(status["complete"]),
        "smoke_test": bool(status["smoke_test"]),
        "endpoint": status["endpoint"],
        "methods": sorted(METHODS),
        "requested_predictor_streams": requested,
        "evaluated_predictor_streams": evaluated,
        "method_stream_evaluations": len(scenarios),
        "batch_records": len(batches),
        "paired_method_comparisons": len(paired),
        "calibration_pairs": len(calibration),
        "duplicate_batch_records": 0,
        "duplicate_scenario_records": 0,
        "paired_oracle_consistent": True,
        "log_loss_fields": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = audit_output(args.output_dir, require_complete=not args.allow_partial)
    print("=== TABMON PHASE 5 AUDIT ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
