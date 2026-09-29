"""Integrity audit for completed TABMON PAPE Phase-1 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD = "pape"
ENDPOINT = "classification_error"
TOLERANCE_COLUMNS = {
    "failure_rate_tau_0_02",
    "failure_rate_tau_0_05",
    "failure_rate_tau_0_1",
}


def audit_output(
    output_dir: Path,
    *,
    require_complete: bool = True,
    require_full_reference: bool = False,
) -> dict:
    output_dir = output_dir.resolve()
    status = json.loads((output_dir / "pape_status.json").read_text())
    manifest = json.loads((output_dir / "pape_manifest.json").read_text())
    batches = pd.read_parquet(output_dir / "batch_metrics.parquet")
    scenarios = pd.read_parquet(output_dir / "scenario_metrics.parquet")
    density = pd.read_parquet(output_dir / "density_diagnostics.parquet")
    calibration = pd.read_parquet(output_dir / "calibration_summary.parquet")

    if require_complete and status.get("complete") is not True:
        raise ValueError("PAPE output is incomplete")
    if status.get("method") != METHOD or manifest.get("method") != METHOD:
        raise ValueError("Unexpected PAPE method identity")
    if status.get("endpoint") != ENDPOINT or manifest.get("endpoint") != ENDPOINT:
        raise ValueError("PAPE endpoint is not classification error")
    if set(batches["method"]) != {METHOD} or set(scenarios["method"]) != {METHOD}:
        raise ValueError("PAPE result contains a mixed method set")
    if set(batches["endpoint"]) != {ENDPOINT} or set(scenarios["endpoint"]) != {
        ENDPOINT
    }:
        raise ValueError("PAPE result contains a mixed endpoint")
    forbidden = {
        "target_label",
        "ground_truth_attribution_json",
        "sample_log_loss",
        "true_excess_risk",
    }
    if forbidden & set(batches.columns):
        raise ValueError("Raw oracle fields leaked into PAPE output")

    batch_keys = ["predictor_stream_id", "_tabmon_batch_index", "method"]
    scenario_keys = ["predictor_stream_id", "method"]
    density_keys = ["stream_id", "_tabmon_batch_index"]
    if batches.duplicated(batch_keys).any():
        raise ValueError("Duplicate PAPE batch records")
    if scenarios.duplicated(scenario_keys).any():
        raise ValueError("Duplicate PAPE scenario records")
    if density.duplicated(density_keys).any():
        raise ValueError("Duplicate PAPE density diagnostics")

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
    if not np.allclose(
        batches["absolute_error"],
        np.abs(batches["estimated_error"] - batches["true_error"]),
    ):
        raise ValueError("PAPE absolute-error calculation is inconsistent")
    if not TOLERANCE_COLUMNS.issubset(scenarios.columns):
        raise ValueError("PAPE tolerance sensitivity columns are incomplete")
    for column in TOLERANCE_COLUMNS | {"sign_accuracy"}:
        valid = scenarios[column].dropna()
        if not valid.between(0.0, 1.0).all():
            raise ValueError(f"{column} falls outside [0, 1]")

    if not density["effective_sample_size_fraction"].between(0.0, 1.0).all():
        raise ValueError("Density-ratio effective sample size is invalid")
    if not np.isfinite(
        density[
            [
                "weight_min",
                "weight_median",
                "weight_mean",
                "weight_max",
                "weight_p99",
            ]
        ].to_numpy()
    ).all():
        raise ValueError("Density-ratio diagnostics contain non-finite weights")
    if (density["weight_min"] <= 0.0).any():
        raise ValueError("Density-ratio weights must be positive")

    evaluated_predictors = int(status["completed_predictor_streams"])
    evaluated_streams = int(status["completed_feature_streams"])
    if scenarios["predictor_stream_id"].nunique() != evaluated_predictors:
        raise ValueError("PAPE scenario count disagrees with status")
    if batches["predictor_stream_id"].nunique() != evaluated_predictors:
        raise ValueError("PAPE batch count disagrees with status")
    if density["stream_id"].nunique() != evaluated_streams:
        raise ValueError("PAPE density-stream count disagrees with status")
    if set(batches["stream_id"].unique()) != set(density["stream_id"].unique()):
        raise ValueError("PAPE result and density streams are not aligned")

    if require_full_reference and calibration["reference_subsampled"].any():
        raise ValueError("Publication audit requires the full reference sample")
    paired_path = output_dir / "paired_stream_differences.parquet"
    paired_rows = 0
    if paired_path.is_file():
        paired = pd.read_parquet(paired_path)
        if set(paired["comparator"]) != {"ac", "cott"}:
            raise ValueError("PAPE paired comparison is missing AC or COTT")
        counts = paired.groupby("predictor_stream_id")["comparator"].nunique()
        if not counts.eq(2).all():
            raise ValueError("PAPE paired comparator grid is incomplete")
        paired_rows = len(paired)

    return {
        "complete": bool(status["complete"]),
        "method": METHOD,
        "endpoint": ENDPOINT,
        "requested_predictor_streams": int(status["requested_predictor_streams"]),
        "evaluated_predictor_streams": evaluated_predictors,
        "evaluated_feature_streams": evaluated_streams,
        "batch_records": len(batches),
        "scenario_records": len(scenarios),
        "density_records": len(density),
        "calibration_pairs": len(calibration),
        "reference_subsampled_pairs": int(calibration["reference_subsampled"].sum()),
        "paired_comparator_rows": paired_rows,
        "duplicate_records": 0,
        "raw_oracle_fields": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-full-reference", action="store_true")
    args = parser.parse_args(argv)
    report = audit_output(
        args.output_dir,
        require_complete=not args.allow_partial,
        require_full_reference=args.require_full_reference,
    )
    print("=== TABMON PAPE PHASE-1 AUDIT ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
