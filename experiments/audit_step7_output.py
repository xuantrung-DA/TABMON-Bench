"""Integrity audit for TABMON Step 7 alarm-recalibration output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_POLICIES = {"original_v7", "global_monitor", "dataset_monitor"}
EXPECTED_MONITORS = {"confidence", "drift_shap"}
FORBIDDEN_COLUMNS = {
    "target_label",
    "y_true",
    "true_excess_risk",
    "true_risk_event",
    "risk_failure",
    "attribution_failure",
    "alarm_failure",
    "monitor_failure",
    "ground_truth_attribution_json",
}


def audit_output(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    status = json.loads((output_dir / "step7_status.json").read_text("utf-8"))
    manifest = json.loads((output_dir / "step7_manifest.json").read_text("utf-8"))
    thresholds = pd.read_parquet(output_dir / "calibration_thresholds.parquet")
    batches = pd.read_parquet(output_dir / "batch_alarm_metrics.parquet")
    scenarios = pd.read_parquet(output_dir / "scenario_alarm_metrics.parquet")
    paired = pd.read_parquet(output_dir / "paired_primary_comparison.parquet")
    null_summary = pd.read_csv(output_dir / "null_calibration_summary.csv")
    detection = pd.read_csv(output_dir / "detection_summary.csv")
    feasibility = pd.read_csv(output_dir / "streamwise_feasibility.csv")

    if not status.get("complete"):
        raise ValueError("Step 7 output is incomplete")
    if manifest.get("primary_policy") != "dataset_monitor":
        raise ValueError("Unexpected primary alarm policy")
    if manifest["cross_fitting"].get("held_out_seed_excluded_from_threshold_fit") is not True:
        raise ValueError("Manifest does not guarantee held-out-seed isolation")
    if manifest["interpretation"].get("streamwise_error_controlled") is not False:
        raise ValueError("Step 7 must not claim streamwise error control")
    forbidden = sorted(set(batches.columns) & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Forbidden oracle fields found in Step 7 output: {forbidden}")
    if set(batches["policy"]) != EXPECTED_POLICIES:
        raise ValueError("Alarm policy set is incomplete")
    if set(batches["monitor"]) != EXPECTED_MONITORS:
        raise ValueError("Monitor set is incomplete")
    if set(scenarios["policy"]) != EXPECTED_POLICIES:
        raise ValueError("Scenario policy set is incomplete")

    batch_keys = ["scenario_id", "batch_index", "policy", "alpha"]
    scenario_keys = ["scenario_id", "policy", "alpha"]
    if batches.duplicated(batch_keys).any():
        raise ValueError("Duplicate Step 7 batch rows")
    if scenarios.duplicated(scenario_keys).any():
        raise ValueError("Duplicate Step 7 scenario rows")
    for policy, group_columns in {
        "global_monitor": ["monitor"],
        "dataset_monitor": ["dataset", "monitor"],
    }.items():
        subset = thresholds[thresholds["policy"] == policy]
        keys = ["held_out_seed", "alpha", *group_columns]
        if subset.duplicated(keys).any():
            raise ValueError(f"Duplicate calibration threshold for {policy}")
        if not (subset["calibration_seed_count"] == 4).all():
            raise ValueError(f"Held-out-seed leakage or missing seed in {policy}")
        for row in subset.itertuples(index=False):
            calibration_seeds = set(json.loads(row.calibration_seeds_json))
            if int(row.held_out_seed) in calibration_seeds:
                raise ValueError("Held-out seed appears in calibration seeds")

    if len(batches) != int(status["batch_alarm_rows"]):
        raise ValueError("Batch row count disagrees with status")
    if len(scenarios) != int(status["scenario_alarm_rows"]):
        raise ValueError("Scenario row count disagrees with status")
    if len(paired) != int(status["paired_primary_rows"]):
        raise ValueError("Paired row count disagrees with status")
    if len(batches) != 434_000 or len(scenarios) != 43_400 or len(paired) != 6_200:
        raise ValueError("Step 7 full-grid counts are incorrect")
    if not batches["normalized_score"].map(np.isfinite).all():
        raise ValueError("Non-finite normalized alarm score")
    if not scenarios["false_alarm_rate"].between(0.0, 1.0).all():
        raise ValueError("False-alarm rate is outside [0,1]")
    if not scenarios["batch_alarm_rate"].between(0.0, 1.0).all():
        raise ValueError("Batch alarm rate is outside [0,1]")
    if null_summary.empty or detection.empty:
        raise ValueError("Step 7 summary table is empty")
    one_percent = feasibility[np.isclose(feasibility["alpha"], 0.01)]
    if one_percent.empty or one_percent["streamwise_control_feasible"].any():
        raise ValueError(
            "Current null design should be recorded as insufficient for "
            "distribution-free 1% streamwise control"
        )

    return {
        "complete": True,
        "source_batch_rows": int(status["source_batch_rows"]),
        "alarm_variants": int(status["alarm_variants"]),
        "batch_alarm_rows": len(batches),
        "scenario_alarm_rows": len(scenarios),
        "paired_primary_rows": len(paired),
        "threshold_rows": len(thresholds),
        "policies": sorted(EXPECTED_POLICIES),
        "monitors": sorted(EXPECTED_MONITORS),
        "forbidden_output_columns": forbidden,
        "held_out_seed_isolation": True,
        "one_percent_streamwise_control_feasible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_output(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
