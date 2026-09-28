"""Fail-closed integrity audit for a TABMON schema-v8 output directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CLASSIFICATION_METHODS = {"ac", "doc", "atc", "cot", "cott"}
CONFIDENCE_METHOD = "confidence_log_loss"
DRIFT_METHOD = "drift_shap_permutation"
METHODS = CLASSIFICATION_METHODS | {CONFIDENCE_METHOD, DRIFT_METHOD}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit(results_dir: Path, require_complete: bool = False) -> dict[str, object]:
    root = results_dir.resolve()
    status = json.loads((root / "v8_status.json").read_text("utf-8"))
    manifest = json.loads((root / "v8_manifest.json").read_text("utf-8"))
    batches = pd.read_parquet(root / "batch_metrics.parquet")
    scenarios = pd.read_parquet(root / "scenario_metrics.parquet")
    thresholds = pd.read_parquet(root / "alarm_calibration_thresholds.parquet")
    maxima = pd.read_parquet(root / "alarm_calibration_maxima.parquet")

    _require(status["schema_version"] == 8, "Wrong status schema version")
    _require(manifest["schema_version"] == 8, "Wrong manifest schema version")
    if require_complete:
        _require(status["complete"] is True, "V8 run is not complete")

    completed = int(status["completed_predictor_streams"])
    requested = int(status["requested_predictor_streams"])
    _require(completed > 0 and completed <= requested, "Invalid stream progress")
    _require(len(scenarios) == completed * 7, "Incomplete method-stream grid")
    _require(len(batches) == completed * 7 * 10, "Incomplete batch-method grid")
    _require(set(batches["method"]) == METHODS, "Unexpected V8 method set")
    _require(set(scenarios["method"]) == METHODS, "Scenario methods differ")
    _require(
        batches.groupby("predictor_stream_id").size().eq(70).all(),
        "Every predictor stream must contain 7 methods x 10 batches",
    )

    endpoint_by_method = batches.groupby("method")["endpoint"].unique().to_dict()
    for method in CLASSIFICATION_METHODS:
        _require(
            endpoint_by_method[method].tolist() == ["classification_error"],
            f"{method} crossed endpoint boundary",
        )
    _require(
        endpoint_by_method[CONFIDENCE_METHOD].tolist() == ["excess_log_loss"],
        "Confidence crossed endpoint boundary",
    )
    _require(
        endpoint_by_method[DRIFT_METHOD].tolist()
        == ["feature_intervention_attribution"],
        "Drift attribution crossed endpoint boundary",
    )

    alarm_rows = batches[batches["supports_alarm"]]
    _require(
        set(alarm_rows["method"]) == {CONFIDENCE_METHOD, DRIFT_METHOD},
        "Only the two declared alarm methods may emit alarms",
    )
    _require(
        batches.loc[~batches["supports_alarm"], "alarm"].isna().all(),
        "Unsupported methods emitted alarm decisions",
    )
    target_by_method = alarm_rows.groupby("method")["alarm_target"].unique().to_dict()
    _require(
        target_by_method[CONFIDENCE_METHOD].tolist() == ["risk_event"],
        "Confidence alarm target is wrong",
    )
    _require(
        target_by_method[DRIFT_METHOD].tolist() == ["distribution_shift"],
        "Drift alarm target is wrong",
    )

    alpha = float(manifest["alarm_calibration"]["alpha"])
    n_null = int(
        manifest["alarm_calibration"][
            "null_trajectories_per_dataset_model_monitor"
        ]
    )
    expected_thresholds = (
        batches[["dataset", "model"]].drop_duplicates().shape[0] * 2
    )
    _require(len(thresholds) == expected_thresholds, "Threshold grid incomplete")
    _require(
        len(maxima) == expected_thresholds * n_null,
        "Calibration maxima grid incomplete",
    )
    expected_rank = math.ceil((n_null + 1) * (1.0 - alpha))
    _require(
        thresholds["conformal_rank"].eq(expected_rank).all(),
        "Finite-sample conformal rank is wrong",
    )
    _require(np.isfinite(thresholds["threshold"]).all(), "Non-finite threshold")
    _require(
        manifest["alarm_calibration"]["source"]
        == "reference calibration data only",
        "Alarm threshold source is not reference-only",
    )
    forbidden = set(manifest["forbidden_monitor_inputs"])
    _require(
        {"target labels", "oracle risk", "failure labels"}.issubset(forbidden),
        "Leakage boundary is incomplete",
    )

    null_scenarios = scenarios[
        scenarios["shift"].eq("no_shift") & scenarios["method"].isin(
            [CONFIDENCE_METHOD, DRIFT_METHOD]
        )
    ]
    null_fwer = (
        null_scenarios.groupby("method")["null_stream_any_alarm"].mean().to_dict()
    )
    summary = {
        "complete": bool(status["complete"]),
        "smoke_test": bool(status["smoke_test"]),
        "predictor_streams": completed,
        "method_stream_evaluations": len(scenarios),
        "batch_method_records": len(batches),
        "thresholds": len(thresholds),
        "calibration_maxima": len(maxima),
        "null_stream_any_alarm_rate": {
            key: float(value) for key, value in null_fwer.items()
        },
        "audit_passed": True,
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = audit(args.results_dir, args.require_complete)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
