"""Fail-closed structural and leakage audit for Step-12 outputs."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.shd import METHOD as SHD_METHOD, SHDMonitor
from src.baselines.xpe import XPEExplainer


GRID_METHODS = {
    "ac",
    "doc",
    "atc",
    "cot",
    "cott",
    "confidence_log_loss",
    SHD_METHOD,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root = args.results_dir.resolve()
    status = json.loads((root / "step12_status.json").read_text("utf-8"))
    manifest = json.loads((root / "step12_manifest.json").read_text("utf-8"))
    if int(manifest.get("schema_version", 0)) < 13:
        raise ValueError("This audit requires the paper-fidelity Step-12 schema")
    batches = pd.read_parquet(root / "batch_metrics.parquet")
    scenarios = pd.read_parquet(root / "scenario_metrics.parquet")
    xpe = pd.read_parquet(root / "xpe_metrics.parquet")
    calibration = pd.read_parquet(root / "shd_calibration.parquet")

    if args.require_complete and status.get("complete") is not True:
        raise ValueError("Step-12 output is incomplete")
    completed = int(status["completed_predictor_streams"])
    batch_counts = batches.groupby(["predictor_stream_id", "method"]).size()
    if batch_counts.nunique() != 1:
        raise ValueError("Method streams do not share a common batch count")
    num_batches = int(batch_counts.iloc[0])
    expected_batch_rows = completed * num_batches * len(GRID_METHODS)
    expected_scenarios = completed * len(GRID_METHODS)
    if len(batches) != expected_batch_rows:
        raise ValueError(f"Batch rows {len(batches)} != {expected_batch_rows}")
    if len(scenarios) != expected_scenarios:
        raise ValueError(f"Scenario rows {len(scenarios)} != {expected_scenarios}")
    if set(scenarios["method"]) != GRID_METHODS:
        raise ValueError("Step-12 grid method set is incomplete")
    if scenarios.duplicated(["predictor_stream_id", "method"]).any():
        raise ValueError("Duplicate method/predictor-stream evaluation")
    if len(xpe) != int(status["completed_xpe_streams"]):
        raise ValueError("XPE status count does not match xpe_metrics")
    if len(xpe) and (
        (~xpe["model"].isin(["lr", "xgb"])).any()
        or (~xpe["shift"].isin(["no_shift", "covariate", "correlated", "pipeline", "support"])).any()
        or ((xpe["shift"] != "no_shift") & (xpe["mode"] != "abrupt")).any()
    ):
        raise ValueError("XPE record escaped its predeclared subset")

    classification = scenarios[scenarios["method"].isin(["ac", "doc", "atc", "cot", "cott"])]
    if set(classification["endpoint"]) != {"classification_error"}:
        raise ValueError("Classification estimators were compared across endpoints")
    confidence = scenarios[scenarios["method"] == "confidence_log_loss"]
    if set(confidence["endpoint"]) != {"excess_log_loss"}:
        raise ValueError("Confidence endpoint is not excess log loss")
    if not np.isfinite(classification["mean_absolute_error"]).all():
        raise ValueError("Classification estimator metrics are non-finite")

    shd_parameters = set(inspect.signature(SHDMonitor.monitor).parameters)
    if shd_parameters != {"self", "target_X"}:
        raise ValueError(f"Unsafe SHD inference API: {shd_parameters}")
    xpe_parameters = set(inspect.signature(XPEExplainer.explain).parameters)
    forbidden = {"target_y", "target_labels", "oracle_risk", "failure_labels"}
    if xpe_parameters & forbidden:
        raise ValueError("XPE inference API accepts forbidden oracle inputs")
    declared_forbidden = set(manifest.get("forbidden_monitor_inputs", []))
    if declared_forbidden != {
        "target labels",
        "oracle risk",
        "failure labels",
        "intervention ground truth",
    }:
        raise ValueError("Step-12 leakage declaration is incomplete")
    if (calibration["source_size"] <= 0).any():
        raise ValueError("Invalid SHD source calibration record")
    required_shd = {
        "selector_feasible",
        "source_selected_high_error_rate",
        "source_selected_high_error_upper",
        "source_false_discovery_joint_upper",
        "alpha_target",
    }
    if not required_shd.issubset(calibration.columns):
        raise ValueError("SHD fidelity diagnostics are incomplete")
    shd_scenarios = scenarios[scenarios["method"] == SHD_METHOD]
    if set(shd_scenarios["endpoint"]) != {
        "sequential_selected_high_error_prevalence"
    }:
        raise ValueError("SHD is not evaluated against its published estimand")
    if len(xpe):
        if set(xpe["method"]) != {manifest.get("xpe_method")}:
            raise ValueError("Unexpected XPE method identity")
        if (xpe["source_transport_rows"] != xpe["target_transport_rows"]).any():
            raise ValueError("XPE transport samples are not equal-sized")
        if (xpe["coupling_retained_mass_fraction"] < 1.0 - 1e-6).any():
            raise ValueError("XPE argmax matching discarded coupling mass")
        if (xpe["coupling_one_to_one_fraction"] < 1.0 - 1e-6).any():
            raise ValueError("XPE coupling is not one-to-one")
        if (xpe["shapley_efficiency_max_abs_error"] > 1e-5).any():
            raise ValueError("XPE Shapley efficiency check failed")

    audit = {
        "pass": True,
        "complete": bool(status["complete"]),
        "predictor_streams": completed,
        "method_streams": len(scenarios),
        "batch_rows": len(batches),
        "batches_per_stream": num_batches,
        "xpe_streams": len(xpe),
        "shd_calibrations": len(calibration),
        "shd_feasible_calibrations": int(calibration["selector_feasible"].sum()),
        "endpoint_separation": True,
        "monitor_api_oracle_free": True,
        "xpe_subset_enforced": True,
        "shd_published_estimand": True,
        "xpe_equal_mass_transport": True,
    }
    (root / "step12_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
