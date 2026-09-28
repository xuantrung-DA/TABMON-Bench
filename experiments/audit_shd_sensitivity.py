"""Fail-closed audit for the cached SHD inference-sensitivity output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root = args.results_dir.resolve()
    status = json.loads(
        (root / "shd_sensitivity_status.json").read_text("utf-8")
    )
    manifest = json.loads(
        (root / "shd_sensitivity_manifest.json").read_text("utf-8")
    )
    metrics = pd.read_parquet(root / "shd_sensitivity_metrics.parquet")
    calibration = pd.read_parquet(
        root / "shd_sensitivity_calibration.parquet"
    )
    if args.require_complete and status.get("complete") is not True:
        raise ValueError("SHD sensitivity output is incomplete")
    alphas = sorted(float(value) for value in manifest["alphas"])
    epsilons = sorted(float(value) for value in manifest["epsilons"])
    expected_configs = len(alphas) * len(epsilons)
    if int(status["sensitivity_configurations"]) != expected_configs:
        raise ValueError("Sensitivity configuration count is inconsistent")
    completed = int(status["completed_predictor_streams"])
    if len(metrics) != completed * expected_configs:
        raise ValueError("Every completed stream must have the full grid")
    if metrics.duplicated(["predictor_stream_id", "alpha", "epsilon"]).any():
        raise ValueError("Duplicate SHD sensitivity evaluation")
    if sorted(metrics["alpha"].unique()) != alphas:
        raise ValueError("Alpha grid mismatch")
    if sorted(metrics["epsilon"].unique()) != epsilons:
        raise ValueError("Epsilon grid mismatch")
    cell_counts = metrics.groupby(["dataset", "model", "alpha", "epsilon"])[
        "selector_feasible"
    ].nunique()
    if not cell_counts.eq(1).all():
        raise ValueError("Selector feasibility changes within a source cell")
    selector_columns = [
        "true_error_threshold",
        "predicted_error_threshold",
        "selector_fdp",
        "selector_power",
        "selector_feasible",
    ]
    for column in selector_columns:
        spread = calibration.groupby(["dataset", "model"])[column].nunique(
            dropna=False
        )
        if not spread.eq(1).all():
            raise ValueError(f"Selector changed across inference grid: {column}")
    evaluated = metrics[metrics["evaluated"]]
    if not evaluated["selector_feasible"].all():
        raise ValueError("An infeasible selector was scored for power")
    finite = evaluated[
        ["stream_any_event", "stream_any_alarm", "max_oracle_assumption_4_1_gap"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(finite).all().all():
        raise ValueError("Evaluated SHD sensitivity metrics are non-finite")
    audit = {
        "pass": True,
        "complete": bool(status["complete"]),
        "predictor_streams": completed,
        "configurations": expected_configs,
        "metric_rows": len(metrics),
        "source_cells": int(
            calibration[["dataset", "model"]].drop_duplicates().shape[0]
        ),
        "feasible_source_cells": int(
            calibration.drop_duplicates(["dataset", "model"])[
                "selector_feasible"
            ].sum()
        ),
        "evaluated_rows": len(evaluated),
        "selector_fixed_across_grid": True,
        "endpoint": "published SHD Phi_q^2 event",
    }
    (root / "shd_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
