"""Integrity audit for TABMON Step 6 common-backend ablation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKENDS = {"tree_shap", "permutation"}
MODELS = {"rf", "xgb"}
SHIFTS = {"no_shift", "covariate", "correlated", "pipeline", "support"}
FORBIDDEN_COLUMNS = {
    "target_label",
    "target_labels",
    "y_true",
    "oracle_risk",
    "target_log_loss",
    "true_excess_risk",
    "sample_log_loss",
}


def audit_output(output_dir: Path, require_complete: bool = True) -> dict:
    output_dir = output_dir.resolve()
    status = json.loads((output_dir / "step6_status.json").read_text("utf-8"))
    manifest = json.loads((output_dir / "step6_manifest.json").read_text("utf-8"))
    batches = pd.read_parquet(output_dir / "batch_metrics.parquet")
    scenarios = pd.read_parquet(output_dir / "scenario_metrics.parquet")
    paired = pd.read_parquet(output_dir / "paired_backend_comparisons.parquet")
    batch_agreement = pd.read_parquet(
        output_dir / "batch_backend_agreement.parquet"
    )
    importance = pd.read_parquet(output_dir / "global_importance.parquet")
    importance_agreement = pd.read_csv(
        output_dir / "global_importance_agreement.csv"
    )

    if require_complete and not status.get("complete"):
        raise ValueError("Step 6 output is incomplete")
    if manifest.get("silent_backend_fallback") is not False:
        raise ValueError("Manifest does not prohibit silent backend fallback")
    if set(batches["backend"]) != BACKENDS or set(scenarios["backend"]) != BACKENDS:
        raise ValueError("Step 6 backend set is incomplete")
    if set(batches["model"]) != MODELS or set(scenarios["model"]) != MODELS:
        raise ValueError("Step 6 model set is incomplete")
    if not set(batches["shift"]).issubset(SHIFTS) or "concept" in set(
        batches["shift"]
    ):
        raise ValueError("Unexpected shift family in Step 6 output")
    forbidden = sorted(set(batches.columns) & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Target-label/oracle fields leaked into output: {forbidden}")

    batch_keys = ["predictor_stream_id", "_tabmon_batch_index", "backend"]
    scenario_keys = ["predictor_stream_id", "backend"]
    if batches.duplicated(batch_keys).any():
        raise ValueError("Duplicate Step 6 batch-backend rows")
    if scenarios.duplicated(scenario_keys).any():
        raise ValueError("Duplicate Step 6 scenario-backend rows")
    if paired["predictor_stream_id"].duplicated().any():
        raise ValueError("Duplicate paired backend rows")

    completed = int(status["completed_model_stream_configurations"])
    if status.get("complete") and status.get("smoke_test"):
        if completed != 10:
            raise ValueError("Complete smoke output must contain 10 configurations")
    if status.get("complete") and not status.get("smoke_test"):
        if completed != 1250:
            raise ValueError("Complete full output must contain 1,250 configurations")
        if batches["dataset"].nunique() != 5:
            raise ValueError("Complete full output must contain five datasets")
    expected_batch_rows = completed * 10 * len(BACKENDS)
    expected_scenario_rows = completed * len(BACKENDS)
    if len(batches) != expected_batch_rows:
        raise ValueError(
            f"Expected {expected_batch_rows} batch rows; found {len(batches)}"
        )
    if len(scenarios) != expected_scenario_rows:
        raise ValueError(
            f"Expected {expected_scenario_rows} scenario rows; found {len(scenarios)}"
        )
    if len(paired) != completed:
        raise ValueError("Paired table does not contain one row per model stream")
    if len(batch_agreement) != completed * 10:
        raise ValueError("Batch agreement table is incomplete")

    backend_counts = scenarios.groupby("predictor_stream_id")["backend"].nunique()
    if not (backend_counts == len(BACKENDS)).all():
        raise ValueError("Not every stream contains both importance backends")
    if importance.duplicated(["dataset", "model", "backend", "feature"]).any():
        raise ValueError("Duplicate global-importance feature rows")
    backend_importance_counts = importance.groupby(["dataset", "model", "feature"])[
        "backend"
    ].nunique()
    if not (backend_importance_counts == len(BACKENDS)).all():
        raise ValueError("Global-importance backend pairing is incomplete")
    expected_pairs = scenarios[["dataset", "model"]].drop_duplicates()
    if len(importance_agreement) != len(expected_pairs):
        raise ValueError("Importance-agreement pair count is incomplete")

    numeric_columns = [
        "attribution_abs_mass",
        "attribution_max_abs",
        "false_attribution_mass",
    ]
    if not np.isfinite(batches[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite attribution magnitude found")
    active = batches["attribution_target_active"]
    if batches.loc[active, "top3_recall"].isna().any():
        raise ValueError("Active intervention is missing Recall@3")
    if not batches.loc[~active, "top3_recall"].isna().all():
        raise ValueError("Inactive intervention unexpectedly has Recall@3")
    if not batches["top3_recall"].dropna().between(0.0, 1.0).all():
        raise ValueError("Recall@3 is outside [0, 1]")
    if not batch_agreement["top3_jaccard"].between(0.0, 1.0).all():
        raise ValueError("Top-3 Jaccard is outside [0, 1]")

    expected_checkpoints = completed
    actual_checkpoints = len(list((output_dir / "batch_streams").glob("*.parquet")))
    if actual_checkpoints != expected_checkpoints:
        raise ValueError(
            f"Expected {expected_checkpoints} checkpoints; found {actual_checkpoints}"
        )

    return {
        "complete": bool(status["complete"]),
        "smoke_test": bool(status["smoke_test"]),
        "model_stream_configurations": completed,
        "backend_scenario_evaluations": len(scenarios),
        "batch_records": len(batches),
        "importance_pairs": len(importance_agreement),
        "backends": sorted(BACKENDS),
        "models": sorted(MODELS),
        "shifts": sorted(set(batches["shift"])),
        "target_label_or_risk_columns": forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = audit_output(args.output_dir, require_complete=not args.allow_incomplete)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
