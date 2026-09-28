"""Train and calibrate TABMON-Bench base models.

This entry point is intentionally resumable: metrics are checkpointed after every
dataset/model pair so a Kaggle session can be restarted without repeating work.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
import xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_models import BaseModelsTrainer


DATASETS = [
    "adult",
    "bank_marketing",
    "acs_income",
    "covertype",
    "diabetes_hospitals",
]
MODELS = ["lr", "rf", "xgb", "mlp"]
SPLIT_FILES = ("train.parquet", "calibration.parquet", "test_pool.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train calibrated base models for TABMON-Bench."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to train (default: all five).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
        help="Model families to train (default: lr rf xgb mlp).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing <dataset>/{train,calibration,test_pool}.parquet.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "base_models",
        help="Writable output directory.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Retrain pairs already present in the metrics checkpoint.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately at the first failed dataset/model pair.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help=(
            "Soft wall-time limit in hours. The sweep stops before starting the "
            "next dataset/model pair; an active fit is allowed to finish."
        ),
    )
    return parser.parse_args()


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Replace a CSV checkpoint without leaving a partially written file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pair_completed(
    metrics: pd.DataFrame, results_dir: Path, dataset: str, model: str
) -> bool:
    required = {"dataset", "model"}
    if metrics.empty or not required.issubset(metrics.columns):
        return False
    row_exists = bool(
        ((metrics["dataset"] == dataset) & (metrics["model"] == model.upper())).any()
    )
    model_exists = (
        results_dir / "models" / dataset / f"{model}_calibrated.pkl"
    ).is_file()
    artifacts_exist = (
        results_dir / "artifacts" / dataset / f"{model}_artifacts.parquet"
    ).is_file()
    return row_exists and model_exists and artifacts_exist


def _replace_pair(
    metrics: pd.DataFrame, new_rows: pd.DataFrame, dataset: str, model: str
) -> pd.DataFrame:
    if not metrics.empty and {"dataset", "model"}.issubset(metrics.columns):
        keep = ~(
            (metrics["dataset"] == dataset)
            & (metrics["model"] == model.upper())
        )
        metrics = metrics.loc[keep]
    return pd.concat([metrics, new_rows], ignore_index=True)


def _validate_inputs(data_dir: Path, datasets: list[str]) -> None:
    missing = [
        data_dir / dataset / filename
        for dataset in datasets
        for filename in SPLIT_FILES
        if not (data_dir / dataset / filename).is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing processed dataset files:\n{formatted}")


def _write_run_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": args.datasets,
        "models": args.models,
        "data_dir": str(args.data_dir.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "resume": not args.no_resume,
        "max_hours": args.max_hours,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    path = args.results_dir / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours must be greater than zero")
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    _validate_inputs(args.data_dir, args.datasets)
    _write_run_manifest(args)

    metrics_path = args.results_dir / "baseline_metrics_summary.csv"
    errors_path = args.results_dir / "training_errors.csv"
    all_metrics = _load_checkpoint(metrics_path)
    errors: list[dict[str, str]] = []
    started_at = time.monotonic()
    deadline = (
        started_at + args.max_hours * 60 * 60
        if args.max_hours is not None
        else None
    )
    stopped_for_time = False

    print("=== PHASE 2: TRAIN BASE MODELS & CALIBRATION ===", flush=True)
    print(f"Data:    {args.data_dir}", flush=True)
    print(f"Results: {args.results_dir}", flush=True)
    if args.max_hours is not None:
        print(f"Soft time limit: {args.max_hours:.2f} hours", flush=True)

    for dataset in args.datasets:
        for model in args.models:
            if deadline is not None and time.monotonic() >= deadline:
                stopped_for_time = True
                print(
                    "\n[time-limit] Stopping before the next dataset/model pair. "
                    "Completed checkpoints are safe.",
                    flush=True,
                )
                break

            if not args.no_resume and _pair_completed(
                all_metrics, args.results_dir, dataset, model
            ):
                print(f"[skip] {dataset}/{model}: checkpoint exists", flush=True)
                continue

            print(f"\n[run] {dataset}/{model}", flush=True)
            try:
                trainer = BaseModelsTrainer(
                    dataset_name=dataset,
                    data_dir=str(args.data_dir),
                    results_dir=str(args.results_dir),
                )
                pair_metrics = trainer.train_and_calibrate(model_names=[model])
                all_metrics = _replace_pair(all_metrics, pair_metrics, dataset, model)
                sort_columns = [
                    column for column in ("dataset", "model") if column in all_metrics
                ]
                if sort_columns:
                    all_metrics = all_metrics.sort_values(sort_columns).reset_index(drop=True)
                _atomic_csv(all_metrics, metrics_path)
                print(f"[ok] checkpointed {dataset}/{model}", flush=True)
            except Exception as exc:  # preserve completed pairs and continue the sweep
                error = {
                    "dataset": dataset,
                    "model": model.upper(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                _atomic_csv(pd.DataFrame(errors), errors_path)
                print(
                    f"[error] {dataset}/{model}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    raise
        if stopped_for_time:
            break

    completed_pairs = sum(
        _pair_completed(all_metrics, args.results_dir, dataset, model)
        for dataset in args.datasets
        for model in args.models
    )
    requested_pairs = len(args.datasets) * len(args.models)
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": completed_pairs == requested_pairs and not errors,
        "stopped_for_time_limit": stopped_for_time,
        "completed_pairs": completed_pairs,
        "requested_pairs": requested_pairs,
        "elapsed_hours": (time.monotonic() - started_at) / 3600,
        "error_count": len(errors),
    }
    (args.results_dir / "training_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    if not all_metrics.empty:
        print("\n=== BASELINE METRICS (TEST POOL) ===")
        print(all_metrics.to_string(index=False))
        print(f"\nSaved to: {args.results_dir}")

    if errors:
        print(
            f"\nCompleted with {len(errors)} error(s); see {errors_path}",
            file=sys.stderr,
        )
        return 1
    if errors_path.exists():
        errors_path.unlink()
    if stopped_for_time:
        print(
            f"Resume the same command to finish the remaining "
            f"{requested_pairs - completed_pairs} pair(s)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
