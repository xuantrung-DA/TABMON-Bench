"""Run AC, DOC, ATC, COT, and COTT on cached TABMON predictor streams.

The five methods consume only source-calibration labels/probabilities and
unlabeled target probabilities.  Hidden target labels are opened afterwards by
the offline evaluator solely to score the 0--1 classification-error endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.performance_estimators import (
    ENDPOINT,
    METHODS,
    PerformanceEstimatorSuite,
)
from src.cache.stream_cache import (
    BATCH_INDEX,
    KEY_COLUMNS,
    TARGET_LABEL,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
)


RESULT_SCHEMA_VERSION = 1
ERROR_FAILURE_TOLERANCE = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--limit-streams", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    if args.limit_streams is not None and args.limit_streams <= 0:
        parser.error("limit-streams must be positive")
    return args


def _load_and_validate_control(
    cache_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    status = json.loads((cache_dir / "cache_status.json").read_text(encoding="utf-8"))
    if int(status.get("cache_schema_version", -1)) != 1:
        raise ValueError("Phase 5 requires stream-cache schema version 1")
    if int(status.get("cache_revision", -1)) < 1:
        raise ValueError("Phase 5 requires repaired stream cache v1.1 or newer")
    if not status.get("complete"):
        raise ValueError("Phase 5 requires a complete stream cache")

    stream_index = pd.read_parquet(cache_dir / "control" / "stream_index.parquet")
    predictor_index = pd.read_parquet(
        cache_dir / "control" / "predictor_stream_index.parquet"
    )
    merged = predictor_index.merge(
        stream_index[
            [
                "stream_id",
                "shift",
                "severity",
                "mode",
                "seed",
                "num_batches",
                "batch_size",
            ]
        ],
        on="stream_id",
        how="left",
        validate="many_to_one",
    )
    if merged[["shift", "severity", "mode"]].isna().any().any():
        raise ValueError("Predictor index contains unknown stream IDs")
    if merged["predictor_stream_id"].duplicated().any():
        raise ValueError("Duplicate predictor-stream IDs")
    return status, merged.sort_values(
        ["dataset", "model", "seed", "shift", "severity", "mode"]
    ).reset_index(drop=True)


def _smoke_selection(index: pd.DataFrame) -> pd.DataFrame:
    # One null stream validates calibration and scoring for all 5x4
    # dataset-model pairs. Two Adult/LR stress streams additionally exercise
    # the pipeline and concept-shift failure paths.
    null_rows = (
        index[(index["shift"] == "no_shift") & (index["seed"] == 42)]
        .groupby(["dataset", "model"], sort=True, observed=True)
        .head(1)
    )
    candidates = index[(index["dataset"] == "adult") & (index["model"] == "lr")]
    selected = pd.concat(
        [
            null_rows,
            candidates[
                (candidates["shift"] == "pipeline")
                & (candidates["severity"] == "high")
                & (candidates["mode"] == "abrupt")
                & (candidates["seed"] == 42)
            ].head(1),
            candidates[
                (candidates["shift"] == "concept")
                & (candidates["severity"] == "high")
                & (candidates["mode"] == "gradual")
                & (candidates["seed"] == 42)
            ].head(1),
        ],
        ignore_index=True,
    )
    if len(null_rows) != 20 or len(selected) != 22:
        raise ValueError("Cache cannot provide the 22-stream Phase 5 smoke grid")
    return selected


def _fit_suite(
    observable: ObservableCacheReader,
    dataset: str,
    model: str,
    classes_json: str,
) -> PerformanceEstimatorSuite:
    reference_predictions = observable.load_reference_probabilities(dataset, model)
    source_labels = observable.load_source_calibration_labels(dataset)
    return PerformanceEstimatorSuite.fit(
        reference_predictions,
        source_labels,
        json.loads(classes_json),
    )


def _evaluate_predictor_stream(
    row: Any,
    suite: PerformanceEstimatorSuite,
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
) -> pd.DataFrame:
    # Monitor-facing computation happens before any target oracle is opened.
    target_predictions = observable.load_target_probabilities(
        row.predictor_stream_id
    )
    estimates_by_batch: dict[int, dict[str, float]] = {}
    for batch_index, batch in target_predictions.groupby(BATCH_INDEX, sort=True):
        estimates_by_batch[int(batch_index)] = suite.estimate(batch)

    # Offline-only evaluation starts here.
    targets = oracle.load_target_labels(row.stream_id)
    interventions = oracle.load_intervention_targets(row.stream_id)
    paired = target_predictions[KEY_COLUMNS + ["predicted_class"]].merge(
        targets[KEY_COLUMNS + [TARGET_LABEL]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(target_predictions):
        raise ValueError("Target prediction/label keys are not one-to-one")
    true_error = (
        paired.assign(
            _classification_error=(
                paired["predicted_class"] != paired[TARGET_LABEL]
            ).astype(float)
        )
        .groupby(BATCH_INDEX, sort=True)["_classification_error"]
        .mean()
    )
    shift_fraction = interventions.set_index(BATCH_INDEX)["shift_fraction"]

    records = []
    for batch_index, estimates in estimates_by_batch.items():
        observed_error = float(true_error.loc[batch_index])
        fraction = float(shift_fraction.loc[batch_index])
        for method, estimated_error in estimates.items():
            records.append(
                {
                    "predictor_stream_id": row.predictor_stream_id,
                    "stream_id": row.stream_id,
                    "dataset": row.dataset,
                    "model": row.model,
                    "shift": row.shift,
                    "severity": row.severity,
                    "mode": row.mode,
                    "seed": int(row.seed),
                    BATCH_INDEX: batch_index,
                    "shift_fraction": fraction,
                    "method": method,
                    "endpoint": ENDPOINT,
                    "source_error": suite.source_error,
                    "true_error": observed_error,
                    "true_accuracy": 1.0 - observed_error,
                    "estimated_error": estimated_error,
                    "estimated_accuracy": 1.0 - estimated_error,
                    "signed_error": estimated_error - observed_error,
                    "absolute_error": abs(estimated_error - observed_error),
                    "true_excess_error": observed_error - suite.source_error,
                    "estimated_excess_error": estimated_error - suite.source_error,
                }
            )
    result = pd.DataFrame(records)
    if set(result["method"]) != set(METHODS):
        raise RuntimeError("A Phase 5 method is missing from batch output")
    forbidden_endpoint_columns = [
        column for column in result if "log_loss" in column or column == "risk"
    ]
    if forbidden_endpoint_columns:
        raise RuntimeError(
            f"Classification endpoint mixed with log-loss fields: {forbidden_endpoint_columns}"
        )
    return result


def _scenario_summaries(batch_results: pd.DataFrame) -> pd.DataFrame:
    eligible = batch_results[
        (batch_results["shift"] == "no_shift")
        | (batch_results["shift_fraction"] > 0)
    ].copy()
    eligible["error_failure"] = (
        eligible["absolute_error"] > ERROR_FAILURE_TOLERANCE
    ).astype(float)
    eligible["harmful_batch"] = (
        eligible["true_excess_error"] > ERROR_FAILURE_TOLERANCE
    )
    eligible["harmful_sign_reversal"] = (
        eligible["harmful_batch"] & (eligible["estimated_excess_error"] < 0)
    )
    group_columns = [
        "predictor_stream_id",
        "stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
        "method",
        "endpoint",
    ]
    summary = (
        eligible.groupby(group_columns, sort=True, observed=True)
        .agg(
            mean_true_error=("true_error", "mean"),
            mean_estimated_error=("estimated_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_signed_error=("signed_error", "mean"),
            rmse=("signed_error", lambda values: float(np.sqrt(np.mean(values**2)))),
            error_failure_rate=("error_failure", "mean"),
            harmful_batches=("harmful_batch", "sum"),
            harmful_sign_reversals=("harmful_sign_reversal", "sum"),
            evaluated_batches=(BATCH_INDEX, "size"),
        )
        .reset_index()
    )
    summary["harmful_sign_reversal_rate"] = np.where(
        summary["harmful_batches"] > 0,
        summary["harmful_sign_reversals"] / summary["harmful_batches"],
        np.nan,
    )
    return summary


def _paired_comparisons(scenarios: pd.DataFrame) -> pd.DataFrame:
    wide = scenarios.pivot(
        index="predictor_stream_id",
        columns="method",
        values="mean_absolute_error",
    )
    if set(wide.columns) != set(METHODS) or wide.isna().any().any():
        raise ValueError("Paired comparison grid is incomplete")
    records = []
    for method_a, method_b in combinations(METHODS, 2):
        difference = wide[method_a] - wide[method_b]
        records.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "paired_streams": len(difference),
                "mean_mae_difference_a_minus_b": float(difference.mean()),
                "median_mae_difference_a_minus_b": float(difference.median()),
                "method_a_win_rate": float(np.mean(difference < 0)),
                "tie_rate": float(np.mean(np.isclose(difference, 0.0))),
            }
        )
    return pd.DataFrame(records)


def _write_status(
    output_dir: Path,
    requested: int,
    completed: int,
    stopped_for_time: bool,
    smoke_test: bool,
) -> None:
    atomic_json(
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "endpoint": ENDPOINT,
            "methods": list(METHODS),
            "complete": completed == requested and not stopped_for_time,
            "smoke_test": smoke_test,
            "requested_predictor_streams": requested,
            "completed_predictor_streams": completed,
            "completed_method_stream_evaluations": completed * len(METHODS),
            "stopped_for_time": stopped_for_time,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "phase5_status.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_status, index = _load_and_validate_control(cache_dir)
    if args.smoke_test:
        index = _smoke_selection(index)
    elif args.limit_streams is not None:
        index = index.head(args.limit_streams)

    observable = ObservableCacheReader(cache_dir)
    oracle = OracleCacheReader(cache_dir)
    stream_dir = output_dir / "batch_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    suites: dict[tuple[str, str], PerformanceEstimatorSuite] = {}
    calibration_records = []
    for row in index.drop_duplicates(["dataset", "model"]).itertuples(index=False):
        key = (row.dataset, row.model)
        suite = _fit_suite(observable, row.dataset, row.model, row.classes_json)
        suites[key] = suite
        calibration_records.append(
            {"dataset": row.dataset, "model": row.model, **suite.calibration_record()}
        )
    atomic_parquet(
        pd.DataFrame(calibration_records), output_dir / "calibration_summary.parquet"
    )

    started = time.monotonic()
    stopped_for_time = False
    completed = 0
    for position, row in enumerate(index.itertuples(index=False), start=1):
        result_path = stream_dir / f"{row.predictor_stream_id}.parquet"
        if not result_path.is_file():
            result = _evaluate_predictor_stream(
                row,
                suites[(row.dataset, row.model)],
                observable,
                oracle,
            )
            atomic_parquet(result, result_path)
        completed += 1
        if position == 1 or position % 25 == 0 or position == len(index):
            print(
                f"[{position}/{len(index)}] {row.dataset}/{row.model}/"
                f"{row.shift}/{row.severity}/{row.mode}/seed={row.seed}",
                flush=True,
            )
            _write_status(
                output_dir, len(index), completed, False, args.smoke_test
            )
        if args.max_hours is not None:
            elapsed_hours = (time.monotonic() - started) / 3600.0
            if elapsed_hours >= args.max_hours:
                stopped_for_time = True
                break

    result_files = [
        stream_dir / f"{row.predictor_stream_id}.parquet"
        for row in index.iloc[:completed].itertuples(index=False)
    ]
    batch_results = pd.concat(
        [pd.read_parquet(path) for path in result_files], ignore_index=True
    )
    scenarios = _scenario_summaries(batch_results)
    paired = _paired_comparisons(scenarios)
    atomic_parquet(batch_results, output_dir / "batch_metrics.parquet")
    atomic_parquet(scenarios, output_dir / "scenario_metrics.parquet")
    paired.to_csv(output_dir / "paired_method_comparisons.csv", index=False)

    expected_batch_records = completed * int(index.iloc[0]["num_batches"]) * len(METHODS)
    if len(batch_results) != expected_batch_records:
        raise RuntimeError(
            f"Expected {expected_batch_records} batch records; found {len(batch_results)}"
        )
    if len(scenarios) != completed * len(METHODS):
        raise RuntimeError("Scenario-method pairing is incomplete")

    _write_status(
        output_dir, len(index), completed, stopped_for_time, args.smoke_test
    )
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "endpoint": {
            "name": ENDPOINT,
            "loss": "0-1 classification error",
            "accuracy_is_one_minus_error": True,
            "log_loss_comparison_permitted": False,
        },
        "methods": {
            "ac": "1 - mean target maximum confidence",
            "doc": "source error - (target mean confidence - source mean confidence)",
            "atc": "source-error-calibrated upper tail of 1 - target confidence",
            "cot": "optimal L-infinity transport cost to source label distribution",
            "cott": "source-error-calibrated upper tail of individual COT costs",
        },
        "method_references": {
            "ac_doc": "Guillory et al., ICCV 2021, Predicting With Confidence on Unseen Distributions",
            "atc": "Garg et al., ICLR 2022, Leveraging Unlabeled Data to Predict Out-of-Distribution Performance",
            "cot_cott": "Lu et al., NeurIPS 2023, Characterizing Out-of-Distribution Error via Optimal Transport",
        },
        "doc_variant": (
            "unit-slope source-calibration correction; no target labels or "
            "shift-specific validation domains are used"
        ),
        "threshold_ties": (
            "deterministic fractional tie weight fitted on source calibration data"
        ),
        "monitor_inputs": [
            "labeled source-calibration predictions",
            "unlabeled target prediction probabilities",
        ],
        "forbidden_monitor_inputs": [
            "target labels",
            "target oracle log loss",
            "target classification error",
            "failure labels",
            "shift fraction",
            "intervention ground truth",
        ],
        "offline_oracle_use": "target labels used only after method estimation",
        "paired_design": {
            "shared_predictor_streams": True,
            "methods_per_stream": len(METHODS),
            "requested_predictor_streams": len(index),
        },
        "cache": {
            "schema_version": cache_status["cache_schema_version"],
            "revision": cache_status.get("cache_revision"),
            "completed_predictor_streams": cache_status[
                "completed_predictor_streams"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_json(manifest, output_dir / "phase5_manifest.json")
    print("=== TABMON PHASE 5 PERFORMANCE ESTIMATORS ===")
    print(f"Predictor streams: {completed}/{len(index)}")
    print(f"Method-stream evaluations: {len(scenarios)}")
    print(f"Batch records: {len(batch_results)}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Output: {output_dir}")
    return 2 if stopped_for_time else 0


if __name__ == "__main__":
    raise SystemExit(main())
