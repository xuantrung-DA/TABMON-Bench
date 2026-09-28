"""Run the unified TABMON schema-v8 controlled benchmark from cached streams.

V8 evaluates seven label-free methods on the same 3,100 predictor streams:
Calibrated Confidence (excess log loss), DriftSHAP-style attribution with one
explicit permutation-importance backend, and AC/DOC/ATC/COT/COTT on the
classification-error endpoint.  Confidence and DriftSHAP alarms use thresholds
calibrated on maxima from independent reference-resampled null trajectories,
so alpha targets the probability of any alarm over the ten-batch stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.cached_confidence import (
    CachedConfidenceEstimator,
    probability_matrix,
)
from src.baselines.drift_importance import (
    ImportanceResult,
    PreparedDriftReference,
    compute_global_importance,
    weight_drift_discrepancies,
)
from src.baselines.performance_estimators import (
    ENDPOINT as CLASSIFICATION_ENDPOINT,
    METHODS as CLASSIFICATION_METHODS,
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
from src.evaluation.alarm_calibration import stream_max_threshold
from src.evaluation.metrics import (
    compute_alarm_event_metrics,
    compute_attribution_metrics,
)
from src.evaluation.risk import binary_log_losses


SCHEMA_VERSION = 8
CONFIDENCE_METHOD = "confidence_log_loss"
DRIFT_METHOD = "drift_shap_permutation"
METHODS = (*CLASSIFICATION_METHODS, CONFIDENCE_METHOD, DRIFT_METHOD)
RISK_EVENT_THRESHOLD = 0.05
RISK_FAILURE_TOLERANCE = 0.05
ATTRIBUTION_FAILURE_THRESHOLD = 0.5
TOP_K = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--alarm-alpha", type=float, default=0.01)
    parser.add_argument("--null-trajectories", type=int, default=200)
    parser.add_argument("--importance-sample-size", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    if not 0.0 < args.alarm_alpha < 0.5:
        parser.error("alarm-alpha must lie between zero and 0.5")
    minimum = int(np.ceil((1.0 - args.alarm_alpha) / args.alarm_alpha))
    if args.null_trajectories < minimum:
        parser.error(
            f"null-trajectories must be at least {minimum} for a finite "
            f"alpha={args.alarm_alpha} streamwise threshold"
        )
    if args.importance_sample_size <= 0 or args.permutation_repeats <= 0:
        parser.error("importance sample size and repeats must be positive")
    return args


def _stable_seed(*values: Any) -> int:
    payload = "|".join(str(value) for value in values)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _load_control(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    status = json.loads((cache_dir / "cache_status.json").read_text("utf-8"))
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text("utf-8"))
    if status.get("complete") is not True:
        raise ValueError("V8 requires a complete stream cache")
    if int(status.get("cache_schema_version", -1)) != 1:
        raise ValueError("V8 requires cache schema version 1")
    if int(status.get("cache_revision", -1)) < 1:
        raise ValueError("V8 requires repaired cache revision 1 or newer")
    streams = pd.read_parquet(cache_dir / "control" / "stream_index.parquet")
    predictors = pd.read_parquet(
        cache_dir / "control" / "predictor_stream_index.parquet"
    )
    index = predictors.merge(
        streams[
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
    if len(index) != 3100 or index["stream_id"].nunique() != 775:
        raise ValueError("V8 cache grid must contain 3,100 predictor streams")
    return status, manifest, index.sort_values(
        ["dataset", "stream_id", "model"]
    ).reset_index(drop=True)


def _smoke_selection(index: pd.DataFrame) -> pd.DataFrame:
    selected = index[
        (index["dataset"] == "adult")
        & (index["seed"] == 42)
        & (
            (index["shift"] == "no_shift")
            | (
                (index["shift"] == "pipeline")
                & (index["severity"] == "high")
                & (index["mode"] == "abrupt")
            )
            | (
                (index["shift"] == "concept")
                & (index["severity"] == "high")
                & (index["mode"] == "gradual")
            )
        )
    ].copy()
    if len(selected) != 12 or selected["model"].nunique() != 4:
        raise ValueError("Cache cannot provide the 12-stream V8 smoke grid")
    return selected.sort_values(["stream_id", "model"]).reset_index(drop=True)


def _validate_model_environment(base_models_dir: Path) -> dict[str, Any]:
    path = base_models_dir / "training_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text("utf-8"))
    expected = manifest.get("environment", {})
    actual = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "xgboost": __import__("xgboost").__version__,
    }
    for package in ("scikit_learn", "xgboost"):
        if str(expected.get(package)) != str(actual[package]):
            raise RuntimeError(
                f"Saved models require {package}={expected.get(package)}; "
                f"runtime has {actual[package]}"
            )
    return {"trained": expected, "runtime": actual}


def _top_features(attribution: dict[str, float], k: int = TOP_K) -> list[str]:
    return [
        feature
        for feature, _ in sorted(
            attribution.items(), key=lambda item: (-abs(item[1]), item[0])
        )[: min(k, len(attribution))]
    ]


def _load_or_compute_importance(
    output_dir: Path,
    reference_X: pd.DataFrame,
    reference_y: pd.Series,
    base_models_dir: Path,
    dataset: str,
    model_name: str,
    sample_size: int,
    repeats: int,
    n_jobs: int,
) -> ImportanceResult:
    path = output_dir / "importance" / f"{dataset}__{model_name}.parquet"
    if path.is_file():
        frame = pd.read_parquet(path).sort_values("feature_order")
        if list(frame["feature"]) != list(reference_X.columns):
            raise ValueError(f"Importance feature mismatch: {path}")
        return ImportanceResult(
            backend="permutation",
            raw_features=tuple(frame["feature"]),
            values=frame["importance"].to_numpy(dtype=float),
            transformed_feature_count=len(frame),
        )
    model_path = base_models_dir / "models" / dataset / f"{model_name}_calibrated.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    started = time.monotonic()
    result = compute_global_importance(
        reference_X,
        reference_y,
        model,
        "permutation",
        sample_size=sample_size,
        random_seed=42,
        permutation_repeats=repeats,
        n_jobs=n_jobs,
    )
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model_name,
            "backend": result.backend,
            "feature": result.raw_features,
            "feature_order": np.arange(len(result.raw_features)),
            "importance": result.values,
            "fit_seconds": time.monotonic() - started,
        }
    )
    atomic_parquet(frame, path)
    return result


def _fit_observable_estimators(
    observable: ObservableCacheReader,
    index: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], PerformanceEstimatorSuite],
    dict[tuple[str, str], CachedConfidenceEstimator],
]:
    performance: dict[tuple[str, str], PerformanceEstimatorSuite] = {}
    confidence: dict[tuple[str, str], CachedConfidenceEstimator] = {}
    for row in index.drop_duplicates(["dataset", "model"]).itertuples(index=False):
        key = (row.dataset, row.model)
        predictions = observable.load_reference_probabilities(row.dataset, row.model)
        labels = observable.load_source_calibration_labels(row.dataset)
        classes = json.loads(row.classes_json)
        performance[key] = PerformanceEstimatorSuite.fit(predictions, labels, classes)
        confidence[key] = CachedConfidenceEstimator.fit(predictions, labels, classes)
    return performance, confidence


def _prepare_alarm_calibration(
    output_dir: Path,
    observable: ObservableCacheReader,
    base_models_dir: Path,
    index: pd.DataFrame,
    confidence_estimators: dict[tuple[str, str], CachedConfidenceEstimator],
    null_trajectories: int,
    num_batches: int,
    batch_size: int,
    alpha: float,
    importance_sample_size: int,
    permutation_repeats: int,
    n_jobs: int,
) -> tuple[
    dict[str, PreparedDriftReference],
    dict[tuple[str, str], ImportanceResult],
    pd.DataFrame,
    pd.DataFrame,
]:
    threshold_path = output_dir / "alarm_calibration_thresholds.parquet"
    maxima_path = output_dir / "alarm_calibration_maxima.parquet"
    prepared = {
        dataset: PreparedDriftReference(observable.load_reference_features(dataset))
        for dataset in sorted(index["dataset"].unique())
    }
    importance: dict[tuple[str, str], ImportanceResult] = {}
    for row in index.drop_duplicates(["dataset", "model"]).itertuples(index=False):
        reference_X = prepared[row.dataset].reference_X
        reference_y = observable.load_source_calibration_labels(row.dataset)
        importance[(row.dataset, row.model)] = _load_or_compute_importance(
            output_dir,
            reference_X,
            reference_y,
            base_models_dir,
            row.dataset,
            row.model,
            importance_sample_size,
            permutation_repeats,
            n_jobs,
        )

    if threshold_path.is_file() and maxima_path.is_file():
        thresholds = pd.read_parquet(threshold_path)
        maxima = pd.read_parquet(maxima_path)
        expected_pairs = len(index.drop_duplicates(["dataset", "model"])) * 2
        if len(thresholds) != expected_pairs:
            raise ValueError("Persisted alarm threshold grid is incomplete")
        expected_methods = {CONFIDENCE_METHOD, DRIFT_METHOD}
        if set(thresholds["method"]) != expected_methods:
            raise ValueError("Persisted alarm thresholds contain wrong methods")
        for column, expected in (
            ("alpha", alpha),
            ("null_trajectories", null_trajectories),
            ("num_batches", num_batches),
            ("batch_size", batch_size),
        ):
            values = thresholds[column].unique()
            if len(values) != 1 or not np.isclose(float(values[0]), float(expected)):
                raise ValueError(
                    f"Persisted alarm calibration has {column}={values}; "
                    f"requested {expected}. Use a new output directory."
                )
        expected_maxima = expected_pairs * null_trajectories
        if len(maxima) != expected_maxima:
            raise ValueError(
                "Persisted alarm maxima are incomplete: "
                f"expected {expected_maxima}, found {len(maxima)}"
            )
        return prepared, importance, thresholds, maxima

    maximum_records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    total_batches = null_trajectories * num_batches
    for dataset in sorted(index["dataset"].unique()):
        reference = prepared[dataset]
        rng = np.random.default_rng(_stable_seed("v8-null", dataset))
        positions = rng.integers(
            0,
            len(reference.reference_X),
            size=(total_batches, batch_size),
            dtype=np.int32,
        )
        print(
            f"[null calibration] {dataset}: {null_trajectories} trajectories",
            flush=True,
        )
        discrepancy_matrix = reference.score_position_batches(positions)
        models = sorted(index.loc[index["dataset"] == dataset, "model"].unique())
        for model_name in models:
            pair = (dataset, model_name)
            confidence_scores = confidence_estimators[pair].score_position_batches(
                positions
            )
            drift_scores = discrepancy_matrix @ importance[pair].values
            for monitor, batch_scores in (
                (CONFIDENCE_METHOD, confidence_scores),
                (DRIFT_METHOD, drift_scores),
            ):
                threshold, maxima, rank = stream_max_threshold(
                    batch_scores,
                    null_trajectories,
                    num_batches,
                    alpha,
                )
                threshold_records.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "method": monitor,
                        "alpha": alpha,
                        "threshold": threshold,
                        "conformal_rank": rank,
                        "null_trajectories": null_trajectories,
                        "num_batches": num_batches,
                        "batch_size": batch_size,
                    }
                )
                for trajectory_index, maximum in enumerate(maxima):
                    maximum_records.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "method": monitor,
                            "trajectory_index": trajectory_index,
                            "maximum_score": float(maximum),
                        }
                    )
    thresholds = pd.DataFrame(threshold_records)
    maxima = pd.DataFrame(maximum_records)
    atomic_parquet(thresholds, threshold_path)
    atomic_parquet(maxima, maxima_path)
    return prepared, importance, thresholds, maxima


def _prediction_diagnostics(predictions: pd.DataFrame) -> tuple[float, float]:
    probabilities = np.clip(probability_matrix(predictions), 1e-12, 1.0)
    confidence = float(np.max(probabilities, axis=1).mean())
    entropy = float((-np.sum(probabilities * np.log(probabilities), axis=1)).mean())
    return confidence, entropy


def _evaluate_predictor_stream(
    row: Any,
    performance: PerformanceEstimatorSuite,
    confidence: CachedConfidenceEstimator,
    prepared_reference: PreparedDriftReference,
    importance: ImportanceResult,
    threshold_by_method: dict[str, float],
    maxima_by_method: dict[str, np.ndarray],
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
) -> pd.DataFrame:
    predictions = observable.load_target_probabilities(row.predictor_stream_id)
    target_X = observable.load_target_features(row.stream_id)
    expected_rows = int(row.num_batches) * int(row.batch_size)
    if len(predictions) != expected_rows or len(target_X) != expected_rows:
        raise ValueError("Cached target stream length is inconsistent")

    # Complete all monitor-facing computation before opening any oracle table.
    monitor_batches: dict[int, dict[str, Any]] = {}
    for batch_index in range(int(row.num_batches)):
        start = batch_index * int(row.batch_size)
        stop = start + int(row.batch_size)
        prediction_batch = predictions.iloc[start:stop]
        feature_batch = target_X.iloc[start:stop]
        classification_estimates = performance.estimate(prediction_batch)
        confidence_score = confidence.estimate_excess_log_loss(prediction_batch)
        discrepancies = prepared_reference.score_batch(feature_batch)
        drift_mass, raw_attr, normalized_attr = weight_drift_discrepancies(
            discrepancies, importance
        )
        prediction_confidence, prediction_entropy = _prediction_diagnostics(
            prediction_batch
        )
        monitor_batches[batch_index] = {
            "classification_estimates": classification_estimates,
            "confidence_score": confidence_score,
            "drift_score": drift_mass,
            "raw_attribution": raw_attr,
            "normalized_attribution": normalized_attr,
            "prediction_confidence": prediction_confidence,
            "prediction_entropy": prediction_entropy,
        }

    # Offline evaluation begins here.
    targets = oracle.load_target_labels(row.stream_id)
    interventions = oracle.load_intervention_targets(row.stream_id).set_index(
        BATCH_INDEX
    )
    paired = predictions.merge(
        targets[KEY_COLUMNS + [TARGET_LABEL]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    true_error = (
        paired.assign(
            _classification_error=(
                paired["predicted_class"] != paired[TARGET_LABEL]
            ).astype(float)
        )
        .groupby(BATCH_INDEX)["_classification_error"]
        .mean()
    )
    paired_losses = binary_log_losses(
        probability_matrix(paired),
        confidence.classes,
        paired[TARGET_LABEL].to_numpy(),
    )
    risk_working = paired[[BATCH_INDEX]].copy()
    risk_working["_sample_log_loss"] = paired_losses
    true_excess_risk_by_batch = (
        risk_working.groupby(BATCH_INDEX)["_sample_log_loss"].mean()
        - confidence.reference_observed_risk
    )

    base = {
        "predictor_stream_id": row.predictor_stream_id,
        "stream_id": row.stream_id,
        "dataset": row.dataset,
        "model": row.model,
        "shift": row.shift,
        "severity": row.severity,
        "mode": row.mode,
        "seed": int(row.seed),
    }
    records: list[dict[str, Any]] = []
    for batch_index, monitor in monitor_batches.items():
        fraction = float(interventions.loc[batch_index, "shift_fraction"])
        true_excess_risk = float(true_excess_risk_by_batch.loc[batch_index])
        observed_error = float(true_error.loc[batch_index])
        true_risk_event = true_excess_risk > RISK_EVENT_THRESHOLD
        shift_event = row.shift != "no_shift" and fraction > 0.0
        common = {
            **base,
            BATCH_INDEX: batch_index,
            "shift_fraction": fraction,
            "prediction_confidence": monitor["prediction_confidence"],
            "prediction_entropy": monitor["prediction_entropy"],
        }
        for method, estimate in monitor["classification_estimates"].items():
            records.append(
                {
                    **common,
                    "method": method,
                    "endpoint": CLASSIFICATION_ENDPOINT,
                    "supports_risk_estimation": True,
                    "supports_attribution": False,
                    "supports_alarm": False,
                    "source_value": performance.source_error,
                    "true_value": observed_error,
                    "estimated_value": float(estimate),
                    "signed_error": float(estimate - observed_error),
                    "absolute_error": float(abs(estimate - observed_error)),
                    "risk_failure": float(
                        abs(estimate - observed_error) > RISK_FAILURE_TOLERANCE
                    ),
                    "monitor_score": float(estimate - performance.source_error),
                    "alarm": pd.NA,
                    "alarm_threshold": np.nan,
                    "alarm_p_value": np.nan,
                    "alarm_target": None,
                    "alarm_target_event": pd.NA,
                    "attribution_target_active": False,
                    "top3_recall": np.nan,
                    "ndcg@3": np.nan,
                    "attribution_failure": np.nan,
                    "attribution_abs_mass": np.nan,
                    "false_attribution_mass": np.nan,
                    "predicted_attribution_json": None,
                }
            )

        confidence_score = float(monitor["confidence_score"])
        confidence_threshold = threshold_by_method[CONFIDENCE_METHOD]
        confidence_maxima = maxima_by_method[CONFIDENCE_METHOD]
        confidence_alarm = confidence_score > confidence_threshold
        records.append(
            {
                **common,
                "method": CONFIDENCE_METHOD,
                "endpoint": "excess_log_loss",
                "supports_risk_estimation": True,
                "supports_attribution": False,
                "supports_alarm": True,
                "source_value": 0.0,
                "true_value": true_excess_risk,
                "estimated_value": confidence_score,
                "signed_error": confidence_score - true_excess_risk,
                "absolute_error": abs(confidence_score - true_excess_risk),
                "risk_failure": float(
                    abs(confidence_score - true_excess_risk)
                    > RISK_FAILURE_TOLERANCE
                ),
                "monitor_score": confidence_score,
                "alarm": confidence_alarm,
                "alarm_threshold": confidence_threshold,
                "alarm_p_value": float(
                    (1 + np.sum(confidence_maxima >= confidence_score))
                    / (len(confidence_maxima) + 1)
                ),
                "alarm_target": "risk_event",
                "alarm_target_event": true_risk_event,
                "attribution_target_active": False,
                "top3_recall": np.nan,
                "ndcg@3": np.nan,
                "attribution_failure": np.nan,
                "attribution_abs_mass": np.nan,
                "false_attribution_mass": np.nan,
                "predicted_attribution_json": None,
            }
        )

        true_attr = json.loads(
            interventions.loc[batch_index, "ground_truth_attribution_json"]
        )
        active = any(abs(float(value)) > 0.0 for value in true_attr.values())
        attr_metrics = (
            compute_attribution_metrics(
                true_attr, monitor["raw_attribution"], k=TOP_K
            )
            if active
            else {
                "top3_recall": np.nan,
                "ndcg@3": np.nan,
                "false_attribution_mass": monitor["drift_score"],
            }
        )
        drift_score = float(monitor["drift_score"])
        drift_threshold = threshold_by_method[DRIFT_METHOD]
        drift_maxima = maxima_by_method[DRIFT_METHOD]
        drift_alarm = drift_score > drift_threshold
        recall = attr_metrics["top3_recall"]
        records.append(
            {
                **common,
                "method": DRIFT_METHOD,
                "endpoint": "feature_intervention_attribution",
                "supports_risk_estimation": False,
                "supports_attribution": True,
                "supports_alarm": True,
                "source_value": np.nan,
                "true_value": np.nan,
                "estimated_value": np.nan,
                "signed_error": np.nan,
                "absolute_error": np.nan,
                "risk_failure": np.nan,
                "monitor_score": drift_score,
                "alarm": drift_alarm,
                "alarm_threshold": drift_threshold,
                "alarm_p_value": float(
                    (1 + np.sum(drift_maxima >= drift_score))
                    / (len(drift_maxima) + 1)
                ),
                "alarm_target": "distribution_shift",
                "alarm_target_event": shift_event,
                "attribution_target_active": active,
                "top3_recall": recall,
                "ndcg@3": attr_metrics["ndcg@3"],
                "attribution_failure": (
                    float(recall < ATTRIBUTION_FAILURE_THRESHOLD)
                    if active
                    else np.nan
                ),
                "attribution_abs_mass": drift_score,
                "false_attribution_mass": attr_metrics[
                    "false_attribution_mass"
                ],
                "predicted_attribution_json": json.dumps(
                    monitor["normalized_attribution"], sort_keys=True
                ),
            }
        )
    result = pd.DataFrame(records)
    if set(result["method"]) != set(METHODS):
        raise RuntimeError("A V8 method is missing from predictor-stream output")
    return result


def _scenario_summaries(batches: pd.DataFrame) -> pd.DataFrame:
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
    records = []
    for key, group in batches.groupby(group_columns, sort=True, observed=True):
        group = group.sort_values(BATCH_INDEX)
        eligible = group[
            (group["shift"] == "no_shift") | (group["shift_fraction"] > 0)
        ]
        row = {
            **dict(zip(group_columns, key)),
            "evaluated_batches": len(eligible),
            "mean_true_value": pd.to_numeric(
                eligible["true_value"], errors="coerce"
            ).mean(),
            "mean_estimated_value": pd.to_numeric(
                eligible["estimated_value"], errors="coerce"
            ).mean(),
            "mean_absolute_error": pd.to_numeric(
                eligible["absolute_error"], errors="coerce"
            ).mean(),
            "mean_signed_error": pd.to_numeric(
                eligible["signed_error"], errors="coerce"
            ).mean(),
            "risk_failure_rate": pd.to_numeric(
                eligible["risk_failure"], errors="coerce"
            ).mean(),
            "mean_top3_recall": pd.to_numeric(
                eligible["top3_recall"], errors="coerce"
            ).mean(),
            "mean_ndcg_at_3": pd.to_numeric(
                eligible["ndcg@3"], errors="coerce"
            ).mean(),
            "attribution_failure_rate": pd.to_numeric(
                eligible["attribution_failure"], errors="coerce"
            ).mean(),
            "mean_attribution_abs_mass": pd.to_numeric(
                eligible["attribution_abs_mass"], errors="coerce"
            ).mean(),
        }
        if bool(group["supports_alarm"].iloc[0]):
            metrics = compute_alarm_event_metrics(
                group["alarm"].astype(bool).to_numpy(),
                group["alarm_target_event"].astype(bool).to_numpy(),
            )
            event_exists = np.isfinite(metrics["event_batch"])
            row.update(
                {
                    **metrics,
                    "event_exists": event_exists,
                    "detected_event": bool(
                        event_exists and not bool(metrics["missed_alarm"])
                    ),
                    "null_stream_any_alarm": bool(
                        group["shift"].iloc[0] == "no_shift"
                        and group["alarm"].astype(bool).any()
                    ),
                }
            )
        else:
            row.update(
                {
                    "event_batch": np.nan,
                    "first_alarm_batch": np.nan,
                    "false_alarm_rate": np.nan,
                    "power": np.nan,
                    "detection_delay": np.nan,
                    "missed_alarm": np.nan,
                    "event_exists": False,
                    "detected_event": False,
                    "null_stream_any_alarm": False,
                }
            )
        records.append(row)
    return pd.DataFrame(records)


def _write_status(
    output_dir: Path,
    requested: int,
    completed: int,
    smoke_test: bool,
    stopped_for_time: bool,
) -> None:
    atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "complete": completed == requested and not stopped_for_time,
            "smoke_test": smoke_test,
            "requested_predictor_streams": requested,
            "completed_predictor_streams": completed,
            "completed_method_stream_evaluations": completed * len(METHODS),
            "stopped_for_time": stopped_for_time,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "v8_status.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    base_models_dir = args.base_models_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    cache_status, cache_manifest, index = _load_control(cache_dir)
    if args.smoke_test:
        index = _smoke_selection(index)
    environment = _validate_model_environment(base_models_dir)
    observable = ObservableCacheReader(cache_dir)
    oracle = OracleCacheReader(cache_dir)
    performance, confidence = _fit_observable_estimators(observable, index)
    num_batches = int(index.iloc[0]["num_batches"])
    batch_size = int(index.iloc[0]["batch_size"])
    prepared, importance, thresholds, maxima = _prepare_alarm_calibration(
        output_dir,
        observable,
        base_models_dir,
        index,
        confidence,
        args.null_trajectories,
        num_batches,
        batch_size,
        args.alarm_alpha,
        args.importance_sample_size,
        args.permutation_repeats,
        args.n_jobs,
    )
    threshold_lookup = thresholds.set_index(["dataset", "model", "method"])[
        "threshold"
    ].to_dict()
    maxima_lookup = {
        key: group.sort_values("trajectory_index")["maximum_score"].to_numpy()
        for key, group in maxima.groupby(["dataset", "model", "method"])
    }

    checkpoint_dir = output_dir / "batch_streams"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {
        row.predictor_stream_id: checkpoint_dir / f"{row.predictor_stream_id}.parquet"
        for row in index.itertuples(index=False)
    }
    stopped_for_time = False
    for position, row in enumerate(index.itertuples(index=False), start=1):
        path = expected_paths[row.predictor_stream_id]
        if not path.is_file():
            pair = (row.dataset, row.model)
            threshold_by_method = {
                method: float(threshold_lookup[(row.dataset, row.model, method)])
                for method in (CONFIDENCE_METHOD, DRIFT_METHOD)
            }
            maxima_by_method = {
                method: maxima_lookup[(row.dataset, row.model, method)]
                for method in (CONFIDENCE_METHOD, DRIFT_METHOD)
            }
            result = _evaluate_predictor_stream(
                row,
                performance[pair],
                confidence[pair],
                prepared[row.dataset],
                importance[pair],
                threshold_by_method,
                maxima_by_method,
                observable,
                oracle,
            )
            atomic_parquet(result, path)
        completed = sum(item.is_file() for item in expected_paths.values())
        if position == 1 or position % 25 == 0 or position == len(index):
            print(
                f"[{position}/{len(index)}] {row.dataset}/{row.model}/"
                f"{row.shift}/{row.severity}/{row.mode}/seed={row.seed}",
                flush=True,
            )
            _write_status(
                output_dir, len(index), completed, args.smoke_test, False
            )
        if args.max_hours is not None:
            if (time.monotonic() - started) / 3600.0 >= args.max_hours:
                stopped_for_time = True
                break

    paths = [path for path in expected_paths.values() if path.is_file()]
    completed = len(paths)
    if not paths:
        raise RuntimeError("V8 did not complete any predictor stream")
    batches = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    scenarios = _scenario_summaries(batches)
    atomic_parquet(batches, output_dir / "batch_metrics.parquet")
    atomic_parquet(scenarios, output_dir / "scenario_metrics.parquet")

    if len(batches) != completed * num_batches * len(METHODS):
        raise RuntimeError("V8 batch-method grid is incomplete")
    if len(scenarios) != completed * len(METHODS):
        raise RuntimeError("V8 scenario-method grid is incomplete")
    _write_status(
        output_dir,
        len(index),
        completed,
        args.smoke_test,
        stopped_for_time,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "methods": {
            "classification_error": list(CLASSIFICATION_METHODS),
            "excess_log_loss": [CONFIDENCE_METHOD],
            "feature_attribution": [DRIFT_METHOD],
        },
        "endpoint_separation": {
            "classification_error_estimators_compared_only_with_0_1_error": True,
            "confidence_compared_only_with_excess_log_loss": True,
            "cross_endpoint_ranking_prohibited": True,
        },
        "drift_attribution": {
            "global_importance_backend": "permutation",
            "silent_backend_fallback": False,
            "tree_shap_results_reserved_for_step6_ablation": True,
        },
        "alarm_calibration": {
            "target": "probability of any alarm over the complete stream",
            "alpha": args.alarm_alpha,
            "null_trajectories_per_dataset_model_monitor": args.null_trajectories,
            "num_batches_per_trajectory": num_batches,
            "threshold_statistic": "maximum batch score in each null trajectory",
            "threshold_rule": "strict upper-tail split-conformal order statistic",
            "source": "reference calibration data only",
            "evaluation_null_streams_used_for_thresholds": False,
        },
        "monitor_inputs": [
            "source calibration labels",
            "source calibration features and predictions",
            "unlabeled target features and predictions",
            "frozen base model for permutation importance",
        ],
        "forbidden_monitor_inputs": [
            "target labels",
            "oracle risk",
            "failure labels",
            "shift fraction",
            "intervention ground truth",
        ],
        "offline_evaluation": (
            "target labels, oracle risk, event flags, and intervention targets "
            "are opened only after monitor outputs are fixed"
        ),
        "configuration": {
            "predictor_streams": len(index),
            "methods_per_stream": len(METHODS),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "risk_event_threshold": RISK_EVENT_THRESHOLD,
            "risk_failure_tolerance": RISK_FAILURE_TOLERANCE,
            "attribution_failure_threshold": ATTRIBUTION_FAILURE_THRESHOLD,
            "top_k": TOP_K,
            "importance_sample_size": args.importance_sample_size,
            "permutation_repeats": args.permutation_repeats,
        },
        "cache": {
            "schema_version": cache_status["cache_schema_version"],
            "revision": cache_status.get("cache_revision"),
            "purpose": cache_manifest.get("purpose"),
        },
        "environment": {
            **environment,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "shap": shap.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest, output_dir / "v8_manifest.json")
    print("=== TABMON SCHEMA V8 CONTROLLED RUN ===")
    print(f"Predictor streams: {completed}/{len(index)}")
    print(f"Method-stream evaluations: {len(scenarios)}")
    print(f"Batch-method records: {len(batches)}")
    print(f"Output: {output_dir}")
    return 2 if stopped_for_time else 0


if __name__ == "__main__":
    raise SystemExit(main())
