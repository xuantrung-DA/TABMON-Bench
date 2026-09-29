"""Run PAPE on cached TABMON binary predictor streams.

PAPE consumes labeled source-calibration data, unlabeled target features, and
target prediction probabilities.  Target labels and intervention metadata are
opened only after every estimate for a feature stream has been computed.

Density-ratio weights are shared across predictor families because they depend
only on the reference and target feature distributions.  The weighted
probability calibrator remains predictor- and batch-specific.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.pape import (
    ENDPOINT,
    METHOD,
    PAPEDensityRatioEstimator,
    PAPEClassificationErrorEstimator,
    diagnostics_record,
)
from src.cache.stream_cache import (
    BATCH_INDEX,
    KEY_COLUMNS,
    TARGET_LABEL,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
    feature_columns,
)


RESULT_SCHEMA_VERSION = 1
FAILURE_TOLERANCES = (0.02, 0.05, 0.10)
SIGN_TOLERANCE = 0.02
IN_ASSUMPTION_SHIFTS = {"no_shift", "covariate", "correlated"}
COMPARATORS = ("ac", "cott")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase5-results-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--limit-streams", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-reference-rows", type=int)
    parser.add_argument("--lightgbm-n-jobs", type=int, default=1)
    parser.add_argument("--lightgbm-n-estimators", type=int, default=100)
    args = parser.parse_args(argv)
    if args.limit_streams is not None and args.limit_streams <= 0:
        parser.error("limit-streams must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    if args.max_reference_rows is not None and args.max_reference_rows <= 1:
        parser.error("max-reference-rows must exceed one")
    if args.lightgbm_n_estimators <= 0:
        parser.error("lightgbm-n-estimators must be positive")
    return args


def _load_control(cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    status_path = cache_dir / "cache_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status.get("cache_schema_version", -1)) != 1:
        raise ValueError("PAPE Phase 1 requires stream-cache schema version 1")
    if int(status.get("cache_revision", -1)) < 1:
        raise ValueError("PAPE Phase 1 requires repaired stream cache v1.1+")
    if status.get("complete") is not True:
        raise ValueError("PAPE Phase 1 requires a complete stream cache")

    stream_index = pd.read_parquet(cache_dir / "control" / "stream_index.parquet")
    predictor_index = pd.read_parquet(
        cache_dir / "control" / "predictor_stream_index.parquet"
    )
    descriptors = [
        "stream_id",
        "shift",
        "severity",
        "mode",
        "seed",
        "num_batches",
        "batch_size",
    ]
    index = predictor_index.merge(
        stream_index[descriptors],
        on="stream_id",
        how="left",
        validate="many_to_one",
    )
    if index[descriptors[1:]].isna().any().any():
        raise ValueError("Predictor index contains unknown feature streams")
    if index["predictor_stream_id"].duplicated().any():
        raise ValueError("Duplicate predictor-stream IDs")
    return status, index.sort_values(
        ["dataset", "seed", "shift", "severity", "mode", "model"]
    ).reset_index(drop=True)


def _select_index(index: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = index.copy()
    for column, requested in (
        ("dataset", args.datasets),
        ("model", args.models),
        ("seed", args.seeds),
    ):
        if requested:
            selected = selected[selected[column].isin(requested)]
    if args.smoke_test:
        null = selected[
            selected["shift"].eq("no_shift") & selected["seed"].eq(42)
        ].groupby(["dataset", "model"], sort=True, observed=True).head(1)
        adult_lr = selected[
            selected["dataset"].eq("adult")
            & selected["model"].eq("lr")
            & selected["seed"].eq(42)
            & selected["shift"].isin(["covariate", "concept", "pipeline"])
            & selected["severity"].eq("high")
            & selected["mode"].eq("abrupt")
        ].groupby("shift", sort=True, observed=True).head(1)
        selected = pd.concat([null, adult_lr], ignore_index=True)
    if args.limit_streams is not None:
        selected = selected.head(args.limit_streams)
    if selected.empty:
        raise ValueError("PAPE selection is empty")
    return selected.sort_values(
        ["dataset", "seed", "shift", "severity", "mode", "model"]
    ).reset_index(drop=True)


def _reference_positions(size: int, limit: int | None) -> np.ndarray:
    if limit is None or size <= limit:
        return np.arange(size, dtype=int)
    return np.linspace(0, size - 1, limit, dtype=int)


def _load_reference_state(
    index: pd.DataFrame,
    observable: ObservableCacheReader,
    max_reference_rows: int | None,
    n_jobs: int,
    n_estimators: int,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[tuple[str, str], PAPEClassificationErrorEstimator],
    pd.DataFrame,
]:
    reference_features: dict[str, pd.DataFrame] = {}
    estimators: dict[tuple[str, str], PAPEClassificationErrorEstimator] = {}
    records: list[dict[str, Any]] = []
    for dataset, dataset_rows in index.groupby("dataset", sort=True):
        features = observable.load_reference_features(dataset).reset_index(drop=True)
        labels = observable.load_source_calibration_labels(dataset).reset_index(
            drop=True
        )
        if len(features) != len(labels):
            raise ValueError(f"Reference feature/label mismatch for {dataset}")
        full_reference_rows = len(features)
        positions = _reference_positions(full_reference_rows, max_reference_rows)
        features = features.iloc[positions].reset_index(drop=True)
        labels = labels.iloc[positions].reset_index(drop=True)
        reference_features[dataset] = features
        for row in dataset_rows.drop_duplicates("model").itertuples(index=False):
            predictions = observable.load_reference_probabilities(
                dataset, row.model
            ).iloc[positions].reset_index(drop=True)
            estimator = PAPEClassificationErrorEstimator(
                predictions,
                labels,
                json.loads(row.classes_json),
                n_jobs=n_jobs,
                n_estimators=n_estimators,
            )
            estimators[(dataset, row.model)] = estimator
            records.append(
                {
                    "dataset": dataset,
                    "model": row.model,
                    "full_reference_rows": full_reference_rows,
                    "used_reference_rows": len(positions),
                    "reference_subsampled": len(positions) < full_reference_rows,
                    **estimator.calibration_record(),
                }
            )
    return reference_features, estimators, pd.DataFrame(records)


def _assumption_regime(shift: str) -> str:
    return (
        "within_declared_covariate_scope"
        if shift in IN_ASSUMPTION_SHIFTS
        else "outside_declared_covariate_scope"
    )


def _evaluate_feature_stream(
    rows: pd.DataFrame,
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
    reference_features: pd.DataFrame,
    estimators: dict[tuple[str, str], PAPEClassificationErrorEstimator],
    density_estimator: PAPEDensityRatioEstimator,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    first = rows.iloc[0]
    stream_id = str(first["stream_id"])
    dataset = str(first["dataset"])
    target_features = observable.load_target_features_with_keys(stream_id)
    target_feature_names = feature_columns(target_features)
    predictions = {
        str(row.model): observable.load_target_probabilities(
            str(row.predictor_stream_id)
        )
        for row in rows.itertuples(index=False)
    }

    # All monitor-facing estimates are completed before the oracle is opened.
    estimates: dict[str, list[dict[str, Any]]] = {
        str(row.model): [] for row in rows.itertuples(index=False)
    }
    density_records: list[dict[str, Any]] = []
    for batch_index, feature_batch in target_features.groupby(
        BATCH_INDEX, sort=True
    ):
        ratio = density_estimator.estimate_weights(
            reference_features,
            feature_batch[target_feature_names].reset_index(drop=True),
        )
        density_records.append(
            {
                "stream_id": stream_id,
                "dataset": dataset,
                BATCH_INDEX: int(batch_index),
                **diagnostics_record(ratio),
            }
        )
        for row in rows.itertuples(index=False):
            model = str(row.model)
            batch_predictions = predictions[model][
                predictions[model][BATCH_INDEX].eq(batch_index)
            ].reset_index(drop=True)
            estimate = estimators[(dataset, model)].estimate(
                batch_predictions, ratio.weights
            )
            estimates[model].append(
                {
                    BATCH_INDEX: int(batch_index),
                    "estimated_error": estimate.estimated_error,
                    "estimated_accuracy": estimate.estimated_accuracy,
                    "calibrated_probability_min": (
                        estimate.calibrated_probability_min
                    ),
                    "calibrated_probability_mean": (
                        estimate.calibrated_probability_mean
                    ),
                    "calibrated_probability_max": (
                        estimate.calibrated_probability_max
                    ),
                }
            )

    # Offline-only scoring starts here.
    targets = oracle.load_target_labels(stream_id)
    interventions = oracle.load_intervention_targets(stream_id).set_index(
        BATCH_INDEX
    )
    outputs: dict[str, pd.DataFrame] = {}
    for row in rows.itertuples(index=False):
        model = str(row.model)
        estimator = estimators[(dataset, model)]
        paired = predictions[model][KEY_COLUMNS + ["predicted_class"]].merge(
            targets[KEY_COLUMNS + [TARGET_LABEL]],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(predictions[model]):
            raise ValueError("Target prediction/label keys are not one-to-one")
        true_error = (
            paired.assign(
                _error=(paired["predicted_class"] != paired[TARGET_LABEL]).astype(
                    float
                )
            )
            .groupby(BATCH_INDEX, sort=True)["_error"]
            .mean()
        )
        records: list[dict[str, Any]] = []
        for estimate in estimates[model]:
            batch_index = int(estimate[BATCH_INDEX])
            observed = float(true_error.loc[batch_index])
            estimated = float(estimate["estimated_error"])
            shift_fraction = float(
                interventions.loc[batch_index, "shift_fraction"]
            )
            records.append(
                {
                    "predictor_stream_id": row.predictor_stream_id,
                    "stream_id": stream_id,
                    "dataset": dataset,
                    "model": model,
                    "shift": row.shift,
                    "severity": row.severity,
                    "mode": row.mode,
                    "seed": int(row.seed),
                    BATCH_INDEX: batch_index,
                    "shift_fraction": shift_fraction,
                    "method": METHOD,
                    "endpoint": ENDPOINT,
                    "benchmark_assumption_regime": _assumption_regime(row.shift),
                    "source_error": estimator.source_error,
                    "true_error": observed,
                    "true_accuracy": 1.0 - observed,
                    **estimate,
                    "signed_error": estimated - observed,
                    "absolute_error": abs(estimated - observed),
                    "true_excess_error": observed - estimator.source_error,
                    "estimated_excess_error": estimated - estimator.source_error,
                }
            )
        outputs[str(row.predictor_stream_id)] = pd.DataFrame(records)
    return outputs, pd.DataFrame(density_records)


def _scenario_summaries(batch_results: pd.DataFrame) -> pd.DataFrame:
    eligible = batch_results[
        batch_results["shift"].eq("no_shift")
        | batch_results["shift_fraction"].gt(0)
    ].copy()
    for tolerance in FAILURE_TOLERANCES:
        suffix = str(tolerance).replace(".", "_")
        eligible[f"failure_tau_{suffix}"] = eligible["absolute_error"].gt(
            tolerance
        )
    eligible["sign_eligible"] = eligible["true_excess_error"].abs().gt(
        SIGN_TOLERANCE
    )
    eligible["sign_correct"] = eligible["sign_eligible"] & (
        np.sign(eligible["estimated_excess_error"])
        == np.sign(eligible["true_excess_error"])
    )
    groups = [
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
        "benchmark_assumption_regime",
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        "mean_true_error": ("true_error", "mean"),
        "mean_estimated_error": ("estimated_error", "mean"),
        "mean_absolute_error": ("absolute_error", "mean"),
        "mean_signed_error": ("signed_error", "mean"),
        "rmse": (
            "signed_error",
            lambda values: float(np.sqrt(np.mean(np.square(values)))),
        ),
        "sign_eligible_batches": ("sign_eligible", "sum"),
        "sign_correct_batches": ("sign_correct", "sum"),
        "evaluated_batches": (BATCH_INDEX, "size"),
    }
    for tolerance in FAILURE_TOLERANCES:
        suffix = str(tolerance).replace(".", "_")
        aggregations[f"failure_rate_tau_{suffix}"] = (
            f"failure_tau_{suffix}",
            "mean",
        )
    summary = eligible.groupby(groups, sort=True, observed=True).agg(
        **aggregations
    ).reset_index()
    summary["sign_accuracy"] = np.where(
        summary["sign_eligible_batches"].gt(0),
        summary["sign_correct_batches"] / summary["sign_eligible_batches"],
        np.nan,
    )
    return summary


def _paired_stream_differences(
    pape_scenarios: pd.DataFrame, phase5_dir: Path
) -> pd.DataFrame:
    phase5 = pd.read_parquet(phase5_dir / "scenario_metrics.parquet")
    phase5 = phase5[phase5["method"].isin(COMPARATORS)].copy()
    if set(phase5["method"]) != set(COMPARATORS):
        raise ValueError("Phase 5 results do not contain both AC and COTT")
    descriptors = [
        "predictor_stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
    ]
    base = pape_scenarios[descriptors + ["mean_absolute_error"]].rename(
        columns={"mean_absolute_error": "pape_mean_absolute_error"}
    )
    records = []
    for comparator in COMPARATORS:
        other = phase5[phase5["method"].eq(comparator)][
            ["predictor_stream_id", "mean_absolute_error"]
        ].rename(columns={"mean_absolute_error": "comparator_mean_absolute_error"})
        paired = base.merge(
            other, on="predictor_stream_id", how="inner", validate="one_to_one"
        )
        paired["comparator"] = comparator
        paired["pape_minus_comparator_mae"] = (
            paired["pape_mean_absolute_error"]
            - paired["comparator_mean_absolute_error"]
        )
        records.append(paired)
    return pd.concat(records, ignore_index=True)


def _write_status(
    output_dir: Path,
    requested_predictors: int,
    requested_streams: int,
    completed_predictors: int,
    completed_streams: int,
    stopped_for_time: bool,
    smoke_test: bool,
) -> None:
    atomic_json(
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "method": METHOD,
            "endpoint": ENDPOINT,
            "complete": (
                completed_predictors == requested_predictors
                and completed_streams == requested_streams
                and not stopped_for_time
            ),
            "smoke_test": smoke_test,
            "requested_predictor_streams": requested_predictors,
            "requested_feature_streams": requested_streams,
            "completed_predictor_streams": completed_predictors,
            "completed_feature_streams": completed_streams,
            "stopped_for_time": stopped_for_time,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "pape_status.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_status, index = _load_control(cache_dir)
    index = _select_index(index, args)
    observable = ObservableCacheReader(cache_dir)
    oracle = OracleCacheReader(cache_dir)

    reference_features, estimators, calibration = _load_reference_state(
        index,
        observable,
        args.max_reference_rows,
        args.lightgbm_n_jobs,
        args.lightgbm_n_estimators,
    )
    atomic_parquet(calibration, output_dir / "calibration_summary.parquet")
    density_estimator = PAPEDensityRatioEstimator(
        n_jobs=args.lightgbm_n_jobs,
        n_estimators=args.lightgbm_n_estimators,
    )

    stream_output = output_dir / "batch_streams"
    density_output = output_dir / "density_streams"
    stream_output.mkdir(parents=True, exist_ok=True)
    density_output.mkdir(parents=True, exist_ok=True)
    feature_groups = list(index.groupby("stream_id", sort=False))
    started = time.monotonic()
    stopped_for_time = False

    for position, (stream_id, rows) in enumerate(feature_groups, start=1):
        result_paths = {
            str(row.predictor_stream_id): stream_output
            / f"{row.predictor_stream_id}.parquet"
            for row in rows.itertuples(index=False)
        }
        density_path = density_output / f"{stream_id}.parquet"
        if not density_path.is_file() or not all(
            path.is_file() for path in result_paths.values()
        ):
            outputs, density = _evaluate_feature_stream(
                rows,
                observable,
                oracle,
                reference_features[str(rows.iloc[0]["dataset"])],
                estimators,
                density_estimator,
            )
            atomic_parquet(density, density_path)
            for predictor_stream_id, frame in outputs.items():
                atomic_parquet(frame, result_paths[predictor_stream_id])

        if position == 1 or position % 10 == 0 or position == len(feature_groups):
            completed_predictors = sum(
                path.is_file()
                for path in [
                    stream_output / f"{value}.parquet"
                    for value in index["predictor_stream_id"]
                ]
            )
            completed_streams = sum(
                (density_output / f"{value}.parquet").is_file()
                for value in index["stream_id"].drop_duplicates()
            )
            print(
                f"[{position}/{len(feature_groups)}] "
                f"{rows.iloc[0]['dataset']}/{rows.iloc[0]['shift']}/"
                f"seed={rows.iloc[0]['seed']} models={len(rows)}",
                flush=True,
            )
            _write_status(
                output_dir,
                len(index),
                len(feature_groups),
                completed_predictors,
                completed_streams,
                False,
                args.smoke_test,
            )
        if args.max_hours is not None and (
            (time.monotonic() - started) / 3600.0 >= args.max_hours
        ):
            stopped_for_time = True
            break

    result_files = [
        stream_output / f"{value}.parquet" for value in index["predictor_stream_id"]
    ]
    existing_results = [path for path in result_files if path.is_file()]
    density_files = [
        density_output / f"{value}.parquet"
        for value in index["stream_id"].drop_duplicates()
    ]
    existing_density = [path for path in density_files if path.is_file()]
    if not existing_results:
        raise RuntimeError("PAPE did not complete any predictor stream")
    batch_results = pd.concat(
        [pd.read_parquet(path) for path in existing_results], ignore_index=True
    )
    density_results = pd.concat(
        [pd.read_parquet(path) for path in existing_density], ignore_index=True
    )
    scenarios = _scenario_summaries(batch_results)
    atomic_parquet(batch_results, output_dir / "batch_metrics.parquet")
    atomic_parquet(scenarios, output_dir / "scenario_metrics.parquet")
    atomic_parquet(density_results, output_dir / "density_diagnostics.parquet")

    phase5_dir = (
        args.phase5_results_dir.resolve()
        if args.phase5_results_dir is not None
        else None
    )
    if phase5_dir is not None:
        paired = _paired_stream_differences(scenarios, phase5_dir)
        atomic_parquet(paired, output_dir / "paired_stream_differences.parquet")
        paired.to_csv(output_dir / "paired_stream_differences.csv", index=False)

    complete_predictors = len(existing_results)
    complete_streams = len(existing_density)
    _write_status(
        output_dir,
        len(index),
        len(feature_groups),
        complete_predictors,
        complete_streams,
        stopped_for_time,
        args.smoke_test,
    )

    try:
        import lightgbm

        lightgbm_version = lightgbm.__version__
    except ImportError:  # pragma: no cover
        lightgbm_version = "unavailable"
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "method": METHOD,
        "method_reference": (
            "Bialek et al., NeurIPS 2025, Estimating Model Performance "
            "Under Covariate Shift Without Labels"
        ),
        "implementation": "independent implementation; official code not vendored",
        "endpoint": ENDPOINT,
        "failure_tolerances": list(FAILURE_TOLERANCES),
        "sign_tolerance": SIGN_TOLERANCE,
        "declared_assumption_scope": {
            "within_scope": sorted(IN_ASSUMPTION_SHIFTS),
            "outside_scope_stress_tests": sorted(
                set(index["shift"]) - IN_ASSUMPTION_SHIFTS
            ),
            "requirements": [
                "stable P(Y|X)",
                "target support contained in reference support",
                "sufficient data for density-ratio estimation and calibration",
            ],
        },
        "density_ratio_configuration": density_estimator.configuration(),
        "reference_sampling": {
            "maximum_rows": args.max_reference_rows,
            "full_reference_is_default": True,
            "deterministic_evenly_spaced_subsample_if_limited": True,
        },
        "monitor_inputs": [
            "labeled source-calibration features and labels",
            "source-calibration prediction probabilities",
            "unlabeled target features",
            "unlabeled target prediction probabilities",
        ],
        "forbidden_monitor_inputs": [
            "target labels",
            "target oracle error",
            "failure labels",
            "shift family",
            "shift fraction",
            "intervention targets",
        ],
        "offline_oracle_use": "target labels opened only after all estimates",
        "shared_density_weights_across_predictors": True,
        "cache": {
            "schema_version": cache_status["cache_schema_version"],
            "revision": cache_status.get("cache_revision"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm_version,
        },
    }
    atomic_json(manifest, output_dir / "pape_manifest.json")
    print("=== TABMON PAPE PHASE 1 ===")
    print(f"Feature streams: {complete_streams}/{len(feature_groups)}")
    print(f"Predictor streams: {complete_predictors}/{len(index)}")
    print(f"Batch records: {len(batch_results)}")
    print(f"Output: {output_dir}")
    return 2 if stopped_for_time else 0


if __name__ == "__main__":
    raise SystemExit(main())
