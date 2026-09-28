"""Run the TABMON Step 6 common-backend DriftSHAP ablation.

The experiment reuses cached target feature streams and frozen Phase 2 RF/XGB
models.  It changes only the global feature-importance backend: explicit
TreeSHAP versus explicit permutation importance.  There is no silent fallback.
Target labels and oracle risks are never opened.  Controlled intervention
metadata is read only after monitor-facing attributions have been produced.
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

import joblib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.drift_importance import (
    IMPORTANCE_BACKENDS,
    ImportanceResult,
    compute_feature_discrepancies,
    compute_global_importance,
    weight_drift_discrepancies,
)
from src.cache.stream_cache import (
    BATCH_INDEX,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
)
from src.evaluation.metrics import compute_attribution_metrics


RESULT_SCHEMA_VERSION = 1
MODELS = ("rf", "xgb")
SHIFTS = ("no_shift", "covariate", "correlated", "pipeline", "support")
TOP_K = 3
ATTRIBUTION_FAILURE_THRESHOLD = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--importance-sample-size", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    if args.importance_sample_size <= 0:
        parser.error("importance-sample-size must be positive")
    if args.permutation_repeats <= 0:
        parser.error("permutation-repeats must be positive")
    return args


def _load_control(cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    status = json.loads((cache_dir / "cache_status.json").read_text("utf-8"))
    if int(status.get("cache_schema_version", -1)) != 1:
        raise ValueError("Step 6 requires stream-cache schema version 1")
    if int(status.get("cache_revision", -1)) < 1:
        raise ValueError("Step 6 requires repaired stream cache v1.1 or newer")
    if not status.get("complete"):
        raise ValueError("Step 6 requires a complete stream cache")

    streams = pd.read_parquet(cache_dir / "control" / "stream_index.parquet")
    predictors = pd.read_parquet(
        cache_dir / "control" / "predictor_stream_index.parquet"
    )
    streams = streams[streams["shift"].isin(SHIFTS)].copy()
    predictors = predictors[predictors["model"].isin(MODELS)].copy()
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
        how="inner",
        validate="many_to_one",
    )
    if index["predictor_stream_id"].duplicated().any():
        raise ValueError("Duplicate predictor-stream ID in Step 6 grid")
    if len(index) != 1250 or index["stream_id"].nunique() != 625:
        raise ValueError(
            "Step 6 full grid must contain 1,250 model-stream configurations "
            "over 625 shared streams"
        )
    return status, index.sort_values(
        ["dataset", "stream_id", "model"]
    ).reset_index(drop=True)


def _smoke_selection(index: pd.DataFrame) -> pd.DataFrame:
    selected = index[
        (index["dataset"] == "adult")
        & (index["seed"] == 42)
        & (
            (index["shift"] == "no_shift")
            | (
                index["shift"].isin(["covariate", "correlated", "pipeline", "support"])
                & (index["severity"] == "high")
                & (index["mode"] == "abrupt")
            )
        )
    ].copy()
    if len(selected) != 10 or set(selected["model"]) != set(MODELS):
        raise ValueError("Cache cannot provide the 10-configuration Step 6 smoke grid")
    return selected.sort_values(["stream_id", "model"]).reset_index(drop=True)


def _validate_model_environment(base_models_dir: Path) -> dict[str, Any]:
    manifest_path = base_models_dir / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
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
                f"running environment has {actual[package]}"
            )
    return {"trained": expected, "runtime": actual}


def _importance_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "importance" / f"{dataset}__{model}.parquet"


def _compute_or_load_importance(
    output_dir: Path,
    observable: ObservableCacheReader,
    base_models_dir: Path,
    dataset: str,
    model_name: str,
    sample_size: int,
    repeats: int,
    n_jobs: int,
) -> dict[str, ImportanceResult]:
    path = _importance_path(output_dir, dataset, model_name)
    reference_X = observable.load_reference_features(dataset)
    if path.is_file():
        frame = pd.read_parquet(path)
        results: dict[str, ImportanceResult] = {}
        for backend in IMPORTANCE_BACKENDS:
            subset = frame[frame["backend"] == backend].sort_values("feature_order")
            if list(subset["feature"]) != list(reference_X.columns):
                raise ValueError(f"Cached importance feature mismatch: {path}")
            results[backend] = ImportanceResult(
                backend=backend,
                raw_features=tuple(subset["feature"]),
                values=subset["importance"].to_numpy(dtype=float),
                transformed_feature_count=int(subset["transformed_feature_count"].iloc[0]),
            )
        return results

    reference_y = observable.load_source_calibration_labels(dataset)
    model_path = base_models_dir / "models" / dataset / f"{model_name}_calibrated.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    fitted_model = joblib.load(model_path)
    records: list[dict[str, Any]] = []
    results = {}
    for backend in IMPORTANCE_BACKENDS:
        started = time.monotonic()
        result = compute_global_importance(
            reference_X,
            reference_y,
            fitted_model,
            backend,
            sample_size=sample_size,
            random_seed=42,
            permutation_repeats=repeats,
            n_jobs=n_jobs,
        )
        elapsed = time.monotonic() - started
        results[backend] = result
        for order, (feature, value) in enumerate(
            zip(result.raw_features, result.values)
        ):
            records.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "backend": backend,
                    "feature": feature,
                    "feature_order": order,
                    "importance": float(value),
                    "transformed_feature_count": result.transformed_feature_count,
                    "fit_seconds": elapsed,
                }
            )
    atomic_parquet(pd.DataFrame(records), path)
    return results


def _top_features(attribution: dict[str, float], k: int = TOP_K) -> list[str]:
    return [
        feature
        for feature, _ in sorted(
            attribution.items(), key=lambda item: (-abs(item[1]), item[0])
        )[: min(k, len(attribution))]
    ]


def _evaluate_predictor_stream(
    row: Any,
    reference_X: pd.DataFrame,
    importance: dict[str, ImportanceResult],
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
) -> pd.DataFrame:
    target_X = observable.load_target_features(row.stream_id)
    expected_rows = int(row.num_batches) * int(row.batch_size)
    if len(target_X) != expected_rows:
        raise ValueError(f"Unexpected target stream length for {row.stream_id}")

    # All monitor-facing computations are completed before oracle metadata opens.
    monitor_records: list[dict[str, Any]] = []
    for batch_index in range(int(row.num_batches)):
        start = batch_index * int(row.batch_size)
        stop = start + int(row.batch_size)
        discrepancies = compute_feature_discrepancies(
            reference_X, target_X.iloc[start:stop]
        )
        for backend in IMPORTANCE_BACKENDS:
            mass, raw, normalized = weight_drift_discrepancies(
                discrepancies, importance[backend]
            )
            monitor_records.append(
                {
                    BATCH_INDEX: batch_index,
                    "backend": backend,
                    "attribution_abs_mass": mass,
                    "attribution_max_abs": float(
                        max((abs(value) for value in raw.values()), default=0.0)
                    ),
                    "predicted_top3_json": json.dumps(_top_features(raw)),
                    "raw_attribution_json": json.dumps(raw, sort_keys=True),
                    "normalized_attribution_json": json.dumps(
                        normalized, sort_keys=True
                    ),
                }
            )

    # Offline evaluation uses only generator intervention metadata, never labels.
    interventions = oracle.load_intervention_targets(row.stream_id)
    intervention_by_batch = interventions.set_index(BATCH_INDEX)
    records = []
    for monitor_record in monitor_records:
        batch_index = int(monitor_record[BATCH_INDEX])
        intervention = intervention_by_batch.loc[batch_index]
        true_attr = json.loads(intervention["ground_truth_attribution_json"])
        raw_attr = json.loads(monitor_record["raw_attribution_json"])
        active = any(abs(float(value)) > 0.0 for value in true_attr.values())
        metrics = (
            compute_attribution_metrics(true_attr, raw_attr, k=TOP_K)
            if active
            else {
                f"ndcg@{TOP_K}": np.nan,
                f"top{TOP_K}_recall": np.nan,
                "sign_accuracy": np.nan,
                "false_attribution_mass": monitor_record["attribution_abs_mass"],
            }
        )
        recall = metrics[f"top{TOP_K}_recall"]
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
                "shift_fraction": float(intervention["shift_fraction"]),
                "ground_truth_attribution_json": json.dumps(
                    true_attr, sort_keys=True
                ),
                "attribution_target_active": active,
                f"ndcg@{TOP_K}": metrics[f"ndcg@{TOP_K}"],
                f"top{TOP_K}_recall": recall,
                "false_attribution_mass": metrics["false_attribution_mass"],
                "attribution_failure": (
                    float(recall < ATTRIBUTION_FAILURE_THRESHOLD)
                    if active
                    else np.nan
                ),
                **monitor_record,
            }
        )
    return pd.DataFrame(records)


def _scenario_summaries(batch_results: pd.DataFrame) -> pd.DataFrame:
    eligible = batch_results[
        (batch_results["shift"] == "no_shift")
        | (batch_results["shift_fraction"] > 0)
    ].copy()
    group_columns = [
        "predictor_stream_id",
        "stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
        "backend",
    ]
    return (
        eligible.groupby(group_columns, sort=True, observed=True)
        .agg(
            mean_top3_recall=("top3_recall", "mean"),
            mean_ndcg_at_3=("ndcg@3", "mean"),
            attribution_failure_rate=("attribution_failure", "mean"),
            mean_attribution_abs_mass=("attribution_abs_mass", "mean"),
            mean_false_attribution_mass=("false_attribution_mass", "mean"),
            evaluated_batches=(BATCH_INDEX, "size"),
        )
        .reset_index()
    )


def _paired_backend_comparisons(scenarios: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_top3_recall",
        "mean_ndcg_at_3",
        "attribution_failure_rate",
        "mean_attribution_abs_mass",
        "mean_false_attribution_mass",
    ]
    keys = [
        "predictor_stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
    ]
    tree = scenarios[scenarios["backend"] == "tree_shap"][keys + metrics]
    perm = scenarios[scenarios["backend"] == "permutation"][keys + metrics]
    paired = tree.merge(
        perm,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_tree_shap", "_permutation"),
    )
    if len(paired) * 2 != len(scenarios):
        raise ValueError("Backend pairing is incomplete")
    for metric in metrics:
        paired[f"{metric}_tree_minus_permutation"] = (
            paired[f"{metric}_tree_shap"] - paired[f"{metric}_permutation"]
        )
    return paired


def _batch_backend_agreement(batch_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["predictor_stream_id", BATCH_INDEX]
    top = batch_results.pivot(
        index=keys, columns="backend", values="predicted_top3_json"
    ).reset_index()
    if top[list(IMPORTANCE_BACKENDS)].isna().any().any():
        raise ValueError("Batch backend agreement grid is incomplete")

    def jaccard(row: pd.Series) -> float:
        first = set(json.loads(row["tree_shap"]))
        second = set(json.loads(row["permutation"]))
        union = first | second
        return float(len(first & second) / len(union)) if union else 1.0

    top["top3_jaccard"] = top.apply(jaccard, axis=1)
    return top


def _importance_agreement(importance_frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, model), group in importance_frame.groupby(
        ["dataset", "model"], sort=True, observed=True
    ):
        wide = group.pivot(index="feature", columns="backend", values="importance")
        if wide[list(IMPORTANCE_BACKENDS)].isna().any().any():
            raise ValueError("Global importance backend grid is incomplete")
        correlation = spearmanr(
            wide["tree_shap"], wide["permutation"]
        ).statistic
        tree_top = set(wide["tree_shap"].nlargest(min(TOP_K, len(wide))).index)
        perm_top = set(wide["permutation"].nlargest(min(TOP_K, len(wide))).index)
        records.append(
            {
                "dataset": dataset,
                "model": model,
                "raw_feature_count": len(wide),
                "importance_spearman": float(correlation),
                "importance_top3_jaccard": float(
                    len(tree_top & perm_top) / len(tree_top | perm_top)
                ),
            }
        )
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
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "complete": completed == requested and not stopped_for_time,
            "smoke_test": smoke_test,
            "requested_model_stream_configurations": requested,
            "completed_model_stream_configurations": completed,
            "completed_backend_scenario_evaluations": completed
            * len(IMPORTANCE_BACKENDS),
            "stopped_for_time": stopped_for_time,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "step6_status.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir.resolve()
    base_models_dir = args.base_models_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_status, index = _load_control(cache_dir)
    if args.smoke_test:
        index = _smoke_selection(index)
    environment = _validate_model_environment(base_models_dir)
    observable = ObservableCacheReader(cache_dir)
    oracle = OracleCacheReader(cache_dir)

    started = time.monotonic()
    importance_by_pair: dict[tuple[str, str], dict[str, ImportanceResult]] = {}
    for row in index.drop_duplicates(["dataset", "model"]).itertuples(index=False):
        pair = (row.dataset, row.model)
        print(f"[importance] {row.dataset}/{row.model}", flush=True)
        importance_by_pair[pair] = _compute_or_load_importance(
            output_dir,
            observable,
            base_models_dir,
            row.dataset,
            row.model,
            args.importance_sample_size,
            args.permutation_repeats,
            args.n_jobs,
        )

    checkpoint_dir = output_dir / "batch_streams"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {
        row.predictor_stream_id: checkpoint_dir / f"{row.predictor_stream_id}.parquet"
        for row in index.itertuples(index=False)
    }
    stopped_for_time = False
    for position, row in enumerate(index.itertuples(index=False), start=1):
        result_path = expected_paths[row.predictor_stream_id]
        if not result_path.is_file():
            result = _evaluate_predictor_stream(
                row,
                observable.load_reference_features(row.dataset),
                importance_by_pair[(row.dataset, row.model)],
                observable,
                oracle,
            )
            atomic_parquet(result, result_path)
        completed = sum(path.is_file() for path in expected_paths.values())
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
            elapsed_hours = (time.monotonic() - started) / 3600.0
            if elapsed_hours >= args.max_hours:
                stopped_for_time = True
                break

    result_paths = [path for path in expected_paths.values() if path.is_file()]
    completed = len(result_paths)
    if not result_paths:
        raise RuntimeError("Step 6 did not produce any model-stream results")
    batches = pd.concat(
        [pd.read_parquet(path) for path in result_paths], ignore_index=True
    )
    scenarios = _scenario_summaries(batches)
    paired = _paired_backend_comparisons(scenarios)
    batch_agreement = _batch_backend_agreement(batches)
    importance_frame = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((output_dir / "importance").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    importance_agreement = _importance_agreement(importance_frame)

    atomic_parquet(batches, output_dir / "batch_metrics.parquet")
    atomic_parquet(scenarios, output_dir / "scenario_metrics.parquet")
    atomic_parquet(paired, output_dir / "paired_backend_comparisons.parquet")
    atomic_parquet(batch_agreement, output_dir / "batch_backend_agreement.parquet")
    atomic_parquet(importance_frame, output_dir / "global_importance.parquet")
    importance_agreement.to_csv(
        output_dir / "global_importance_agreement.csv", index=False
    )

    expected_batch_records = completed * 10 * len(IMPORTANCE_BACKENDS)
    if len(batches) != expected_batch_records:
        raise RuntimeError(
            f"Expected {expected_batch_records} batch records; found {len(batches)}"
        )
    if len(scenarios) != completed * len(IMPORTANCE_BACKENDS):
        raise RuntimeError("Step 6 backend-scenario pairing is incomplete")

    _write_status(
        output_dir,
        len(index),
        completed,
        args.smoke_test,
        stopped_for_time,
    )
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "purpose": "common-backend DriftSHAP-style attribution ablation",
        "models": list(MODELS),
        "backends": {
            "tree_shap": (
                "TreeSHAP on the unwrapped fitted tree estimator; transformed "
                "feature importance is aggregated to raw features"
            ),
            "permutation": (
                "permutation importance on the complete calibrated raw-feature "
                "pipeline using negative log loss"
            ),
        },
        "silent_backend_fallback": False,
        "shift_families": list(SHIFTS),
        "concept_shift_excluded_reason": (
            "label-only concept shift has no observable feature intervention target"
        ),
        "attribution_metrics": {
            "top_k": TOP_K,
            "failure_rule": f"Recall@{TOP_K} < {ATTRIBUTION_FAILURE_THRESHOLD}",
            "intervention_target_is_causal_ground_truth": False,
        },
        "monitor_inputs": [
            "labeled source-calibration data for global importance",
            "unlabeled cached target features",
            "frozen base model",
        ],
        "forbidden_monitor_inputs": [
            "target labels",
            "oracle risk",
            "failure labels",
            "shift fraction",
            "intervention ground truth",
        ],
        "offline_evaluation": (
            "shift fraction and intervention ground truth are opened only after "
            "attributions are computed"
        ),
        "cache": {
            "schema_version": cache_status["cache_schema_version"],
            "revision": cache_status.get("cache_revision"),
        },
        "paired_design": {
            "same_cached_stream": True,
            "same_feature_discrepancy": True,
            "only_global_importance_backend_changes": True,
            "requested_model_stream_configurations": len(index),
        },
        "importance_configuration": {
            "sample_size": args.importance_sample_size,
            "permutation_repeats": args.permutation_repeats,
            "random_seed": 42,
        },
        "environment": {
            **environment,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "shap": shap.__version__,
        },
    }
    atomic_json(manifest, output_dir / "step6_manifest.json")
    print("=== TABMON STEP 6 COMMON-BACKEND ABLATION ===")
    print(f"Model-stream configurations: {completed}/{len(index)}")
    print(f"Backend-scenario evaluations: {len(scenarios)}")
    print(f"Batch records: {len(batches)}")
    print(f"Output: {output_dir}")
    return 2 if stopped_for_time else 0


if __name__ == "__main__":
    raise SystemExit(main())
