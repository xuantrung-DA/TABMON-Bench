"""Run the Step-12 Q1 extension on a complete reusable stream cache.

The full grid evaluates AC, DOC, ATC, COT, COTT, calibrated Confidence, and
the published SHD quantile detector with a PM-EB confidence sequence. A
predeclared, compute-bounded subset additionally evaluates equal-mass EMD XPE.
Classification error and log-loss endpoints remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
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
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.cached_confidence import CachedConfidenceEstimator
from src.baselines.performance_estimators import (
    METHODS as PERFORMANCE_METHODS,
    MULTICLASS_COTT_CALIBRATION_LIMIT,
    PerformanceEstimatorSuite,
)
from src.baselines.shd import METHOD as SHD_METHOD, SHDMonitor
from src.baselines.xpe import XPEExplainer
from src.cache.stream_cache import (
    BATCH_INDEX,
    TARGET_LABEL,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
)
from src.evaluation.metrics import (
    compute_alarm_event_metrics,
    compute_attribution_metrics,
)


SCHEMA_VERSION = 13
CONFIDENCE_METHOD = "confidence_log_loss"
GRID_METHODS = (*PERFORMANCE_METHODS, CONFIDENCE_METHOD, SHD_METHOD)
RISK_TOLERANCE = 0.05
HARM_THRESHOLD = 0.05
XPE_MODELS = {"lr", "xgb"}
XPE_SHIFTS = {"no_shift", "covariate", "correlated", "pipeline", "support"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-xpe", action="store_true")
    parser.add_argument("--xpe-sample-size", type=int, default=50)
    parser.add_argument(
        "--xpe-backend",
        choices=["grouped_permutation", "kernel_shap"],
        default="grouped_permutation",
    )
    parser.add_argument("--xpe-permutations", type=int, default=128)
    parser.add_argument("--xpe-kernel-nsamples", type=int, default=3000)
    parser.add_argument("--shd-alpha", type=float, default=0.01)
    parser.add_argument("--shd-epsilon", type=float, default=0.02)
    parser.add_argument("--allow-model-version-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    if args.xpe_sample_size < 2:
        parser.error("XPE sample size must be at least two")
    if args.xpe_permutations < 1:
        parser.error("xpe-permutations must be positive")
    if args.xpe_kernel_nsamples < 1:
        parser.error("xpe-kernel-nsamples must be positive")
    return args


def _validate_model_environment(base_models_dir: Path, allow_mismatch: bool) -> dict:
    path = base_models_dir / "training_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text("utf-8"))
    trained = manifest.get("environment", {})
    actual = {"python": platform.python_version(), "scikit_learn": sklearn.__version__}
    expected_sklearn = str(trained.get("scikit_learn"))
    if expected_sklearn not in {"", "None", actual["scikit_learn"]} and not allow_mismatch:
        raise RuntimeError(
            f"Models require scikit-learn={expected_sklearn}; runtime has "
            f"{actual['scikit_learn']}"
        )
    return {"trained": trained, "runtime": actual, "mismatch_allowed": allow_mismatch}


def _stable_seed(*values: Any) -> int:
    payload = "|".join(str(value) for value in values)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _load_index(cache_dir: Path, expected_seeds: list[int] | None) -> pd.DataFrame:
    status = json.loads((cache_dir / "cache_status.json").read_text("utf-8"))
    if status.get("complete") is not True:
        raise ValueError("Step 12 requires a complete cache")
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
        validate="many_to_one",
    )
    seeds = sorted(int(value) for value in index["seed"].unique())
    if expected_seeds is not None and seeds != sorted(expected_seeds):
        raise ValueError(f"Cache seeds {seeds} != requested seeds {expected_seeds}")
    expected_configs = 31
    group_counts = index.groupby(["dataset", "model", "seed"]).size()
    if not (group_counts == expected_configs).all():
        raise ValueError("Every dataset/model/seed cell must contain 31 configurations")
    return index.sort_values(
        ["dataset", "model", "seed", "shift", "severity", "mode"]
    ).reset_index(drop=True)


def _smoke_index(index: pd.DataFrame) -> pd.DataFrame:
    first_dataset = str(index.iloc[0]["dataset"])
    first_seed = int(index[index["dataset"] == first_dataset]["seed"].min())
    chosen = index[
        (index["dataset"] == first_dataset)
        & (index["model"] == "lr")
        & (index["seed"] == first_seed)
        & (
            (index["shift"] == "no_shift")
            | (
                (index["shift"] == "pipeline")
                & (index["severity"] == "high")
                & (index["mode"] == "abrupt")
            )
        )
    ]
    if len(chosen) != 2:
        raise ValueError("Could not construct Step-12 smoke grid")
    return chosen.reset_index(drop=True)


def _reference_estimators(
    observable: ObservableCacheReader,
    index: pd.DataFrame,
    shd_alpha: float,
    shd_epsilon: float,
) -> tuple[dict, dict, dict, dict]:
    performance: dict[tuple[str, str], PerformanceEstimatorSuite] = {}
    confidence: dict[tuple[str, str], CachedConfidenceEstimator] = {}
    shd: dict[tuple[str, str], SHDMonitor] = {}
    reference_error: dict[tuple[str, str], float] = {}
    for row in index.drop_duplicates(["dataset", "model"]).itertuples(index=False):
        key = (row.dataset, row.model)
        predictions = observable.load_reference_probabilities(row.dataset, row.model)
        reference_X = observable.load_reference_features(row.dataset)
        labels = observable.load_source_calibration_labels(row.dataset)
        classes = json.loads(row.classes_json)
        performance[key] = PerformanceEstimatorSuite.fit(predictions, labels, classes)
        confidence[key] = CachedConfidenceEstimator.fit(predictions, labels, classes)
        shd[key] = SHDMonitor.fit(
            reference_X,
            predictions,
            labels,
            classes,
            random_seed=_stable_seed("shd", *key),
            alpha=shd_alpha,
            epsilon=shd_epsilon,
        )
        reference_error[key] = performance[key].source_error
    return performance, confidence, shd, reference_error


def _true_batch_error(
    predictions: pd.DataFrame, target_labels: pd.DataFrame
) -> pd.DataFrame:
    merged = predictions.merge(
        target_labels,
        on=["_tabmon_batch_index", "_tabmon_row_index"],
        validate="one_to_one",
    )
    merged["sample_error"] = (
        merged["predicted_class"].to_numpy()
        != merged[TARGET_LABEL].to_numpy()
    ).astype(float)
    return (
        merged.groupby(BATCH_INDEX, sort=True)["sample_error"]
        .mean()
        .rename("target_error")
        .reset_index()
    )


def _evaluate_grid_stream(
    row: Any,
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
    performance: PerformanceEstimatorSuite,
    confidence: CachedConfidenceEstimator,
    shd: SHDMonitor,
    source_error: float,
) -> pd.DataFrame:
    predictions = observable.load_target_probabilities(row.predictor_stream_id)
    predictions = predictions.sort_values(
        ["_tabmon_batch_index", "_tabmon_row_index"]
    ).reset_index(drop=True)
    targets = oracle.load_target_labels(row.stream_id)
    true_errors = _true_batch_error(predictions, targets)
    true_errors["true_excess_error"] = true_errors["target_error"] - source_error
    oracle_risk = oracle.load_batch_risk(row.predictor_stream_id)
    interventions = oracle.load_intervention_targets(row.stream_id)[
        [BATCH_INDEX, "shift_fraction"]
    ]
    base = true_errors.merge(oracle_risk, on=BATCH_INDEX, validate="one_to_one").merge(
        interventions, on=BATCH_INDEX, validate="one_to_one"
    )
    common = {
        "predictor_stream_id": row.predictor_stream_id,
        "stream_id": row.stream_id,
        "dataset": row.dataset,
        "model": row.model,
        "shift": row.shift,
        "severity": row.severity,
        "mode": row.mode,
        "seed": row.seed,
    }
    records: list[dict[str, Any]] = []
    target_X = observable.load_target_features(row.stream_id).reset_index(drop=True)
    if len(target_X) != len(predictions):
        raise ValueError("Target features and predictions are not aligned")
    shd_rows = shd.monitor(target_X)
    sample_errors = predictions.merge(
        targets,
        on=["_tabmon_batch_index", "_tabmon_row_index"],
        validate="one_to_one",
    ).sort_values(["_tabmon_batch_index", "_tabmon_row_index"])
    sample_errors = (
        sample_errors["predicted_class"].to_numpy()
        != sample_errors[TARGET_LABEL].to_numpy()
    ).astype(float)
    true_high_error = sample_errors > shd.calibration.true_error_threshold
    true_selected_high = (
        true_high_error & shd_rows["high_error_selector"].to_numpy(dtype=bool)
    ).astype(float)
    true_false_discovery = (
        ~true_high_error & shd_rows["high_error_selector"].to_numpy(dtype=bool)
    ).astype(float)
    shd_rows["oracle_running_selected_high_error"] = np.cumsum(
        true_selected_high
    ) / np.arange(1, len(true_selected_high) + 1)
    shd_rows["oracle_phi_q2_event"] = (
        shd_rows["oracle_running_selected_high_error"]
        > shd.calibration.source_selected_high_error_rate
        + shd.calibration.epsilon
    )
    shd_rows["oracle_assumption_4_1_gap"] = (
        np.cumsum(true_false_discovery)
        / np.arange(1, len(true_false_discovery) + 1)
        - shd.calibration.source_false_discovery_joint_rate
    )
    shd_rows[BATCH_INDEX] = predictions[BATCH_INDEX].to_numpy()
    shd_batches = shd_rows.groupby(BATCH_INDEX, sort=True).agg(
        estimated_high_error_rate=("high_error_selector", "mean"),
        estimated_high_error_lower=("estimated_high_error_lower", "last"),
        alarm=("alarm", "last"),
        alarm_phi_q=("alarm_phi_q", "last"),
        mean_predicted_error=("predicted_error", "mean"),
        oracle_selected_high_error=("oracle_running_selected_high_error", "last"),
        oracle_phi_q2_event=("oracle_phi_q2_event", "last"),
        oracle_assumption_4_1_gap=("oracle_assumption_4_1_gap", "last"),
    )
    shd_batches = shd_batches.reset_index()

    for _, batch in base.iterrows():
        batch_index = int(batch[BATCH_INDEX])
        subset = predictions[predictions[BATCH_INDEX] == batch_index]
        estimates = performance.estimate(subset)
        for method, estimate in estimates.items():
            signed = float(estimate - batch["target_error"])
            records.append(
                {
                    **common,
                    BATCH_INDEX: batch_index,
                    "method": method,
                    "endpoint": "classification_error",
                    "true_value": float(batch["target_error"]),
                    "estimated_value": float(estimate),
                    "absolute_error": abs(signed),
                    "signed_error": signed,
                    "risk_failure": abs(signed) > RISK_TOLERANCE,
                    "alarm": False,
                    "alarm_target_event": False,
                    "shift_fraction": float(batch["shift_fraction"]),
                }
            )
        confidence_estimate = confidence.estimate_excess_log_loss(subset)
        signed = float(confidence_estimate - batch["true_excess_risk"])
        records.append(
            {
                **common,
                BATCH_INDEX: batch_index,
                "method": CONFIDENCE_METHOD,
                "endpoint": "excess_log_loss",
                "true_value": float(batch["true_excess_risk"]),
                "estimated_value": confidence_estimate,
                "absolute_error": abs(signed),
                "signed_error": signed,
                "risk_failure": abs(signed) > RISK_TOLERANCE,
                "alarm": False,
                "alarm_target_event": False,
                "shift_fraction": float(batch["shift_fraction"]),
            }
        )
        shd_batch = shd_batches[
            shd_batches[BATCH_INDEX] == batch_index
        ].iloc[0]
        records.append(
            {
                **common,
                BATCH_INDEX: batch_index,
                "method": SHD_METHOD,
                "endpoint": "sequential_selected_high_error_prevalence",
                "true_value": float(shd_batch.oracle_selected_high_error),
                "estimated_value": float(shd_batch.estimated_high_error_lower),
                "absolute_error": np.nan,
                "signed_error": np.nan,
                "risk_failure": np.nan,
                "alarm": bool(shd_batch.alarm),
                "alarm_target_event": bool(shd_batch.oracle_phi_q2_event),
                "alarm_phi_q": bool(shd_batch.alarm_phi_q),
                "selector_feasible": bool(shd.calibration.selector_feasible),
                "oracle_assumption_4_1_gap": float(
                    shd_batch.oracle_assumption_4_1_gap
                ),
                "shift_fraction": float(batch["shift_fraction"]),
            }
        )
    return pd.DataFrame(records)


def _scenario_summaries(batches: pd.DataFrame) -> pd.DataFrame:
    keys = [
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
    for key, group in batches.groupby(keys, sort=True, observed=True):
        group = group.sort_values(BATCH_INDEX)
        eligible = group[
            (group["shift"] == "no_shift") | (group["shift_fraction"] > 0)
        ]
        record = {
            **dict(zip(keys, key)),
            "evaluated_batches": len(eligible),
            "mean_true_value": eligible["true_value"].mean(),
            "mean_estimated_value": eligible["estimated_value"].mean(),
            "mean_absolute_error": eligible["absolute_error"].mean(),
            "mean_signed_error": eligible["signed_error"].mean(),
            "risk_failure_rate": eligible["risk_failure"].mean(),
        }
        if key[-2] == SHD_METHOD:
            record.update(
                compute_alarm_event_metrics(
                    group["alarm"].astype(bool).to_numpy(),
                    group["alarm_target_event"].astype(bool).to_numpy(),
                )
            )
        else:
            record.update(
                {
                    "event_batch": np.nan,
                    "first_alarm_batch": np.nan,
                    "false_alarm_rate": np.nan,
                    "power": np.nan,
                    "detection_delay": np.nan,
                    "missed_alarm": np.nan,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def _is_xpe_scenario(row: Any) -> bool:
    return (
        row.model in XPE_MODELS
        and row.shift in XPE_SHIFTS
        and (row.shift == "no_shift" or row.mode == "abrupt")
    )


def _evaluate_xpe(
    row: Any,
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
    base_models_dir: Path,
    sample_size: int,
    permutations: int,
    backend: str,
    kernel_nsamples: int,
) -> dict[str, Any]:
    reference_X = observable.load_reference_features(row.dataset)
    reference_y = observable.load_source_calibration_labels(row.dataset)
    explainer = XPEExplainer(reference_X, reference_y)
    model = joblib.load(
        base_models_dir / "models" / row.dataset / f"{row.model}_calibrated.pkl"
    )
    all_target = observable.load_target_features(row.stream_id)
    batch_size = int(row.batch_size)
    final_target = all_target.iloc[-batch_size:].reset_index(drop=True)
    result = explainer.explain(
        model,
        final_target,
        sample_size=sample_size,
        attribution_backend=backend,
        permutations=permutations,
        kernel_nsamples=kernel_nsamples,
        random_seed=_stable_seed("xpe", row.predictor_stream_id),
    )
    intervention = oracle.load_intervention_targets(row.stream_id).iloc[-1]
    true_attr = json.loads(intervention["ground_truth_attribution_json"])
    metrics = compute_attribution_metrics(true_attr, result.attribution, k=3)
    oracle_risk = oracle.load_batch_risk(row.predictor_stream_id).iloc[-1]
    final_labels = oracle.load_target_labels(row.stream_id).iloc[-batch_size:]
    final_labels = final_labels.sort_values(
        ["_tabmon_batch_index", "_tabmon_row_index"]
    ).reset_index(drop=True)
    sampled_labels = final_labels.iloc[result.target_sample_positions][
        TARGET_LABEL
    ].to_numpy()
    label_transfer_accuracy = float(
        np.mean(result.anticipated_labels == sampled_labels)
    )
    return {
        "predictor_stream_id": row.predictor_stream_id,
        "stream_id": row.stream_id,
        "dataset": row.dataset,
        "model": row.model,
        "shift": row.shift,
        "severity": row.severity,
        "mode": row.mode,
        "seed": row.seed,
        "method": result.method,
        "endpoint": "transport_anticipated_log_loss_change",
        "estimated_loss_change": result.estimated_loss_change,
        "oracle_excess_log_loss": float(oracle_risk.true_excess_risk),
        "absolute_risk_error": abs(
            result.estimated_loss_change - float(oracle_risk.true_excess_risk)
        ),
        "attribution_json": json.dumps(result.attribution, sort_keys=True),
        **metrics,
        "attribution_failure": bool(metrics["top3_recall"] < 0.5)
        if np.isfinite(metrics["top3_recall"])
        else False,
        "explained_targets": result.explained_targets,
        "source_transport_rows": result.source_transport_rows,
        "target_transport_rows": result.target_transport_rows,
        "coupling_retained_mass_fraction": result.coupling_retained_mass_fraction,
        "coupling_one_to_one_fraction": result.coupling_one_to_one_fraction,
        "shapley_efficiency_max_abs_error": result.shapley_efficiency_max_abs_error,
        "attribution_backend": result.attribution_backend,
        "attribution_budget": result.attribution_budget,
        "oracle_label_transfer_accuracy": label_transfer_accuracy,
        "permutations": permutations,
    }


def _write_status(
    output: Path,
    requested: int,
    completed: int,
    xpe_requested: int,
    xpe_completed: int,
    smoke: bool,
    stopped: bool,
) -> None:
    atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "complete": completed == requested
            and xpe_completed == xpe_requested
            and not stopped,
            "smoke_test": smoke,
            "requested_predictor_streams": requested,
            "completed_predictor_streams": completed,
            "completed_method_streams": completed * len(GRID_METHODS),
            "requested_xpe_streams": xpe_requested,
            "completed_xpe_streams": xpe_completed,
            "stopped_for_time": stopped,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output / "step12_status.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = args.cache_dir.resolve()
    models = args.base_models_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_environment = _validate_model_environment(
        models, args.allow_model_version_mismatch
    )
    index = _load_index(cache, args.expected_seeds)
    if args.smoke_test:
        index = _smoke_index(index)
    observable = ObservableCacheReader(cache)
    oracle = OracleCacheReader(cache)
    performance, confidence, shd, reference_error = _reference_estimators(
        observable, index, args.shd_alpha, args.shd_epsilon
    )
    calibration_records = []
    for (dataset, model), monitor in shd.items():
        calibration_records.append(
            {"dataset": dataset, "model": model, **monitor.calibration_record()}
        )
    atomic_parquet(
        pd.DataFrame(calibration_records), output / "shd_calibration.parquet"
    )

    checkpoints = output / "grid_checkpoints"
    xpe_checkpoints = output / "xpe_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    xpe_checkpoints.mkdir(parents=True, exist_ok=True)
    xpe_index = index[
        [_is_xpe_scenario(row) for row in index.itertuples(index=False)]
    ].copy()
    if args.skip_xpe:
        xpe_index = xpe_index.iloc[0:0]
    started = time.monotonic()
    stopped = False
    for position, row in enumerate(index.itertuples(index=False), start=1):
        path = checkpoints / f"{row.predictor_stream_id}.parquet"
        pair = (row.dataset, row.model)
        if not path.is_file():
            frame = _evaluate_grid_stream(
                row,
                observable,
                oracle,
                performance[pair],
                confidence[pair],
                shd[pair],
                reference_error[pair],
            )
            atomic_parquet(frame, path)
        if _is_xpe_scenario(row) and not args.skip_xpe:
            xpe_path = xpe_checkpoints / f"{row.predictor_stream_id}.json"
            if not xpe_path.is_file():
                atomic_json(
                    _evaluate_xpe(
                        row,
                        observable,
                        oracle,
                        models,
                        args.xpe_sample_size,
                        args.xpe_permutations,
                        args.xpe_backend,
                        args.xpe_kernel_nsamples,
                    ),
                    xpe_path,
                )
        if position == 1 or position % 25 == 0 or position == len(index):
            completed = sum(path.is_file() for path in checkpoints.glob("*.parquet"))
            xpe_completed = sum(path.is_file() for path in xpe_checkpoints.glob("*.json"))
            _write_status(
                output,
                len(index),
                completed,
                len(xpe_index),
                xpe_completed,
                args.smoke_test,
                False,
            )
            print(
                f"[{position}/{len(index)}] grid={completed}, "
                f"xpe={xpe_completed}/{len(xpe_index)}",
                flush=True,
            )
        if args.max_hours is not None and (time.monotonic() - started) / 3600 >= args.max_hours:
            stopped = True
            break

    grid_paths = sorted(checkpoints.glob("*.parquet"))
    xpe_paths = sorted(xpe_checkpoints.glob("*.json"))
    batches = pd.concat([pd.read_parquet(path) for path in grid_paths], ignore_index=True)
    scenarios = _scenario_summaries(batches)
    xpe = pd.DataFrame(
        [json.loads(path.read_text("utf-8")) for path in xpe_paths]
    )
    atomic_parquet(batches, output / "batch_metrics.parquet")
    atomic_parquet(scenarios, output / "scenario_metrics.parquet")
    atomic_parquet(xpe, output / "xpe_metrics.parquet")
    completed = len(grid_paths)
    xpe_completed = len(xpe_paths)
    _write_status(
        output,
        len(index),
        completed,
        len(xpe_index),
        xpe_completed,
        args.smoke_test,
        stopped,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Q1 extension: SHD, XPE, multiclass compatibility, ten seeds",
        "grid_methods": list(GRID_METHODS),
        "xpe_method": (
            "xpe_reference_kernel_shap"
            if args.xpe_backend == "kernel_shap"
            else "xpe_grouped_permutation"
        ),
        "endpoint_separation": {
            "ac_doc_atc_cot_cott": "classification_error",
            "confidence": "excess_log_loss",
            "shd": "sequential harmful classification-error detection",
            "xpe": "transport-anticipated log-loss change and intervention attribution",
        },
        "multiclass_cott": {
            "class_mass_source": "full source calibration split",
            "threshold_score_calibration_limit": MULTICLASS_COTT_CALIBRATION_LIMIT,
            "subsample_rule": "deterministic evenly spaced row positions",
            "target_batch_transport": "full batch exact EMD",
        },
        "xpe_subset": {
            "models": sorted(XPE_MODELS),
            "shifts": sorted(XPE_SHIFTS),
            "modes": "null static or abrupt only",
            "source_and_target_sample_size": args.xpe_sample_size,
            "attribution_backend": args.xpe_backend,
            "permutations": args.xpe_permutations,
            "kernel_nsamples": args.xpe_kernel_nsamples,
        },
        "method_fidelity": {
            "shd": "published quantile selector, Phi_q^2, PM-EB CS, Hoeffding source CI",
            "xpe": "equal-mass EMD best match with selectable KernelSHAP or grouped-permutation Shapley",
            "official_code_vendored": False,
        },
        "forbidden_monitor_inputs": [
            "target labels",
            "oracle risk",
            "failure labels",
            "intervention ground truth",
        ],
        "environment": {
            **model_environment,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest, output / "step12_manifest.json")
    print("=== TABMON STEP 12 Q1 EXTENSION ===")
    print(f"Predictor streams: {completed}/{len(index)}")
    print(f"Method streams: {len(scenarios)}")
    print(f"XPE streams: {xpe_completed}/{len(xpe_index)}")
    print(f"Output: {output}")
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
