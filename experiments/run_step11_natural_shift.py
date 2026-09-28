"""Step 11: natural temporal/geographic external validation for TABMON-Bench."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from folktables import ACSIncome
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_schema_v7 import benjamini_hochberg
from src.baselines.cached_confidence import CachedConfidenceEstimator
from src.baselines.performance_estimators import METHODS, PerformanceEstimatorSuite
from src.baselines.schema_guard import SchemaGuardProfile
from src.cache.stream_cache import atomic_json, atomic_parquet
from src.evaluation.alarm_calibration import split_conformal_threshold
from src.evaluation.natural_shift_protocol import (
    MONITOR_OUTPUT_COLUMNS,
    assert_disjoint_complete,
    select_largest_groups,
    split_source_indices,
    temporal_source_and_windows,
    validate_monitor_output_columns,
)
from src.evaluation.observable_diagnostics import observable_domain_diagnostics
from src.evaluation.risk import binary_log_losses


MODEL_NAMES = ("lr", "rf", "xgb", "mlp")
ALL_METHODS = (*METHODS, "confidence_log_loss")
TARGET_DOMAIN_TYPES = ("external_geographic", "external_temporal")


@dataclass
class Domain:
    dataset: str
    domain_id: str
    domain_type: str
    X: pd.DataFrame
    y: pd.Series
    row_ids: np.ndarray


@dataclass
class DatasetProtocol:
    dataset: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    domains: list[Domain]
    assignments: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-file", type=Path, required=True)
    parser.add_argument("--bank-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--guard-null-samples", type=int, default=200)
    parser.add_argument("--guard-alpha", type=float, default=0.01)
    parser.add_argument("--diagnostic-max-rows", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1000:
        parser.error("Use at least 1,000 bootstrap replicates")
    if args.guard_null_samples < 100:
        parser.error("Use at least 100 guard null samples")
    if not 0.0 < args.guard_alpha < 1.0:
        parser.error("--guard-alpha must lie between zero and one")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _assignment_rows(
    dataset: str,
    row_ids: np.ndarray,
    partition: str,
    domain_id: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "row_id": np.asarray(row_ids, dtype=np.int64),
            "partition": partition,
            "domain_id": domain_id,
        }
    )


def _build_acs_protocol(path: Path, seed: int) -> DatasetProtocol:
    columns = list(
        dict.fromkeys([*ACSIncome.features, ACSIncome.target, "PWGTP", "PUMA"])
    )
    raw = pd.read_csv(path, usecols=columns, low_memory=False)
    filtered = ACSIncome._preprocess(raw).copy()
    raw_ids = filtered.index.to_numpy(dtype=np.int64)
    X, y_frame, _ = ACSIncome.df_to_pandas(filtered)
    X = X.reset_index(drop=True)
    y = y_frame.iloc[:, 0].astype(np.int8).reset_index(drop=True)
    puma = filtered["PUMA"].reset_index(drop=True)

    target_pumas = select_largest_groups(puma, 5)
    target_mask = puma.isin(target_pumas).to_numpy()
    source_positions = np.flatnonzero(~target_mask)
    train, calibration, holdout = split_source_indices(
        source_positions, y.iloc[source_positions].to_numpy(), seed
    )
    target_parts = [np.flatnonzero(puma.to_numpy() == value) for value in target_pumas]
    assert_disjoint_complete(
        np.arange(len(X)), [train, calibration, holdout, *target_parts]
    )

    domains = [
        Domain(
            "acs_income",
            "source_holdout",
            "source_holdout",
            X.iloc[holdout].reset_index(drop=True),
            y.iloc[holdout].reset_index(drop=True),
            raw_ids[holdout],
        )
    ]
    for puma_value, positions in zip(target_pumas, target_parts):
        domains.append(
            Domain(
                "acs_income",
                f"puma_{puma_value}",
                "external_geographic",
                X.iloc[positions].reset_index(drop=True),
                y.iloc[positions].reset_index(drop=True),
                raw_ids[positions],
            )
        )
    assignments = pd.concat(
        [
            _assignment_rows("acs_income", raw_ids[train], "train", "source"),
            _assignment_rows(
                "acs_income", raw_ids[calibration], "calibration", "source"
            ),
            _assignment_rows(
                "acs_income", raw_ids[holdout], "source_holdout", "source_holdout"
            ),
            *[
                _assignment_rows(
                    "acs_income", raw_ids[positions], "external_target", f"puma_{value}"
                )
                for value, positions in zip(target_pumas, target_parts)
            ],
        ],
        ignore_index=True,
    )
    return DatasetProtocol(
        "acs_income",
        X.iloc[train].reset_index(drop=True),
        y.iloc[train].reset_index(drop=True),
        X.iloc[calibration].reset_index(drop=True),
        y.iloc[calibration].reset_index(drop=True),
        domains,
        assignments,
    )


def _build_bank_protocol(path: Path, seed: int) -> DatasetProtocol:
    frame = pd.read_csv(path, sep=";")
    y = frame["y"].eq("yes").astype(np.int8).reset_index(drop=True)
    X = frame.drop(columns="y").reset_index(drop=True)
    row_ids = np.arange(len(frame), dtype=np.int64)
    source, windows = temporal_source_and_windows(len(frame), 0.30, 5)
    train, calibration, holdout = split_source_indices(
        source, y.iloc[source].to_numpy(), seed
    )
    assert_disjoint_complete(
        np.arange(len(frame)), [train, calibration, holdout, *windows]
    )
    domains = [
        Domain(
            "bank_marketing",
            "source_holdout",
            "source_holdout",
            X.iloc[holdout].reset_index(drop=True),
            y.iloc[holdout].reset_index(drop=True),
            row_ids[holdout],
        )
    ]
    for index, positions in enumerate(windows, start=1):
        domains.append(
            Domain(
                "bank_marketing",
                f"late_window_{index}",
                "external_temporal",
                X.iloc[positions].reset_index(drop=True),
                y.iloc[positions].reset_index(drop=True),
                row_ids[positions],
            )
        )
    assignments = pd.concat(
        [
            _assignment_rows("bank_marketing", row_ids[train], "train", "source"),
            _assignment_rows(
                "bank_marketing", row_ids[calibration], "calibration", "source"
            ),
            _assignment_rows(
                "bank_marketing", row_ids[holdout], "source_holdout", "source_holdout"
            ),
            *[
                _assignment_rows(
                    "bank_marketing",
                    row_ids[positions],
                    "external_target",
                    f"late_window_{index}",
                )
                for index, positions in enumerate(windows, start=1)
            ],
        ],
        ignore_index=True,
    )
    return DatasetProtocol(
        "bank_marketing",
        X.iloc[train].reset_index(drop=True),
        y.iloc[train].reset_index(drop=True),
        X.iloc[calibration].reset_index(drop=True),
        y.iloc[calibration].reset_index(drop=True),
        domains,
        assignments,
    )


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X if column not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )


def _models(seed: int) -> dict[str, Any]:
    return {
        "lr": LogisticRegression(max_iter=1000, random_state=seed),
        "rf": RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed
        ),
        "xgb": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    }


def _calibrate_prefit(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    return calibrated.fit(X, y)


def _prediction_frame(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    result = pd.DataFrame(
        {
            f"probability_class_{index}": probabilities[:, index]
            for index in range(probabilities.shape[1])
        }
    )
    result["predicted_class"] = np.asarray(model.classes_)[
        np.argmax(probabilities, axis=1)
    ]
    return result


def _guard_domain_record(
    protocol: DatasetProtocol,
    domain: Domain,
    profile: SchemaGuardProfile,
    null_samples: int,
    alpha: float,
    max_rows: int,
) -> dict[str, Any]:
    sample_size = min(max_rows, len(domain.X))
    rng = np.random.default_rng(_stable_seed("natural-guard", protocol.dataset, domain.domain_id))
    positions = rng.integers(
        0,
        len(protocol.X_calibration),
        size=(null_samples, sample_size),
        dtype=np.int32,
    )
    null_scores = profile.score_position_batches(protocol.X_calibration, positions)
    threshold, _, _ = split_conformal_threshold(null_scores, alpha)
    if len(domain.X) > sample_size:
        target_positions = rng.choice(len(domain.X), sample_size, replace=False)
        target_sample = domain.X.iloc[target_positions]
    else:
        target_sample = domain.X
    report = profile.score_batch(target_sample.reset_index(drop=True))
    p_value = float(
        (1 + np.count_nonzero(null_scores >= report.score)) / (len(null_scores) + 1)
    )
    return {
        "dataset": protocol.dataset,
        "domain_id": domain.domain_id,
        "domain_type": domain.domain_type,
        "target_rows": len(domain.X),
        "guard_sample_rows": sample_size,
        "schema_guard_score": report.score,
        "schema_guard_threshold": threshold,
        "schema_guard_alarm": bool(report.score > threshold),
        "schema_guard_p_value": p_value,
        "schema_guard_top_feature": report.top_feature,
        "schema_guard_top_component": report.top_component,
    }


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    cluster_means = frame.groupby(cluster)[value].mean().to_numpy(dtype=float)
    estimate = float(cluster_means.mean())
    rng = np.random.default_rng(seed)
    draws = cluster_means[
        rng.integers(0, len(cluster_means), size=(replicates, len(cluster_means)))
    ].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def _summary(offline: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows = []
    scopes = [
        ("source_holdout", "all", offline[offline["domain_type"].eq("source_holdout")]),
        ("external_overall", "all", offline[offline["domain_type"].isin(TARGET_DOMAIN_TYPES)]),
    ]
    for dataset, group in offline[offline["domain_type"].isin(TARGET_DOMAIN_TYPES)].groupby("dataset"):
        scopes.append(("external_dataset", dataset, group))
    for scope, level, frame in scopes:
        for (method, endpoint), group in frame.groupby(["method", "endpoint"]):
            cluster = "natural_domain_key"
            for metric in ("absolute_error", "failure"):
                estimate, lower, upper = _cluster_bootstrap_mean(
                    group,
                    metric,
                    cluster,
                    replicates,
                    _stable_seed("step11-summary", scope, level, method, metric),
                )
                rows.append(
                    {
                        "scope": scope,
                        "level": level,
                        "method": method,
                        "endpoint": endpoint,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "domain_count": group[cluster].nunique(),
                        "predictor_domain_count": len(group),
                    }
                )
            harmful = group[group["harmful"]]
            rows.append(
                {
                    "scope": scope,
                    "level": level,
                    "method": method,
                    "endpoint": endpoint,
                    "metric": "harmful_sign_reversal_rate",
                    "estimate": float(harmful["sign_reversal"].mean()) if len(harmful) else np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "domain_count": group[cluster].nunique(),
                    "predictor_domain_count": len(group),
                }
            )
    return pd.DataFrame(rows)


def _exact_sign_flip_p_value(cluster_differences: np.ndarray) -> float:
    values = np.asarray(cluster_differences, dtype=float)
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(values))))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def _paired_comparisons(offline: pd.DataFrame, replicates: int) -> pd.DataFrame:
    frame = offline[
        offline["domain_type"].isin(TARGET_DOMAIN_TYPES)
        & offline["endpoint"].eq("classification_error")
    ]
    index = ["natural_domain_key", "dataset", "model", "domain_id"]
    rows = []
    for outcome in ("absolute_error", "failure"):
        wide = frame.pivot(index=index, columns="method", values=outcome)
        if set(wide.columns) != set(METHODS) or wide.isna().any().any():
            raise ValueError("Natural-shift paired method grid is incomplete")
        for method_a, method_b in combinations(METHODS, 2):
            differences = (wide[method_a] - wide[method_b]).rename("difference").reset_index()
            cluster_values = differences.groupby("natural_domain_key")["difference"].mean()
            rng = np.random.default_rng(_stable_seed("step11-pair", outcome, method_a, method_b))
            draws = cluster_values.to_numpy()[
                rng.integers(
                    0,
                    len(cluster_values),
                    size=(replicates, len(cluster_values)),
                )
            ].mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
            estimate = float(cluster_values.mean())
            rows.append(
                {
                    "outcome": outcome,
                    "method_a": method_a,
                    "method_b": method_b,
                    "difference_a_minus_b": estimate,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "exact_cluster_sign_flip_p": _exact_sign_flip_p_value(
                        cluster_values.to_numpy()
                    ),
                    "natural_domains": len(cluster_values),
                    "predictor_domains": len(differences),
                    "ci_winner": method_a
                    if upper < 0
                    else method_b
                    if lower > 0
                    else "no_clear_winner",
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh_global"] = benjamini_hochberg(
        result["exact_cluster_sign_flip_p"]
    )
    result["globally_significant"] = result["q_value_bh_global"].lt(0.05)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocols = [
        _build_acs_protocol(args.acs_file.resolve(), args.random_seed),
        _build_bank_protocol(args.bank_file.resolve(), args.random_seed),
    ]
    assignments = pd.concat([item.assignments for item in protocols], ignore_index=True)
    atomic_parquet(assignments, output / "natural_domain_assignments.parquet")

    guard_rows = []
    for protocol in protocols:
        profile = SchemaGuardProfile.fit(protocol.X_calibration)
        for domain in protocol.domains:
            guard_rows.append(
                _guard_domain_record(
                    protocol,
                    domain,
                    profile,
                    args.guard_null_samples,
                    args.guard_alpha,
                    args.diagnostic_max_rows,
                )
            )
    guard_frame = pd.DataFrame(guard_rows)
    atomic_parquet(guard_frame, output / "natural_schema_guard.parquet")
    guard_lookup = guard_frame.set_index(["dataset", "domain_id"])

    monitor_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, np.ndarray]] = {}
    source_reference: dict[tuple[str, str], dict[str, float]] = {}
    for protocol in protocols:
        preprocessor = _preprocessor(protocol.X_train)
        for model_name, estimator in _models(args.random_seed).items():
            print(f"[train] {protocol.dataset}/{model_name}", flush=True)
            pipeline = Pipeline(
                [("preprocessor", clone(preprocessor)), ("estimator", estimator)]
            ).fit(protocol.X_train, protocol.y_train)
            model = _calibrate_prefit(
                pipeline, protocol.X_calibration, protocol.y_calibration
            )
            reference_predictions = _prediction_frame(model, protocol.X_calibration)
            classes = np.asarray(model.classes_)
            performance = PerformanceEstimatorSuite.fit(
                reference_predictions, protocol.y_calibration, classes
            )
            confidence = CachedConfidenceEstimator.fit(
                reference_predictions, protocol.y_calibration, classes
            )
            reference_probabilities = reference_predictions[
                ["probability_class_0", "probability_class_1"]
            ].to_numpy(float)
            source_reference[(protocol.dataset, model_name)] = {
                "source_error": performance.source_error,
                "source_log_loss": float(
                    binary_log_losses(
                        reference_probabilities, classes, protocol.y_calibration.to_numpy()
                    ).mean()
                ),
            }
            for domain in protocol.domains:
                target_predictions = _prediction_frame(model, domain.X)
                prediction_cache[(protocol.dataset, model_name, domain.domain_id)] = (
                    target_predictions,
                    classes,
                )
                estimates = performance.estimate(target_predictions)
                confidence_estimate = confidence.estimate_excess_log_loss(
                    target_predictions
                )
                diagnostics = observable_domain_diagnostics(
                    protocol.X_calibration,
                    domain.X,
                    model,
                    random_seed=_stable_seed(
                        "natural-diagnostic", protocol.dataset, model_name, domain.domain_id
                    ),
                    max_rows=args.diagnostic_max_rows,
                )
                guard = guard_lookup.loc[(protocol.dataset, domain.domain_id)]
                shared = {
                    "dataset": protocol.dataset,
                    "model": model_name,
                    "domain_id": domain.domain_id,
                    "domain_type": domain.domain_type,
                    "target_rows": len(domain.X),
                    **diagnostics,
                    "schema_guard_score": float(guard["schema_guard_score"]),
                    "schema_guard_alarm": bool(guard["schema_guard_alarm"]),
                    "schema_guard_p_value": float(guard["schema_guard_p_value"]),
                }
                for method, estimate in estimates.items():
                    monitor_rows.append(
                        {
                            **shared,
                            "method": method,
                            "endpoint": "classification_error",
                            "source_value": performance.source_error,
                            "estimated_value": estimate,
                            "estimated_excess_value": estimate - performance.source_error,
                        }
                    )
                monitor_rows.append(
                    {
                        **shared,
                        "method": "confidence_log_loss",
                        "endpoint": "excess_log_loss",
                        "source_value": 0.0,
                        "estimated_value": confidence_estimate,
                        "estimated_excess_value": confidence_estimate,
                    }
                )

    monitor = pd.DataFrame(monitor_rows)[list(MONITOR_OUTPUT_COLUMNS)]
    validate_monitor_output_columns(monitor.columns)
    atomic_parquet(monitor, output / "natural_monitor_outputs.parquet")
    monitor_hash = _sha256(output / "natural_monitor_outputs.parquet")

    # Offline evaluation begins only after observable outputs are frozen.
    domain_lookup = {
        (protocol.dataset, domain.domain_id): domain
        for protocol in protocols
        for domain in protocol.domains
    }
    oracle_rows = []
    for row in monitor.itertuples(index=False):
        domain = domain_lookup[(row.dataset, row.domain_id)]
        predictions, classes = prediction_cache[(row.dataset, row.model, row.domain_id)]
        if row.endpoint == "classification_error":
            true_value = float(
                np.mean(predictions["predicted_class"].to_numpy() != domain.y.to_numpy())
            )
            true_excess = true_value - float(row.source_value)
        else:
            probabilities = predictions[
                ["probability_class_0", "probability_class_1"]
            ].to_numpy(float)
            target_log_loss = float(
                binary_log_losses(probabilities, classes, domain.y.to_numpy()).mean()
            )
            true_excess = target_log_loss - source_reference[
                (row.dataset, row.model)
            ]["source_log_loss"]
            true_value = true_excess
        absolute_error = abs(float(row.estimated_value) - true_value)
        harmful = true_excess > 0.05
        oracle_rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "domain_id": row.domain_id,
                "domain_type": row.domain_type,
                "method": row.method,
                "endpoint": row.endpoint,
                "true_value": true_value,
                "true_excess_value": true_excess,
                "signed_error": float(row.estimated_value) - true_value,
                "absolute_error": absolute_error,
                "failure": float(absolute_error > 0.05),
                "harmful": bool(harmful),
                "sign_reversal": bool(harmful and row.estimated_excess_value < 0.0),
                "target_positive_rate": float(domain.y.mean()),
            }
        )
    oracle = pd.DataFrame(oracle_rows)
    offline = monitor.merge(
        oracle,
        on=["dataset", "model", "domain_id", "domain_type", "method", "endpoint"],
        validate="one_to_one",
    )
    offline["natural_domain_key"] = offline["dataset"] + "/" + offline["domain_id"]
    atomic_parquet(oracle, output / "natural_oracle_metrics.parquet")
    atomic_parquet(offline, output / "natural_offline_evaluation.parquet")

    summary = _summary(offline, args.bootstrap_replicates)
    paired = _paired_comparisons(offline, args.bootstrap_replicates)
    summary.to_csv(output / "natural_method_summary.csv", index=False)
    paired.to_csv(output / "natural_paired_comparisons.csv", index=False)

    input_paths = {
        "acs_income_raw": args.acs_file.resolve(),
        "bank_marketing_raw": args.bank_file.resolve(),
    }
    manifest = {
        "step": 11,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "natural geographic and temporal external validation",
        "uses_synthetic_shift_generator": False,
        "target_labels_available_to_monitors": False,
        "source_protocol": "60% train, 20% calibration, 20% internal holdout",
        "external_domains": {
            "acs_income": "five largest California PUMAs by row count; PUMA excluded from features",
            "bank_marketing": "last 30% of date-ordered bank-full.csv in five contiguous windows",
        },
        "domain_selection_uses_outcomes": False,
        "methods": list(ALL_METHODS),
        "endpoints_kept_separate": True,
        "classification_endpoint": "absolute 0-1 error",
        "confidence_endpoint": "excess binary log loss",
        "failure_tolerance": 0.05,
        "guard_alpha": args.guard_alpha,
        "guard_null_samples": args.guard_null_samples,
        "monitor_output_columns": list(MONITOR_OUTPUT_COLUMNS),
        "monitor_output_sha256_before_oracle_access": monitor_hash,
        "input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
        "source_urls": {
            "acs_puma_definition": "https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html",
            "bank_marketing": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        },
    }
    status = {
        "step": 11,
        "complete": True,
        "datasets": len(protocols),
        "source_holdout_domains": 2,
        "external_natural_domains": 10,
        "trained_models": len(protocols) * len(MODEL_NAMES),
        "predictor_domain_evaluations": len(protocols) * 6 * len(MODEL_NAMES),
        "monitor_domain_evaluations": len(monitor),
        "globally_significant_paired_tests": int(paired["globally_significant"].sum()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest, output / "step11_manifest.json")
    atomic_json(status, output / "step11_status.json")

    external = summary[summary["scope"].eq("external_overall")]
    print("\n=== STEP 11 COMPLETE ===")
    print(json.dumps(status, indent=2))
    print("\nExternal natural-domain summary:")
    print(external.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

