"""Resumable experiment runner for the TABMON-Bench core benchmark.

The SQLite database is the live checkpoint. Human-readable CSV and Parquet
tables are exported whenever the run stops, completes, or encounters errors.
Target labels are sent only to the offline oracle, never to a monitor call.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sqlite3
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from itertools import product
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.b0_confidence import ConfidenceShiftMonitor
from src.baselines.b1_drift_shap import DriftSHAPMonitor
from src.benchmark_config import (
    DATASETS,
    DATASET_CONFIG,
    MODELS,
    MODES,
    MONITORS,
    SEVERITIES,
    SHIFT_FAMILIES,
)
from src.evaluation.metrics import (
    compute_alarm_event_metrics,
    compute_event_based_sequential_metrics,
    compute_risk_metrics,
)
from src.evaluation.oracle_evaluator import OracleEvaluator
from src.evaluation.protocol import TABMONProtocol
from src.shift_generator import TabularShiftGenerator
from src.stream_protocol import (
    generate_stream as _generate_stream_shared,
    shift_fraction as _shift_fraction_shared,
)


SCHEMA_VERSION = 7
ALARM_TARGETS = {"risk_event", "distribution_shift"}


@dataclass(frozen=True)
class Scenario:
    dataset: str
    model: str
    monitor: str
    shift: str
    severity: str
    mode: str
    seed: int
    num_batches: int
    batch_size: int
    reliability_signature: str

    @property
    def scenario_id(self) -> str:
        payload = {"schema_version": SCHEMA_VERSION, **asdict(self)}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TABMON-Bench core sweep.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--monitors", nargs="+", choices=MONITORS, default=MONITORS)
    parser.add_argument(
        "--shifts", nargs="+", choices=SHIFT_FAMILIES, default=SHIFT_FAMILIES
    )
    parser.add_argument(
        "--severities", nargs="+", choices=SEVERITIES, default=SEVERITIES
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=["abrupt"],
        help="Stream modes to evaluate (default: abrupt).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--k-features", type=int, default=3)
    parser.add_argument("--risk-failure-tolerance", type=float, default=0.05)
    parser.add_argument("--attribution-failure-threshold", type=float, default=0.5)
    parser.add_argument("--risk-event-threshold", type=float, default=0.05)
    parser.add_argument("--alarm-quantile", type=float, default=0.99)
    parser.add_argument("--null-calibration-batches", type=int, default=100)
    parser.add_argument(
        "--observable-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute label-free domain, confidence, ESS, and novelty diagnostics.",
    )
    parser.add_argument("--diagnostic-max-rows", type=int, default=1000)
    parser.add_argument("--density-ratio-clip", type=float, default=10.0)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--base-models-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "base_models",
        help="Directory containing models/<dataset>/<model>_calibrated.pkl.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "core_benchmark",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help=(
            "Soft time limit checked between scenarios. Use 10 on a 12-hour "
            "Kaggle session to leave a safety buffer."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-model-version-mismatch",
        action="store_true",
        help="Allow loading pickled models with a different sklearn/xgboost version.",
    )
    parser.add_argument(
        "--save-batch-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-batch trajectories in the checkpoint and Parquet export.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one small Adult/LR/DriftSHAP scenario.",
    )
    parser.add_argument(
        "--reliability-smoke",
        action="store_true",
        help="Run a compact Adult reliability pilot before the stability sweep.",
    )
    parser.add_argument(
        "--stability-pilot",
        action="store_true",
        help="Run Adult, four models, five seeds, null + abrupt + gradual shifts.",
    )
    args = parser.parse_args(argv)

    if sum((args.smoke_test, args.reliability_smoke, args.stability_pilot)) > 1:
        parser.error("Choose only one pilot/smoke preset")

    if args.smoke_test:
        args.datasets = ["adult"]
        args.models = ["lr"]
        args.monitors = ["drift_shap"]
        args.shifts = ["covariate"]
        args.severities = ["high"]
        args.modes = ["abrupt"]
        args.seeds = [42]
        args.num_batches = 4
        args.batch_size = min(args.batch_size, 128)
        args.null_calibration_batches = min(args.null_calibration_batches, 10)

    if args.reliability_smoke:
        args.datasets = ["adult"]
        args.models = ["lr", "xgb"]
        args.monitors = ["drift_shap", "confidence"]
        args.shifts = ["no_shift", "covariate", "concept"]
        args.severities = ["high"]
        args.modes = ["abrupt", "gradual"]
        args.seeds = [42, 43]
        args.num_batches = 8
        args.batch_size = min(args.batch_size, 256)
        args.null_calibration_batches = min(args.null_calibration_batches, 100)

    if args.stability_pilot:
        args.datasets = ["adult"]
        args.models = list(MODELS)
        args.monitors = list(MONITORS)
        args.shifts = list(SHIFT_FAMILIES)
        args.severities = list(SEVERITIES)
        args.modes = ["abrupt", "gradual"]
        args.seeds = [42, 43, 44, 45, 46]

    for name in ("num_batches", "batch_size", "k_features"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours must be greater than zero")
    if args.risk_failure_tolerance <= 0 or args.risk_event_threshold <= 0:
        parser.error("Risk thresholds must be greater than zero")
    if not 0.0 <= args.attribution_failure_threshold <= 1.0:
        parser.error("--attribution-failure-threshold must be in [0, 1]")
    if not 0.5 < args.alarm_quantile < 1.0:
        parser.error("--alarm-quantile must be between 0.5 and 1.0")
    if args.null_calibration_batches < 5:
        parser.error("--null-calibration-batches must be at least 5")
    if args.diagnostic_max_rows < 20 or args.density_ratio_clip <= 1.0:
        parser.error("Invalid observable diagnostic configuration")
    return args


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    reliability_config = {
        "k_features": args.k_features,
        "risk_failure_tolerance": args.risk_failure_tolerance,
        "attribution_failure_threshold": args.attribution_failure_threshold,
        "risk_event_threshold": args.risk_event_threshold,
        "alarm_quantile": args.alarm_quantile,
        "null_calibration_batches": args.null_calibration_batches,
        "observable_diagnostics": args.observable_diagnostics,
        "diagnostic_max_rows": args.diagnostic_max_rows,
        "density_ratio_clip": args.density_ratio_clip,
    }
    reliability_signature = hashlib.sha256(
        json.dumps(reliability_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    scenarios: list[Scenario] = []
    common = product(args.datasets, args.models, args.monitors, args.seeds)
    for dataset, model, monitor, seed in common:
        if "no_shift" in args.shifts:
            scenarios.append(
                Scenario(
                    dataset, model, monitor, "no_shift", "none", "static", seed,
                    args.num_batches, args.batch_size, reliability_signature,
                )
            )
        for shift, severity, mode in product(
            [name for name in args.shifts if name != "no_shift"],
            args.severities,
            args.modes,
        ):
            scenarios.append(
                Scenario(
                    dataset, model, monitor, shift, severity, mode, seed,
                    args.num_batches, args.batch_size, reliability_signature,
                )
            )
    return scenarios


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def open_checkpoint(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scenarios (
            scenario_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            model TEXT NOT NULL,
            monitor TEXT NOT NULL,
            shift TEXT NOT NULL,
            severity TEXT NOT NULL,
            mode TEXT NOT NULL,
            seed INTEGER NOT NULL,
            num_batches INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            status TEXT NOT NULL,
            metrics_json TEXT,
            updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batches (
            scenario_id TEXT NOT NULL,
            batch_index INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            PRIMARY KEY (scenario_id, batch_index),
            FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
        );
        CREATE TABLE IF NOT EXISTS errors (
            scenario_id TEXT PRIMARY KEY,
            error TEXT NOT NULL,
            traceback TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def completed_scenario_ids(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT scenario_id FROM scenarios WHERE status = 'completed'"
    ).fetchall()
    return {row[0] for row in rows}


def reset_scenarios(connection: sqlite3.Connection, scenarios: list[Scenario]) -> None:
    ids = [(scenario.scenario_id,) for scenario in scenarios]
    with connection:
        connection.executemany("DELETE FROM batches WHERE scenario_id = ?", ids)
        connection.executemany("DELETE FROM errors WHERE scenario_id = ?", ids)
        connection.executemany("DELETE FROM scenarios WHERE scenario_id = ?", ids)


def checkpoint_success(
    connection: sqlite3.Connection,
    scenario: Scenario,
    aggregate: dict[str, Any],
    batches: list[dict[str, Any]],
    save_batches: bool,
) -> None:
    metadata_values = (
        scenario.scenario_id,
        scenario.dataset,
        scenario.model,
        scenario.monitor,
        scenario.shift,
        scenario.severity,
        scenario.mode,
        scenario.seed,
        scenario.num_batches,
        scenario.batch_size,
        "completed",
        json.dumps(_json_ready(aggregate), sort_keys=True),
        _utc_now(),
    )
    with connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            metadata_values,
        )
        connection.execute(
            "DELETE FROM batches WHERE scenario_id = ?", (scenario.scenario_id,)
        )
        if save_batches:
            connection.executemany(
                "INSERT INTO batches VALUES (?, ?, ?)",
                [
                    (
                        scenario.scenario_id,
                        int(batch["batch_index"]),
                        json.dumps(_json_ready(batch), sort_keys=True),
                    )
                    for batch in batches
                ],
            )
        connection.execute(
            "DELETE FROM errors WHERE scenario_id = ?", (scenario.scenario_id,)
        )


def checkpoint_error(
    connection: sqlite3.Connection, scenario: Scenario, exc: Exception
) -> None:
    values = (
        scenario.scenario_id,
        scenario.dataset,
        scenario.model,
        scenario.monitor,
        scenario.shift,
        scenario.severity,
        scenario.mode,
        scenario.seed,
        scenario.num_batches,
        scenario.batch_size,
        "error",
        None,
        _utc_now(),
    )
    with connection:
        connection.execute("INSERT OR REPLACE INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        connection.execute(
            "INSERT OR REPLACE INTO errors VALUES (?, ?, ?, ?)",
            (scenario.scenario_id, repr(exc), traceback.format_exc(), _utc_now()),
        )


def export_checkpoint(connection: sqlite3.Connection, results_dir: Path) -> None:
    scenario_rows = connection.execute(
        """
        SELECT scenario_id, dataset, model, monitor, shift, severity, mode, seed,
               num_batches, batch_size, metrics_json
        FROM scenarios WHERE status = 'completed' ORDER BY dataset, model, monitor,
               shift, severity, mode, seed
        """
    ).fetchall()
    scenario_columns = [
        "scenario_id", "dataset", "model", "monitor", "shift", "severity",
        "mode", "seed", "num_batches", "batch_size",
    ]
    scenario_records = []
    scenario_metadata: dict[str, dict[str, Any]] = {}
    for row in scenario_rows:
        record = dict(zip(scenario_columns, row[:10]))
        scenario_metadata[row[0]] = record
        record.update(json.loads(row[10]))
        scenario_records.append(record)

    if scenario_records:
        scenario_frame = pd.DataFrame(scenario_records)
        _write_table_atomic(scenario_frame, results_dir / "aggregate_metrics.csv")
        _write_table_atomic(scenario_frame, results_dir / "aggregate_metrics.parquet")

    batch_rows = connection.execute(
        "SELECT scenario_id, batch_index, metrics_json FROM batches ORDER BY scenario_id, batch_index"
    ).fetchall()
    batch_records = []
    for scenario_id, batch_index, metrics_json in batch_rows:
        if scenario_id not in scenario_metadata:
            continue
        record = dict(scenario_metadata[scenario_id])
        record.update(json.loads(metrics_json))
        record["batch_index"] = batch_index
        batch_records.append(record)
    if batch_records:
        batch_frame = pd.DataFrame(batch_records)
        _write_table_atomic(batch_frame, results_dir / "batch_metrics.parquet")

    error_rows = connection.execute(
        """
        SELECT s.scenario_id, s.dataset, s.model, s.monitor, s.shift, s.severity,
               s.mode, s.seed, e.error, e.traceback, e.updated_at_utc
        FROM errors e JOIN scenarios s USING (scenario_id)
        ORDER BY e.updated_at_utc
        """
    ).fetchall()
    if error_rows:
        error_columns = [
            "scenario_id", "dataset", "model", "monitor", "shift", "severity",
            "mode", "seed", "error", "traceback", "updated_at_utc",
        ]
        _write_table_atomic(
            pd.DataFrame(error_rows, columns=error_columns),
            results_dir / "scenario_errors.csv",
        )
    else:
        error_path = results_dir / "scenario_errors.csv"
        if error_path.exists():
            error_path.unlink()


def validate_inputs(args: argparse.Namespace) -> None:
    missing: list[Path] = []
    for dataset in args.datasets:
        dataset_dir = args.data_dir / dataset
        for filename in ("calibration.parquet", "test_pool.parquet"):
            path = dataset_dir / filename
            if not path.is_file():
                missing.append(path)
        for model in args.models:
            path = args.base_models_dir / "models" / dataset / f"{model}_calibrated.pkl"
            if not path.is_file():
                missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing benchmark inputs:\n{formatted}")

    training_manifest = args.base_models_dir / "training_manifest.json"
    if training_manifest.is_file() and not args.allow_model_version_mismatch:
        payload = json.loads(training_manifest.read_text(encoding="utf-8"))
        trained_environment = payload.get("environment", {})
        expected_versions = {
            "scikit_learn": sklearn.__version__,
            "xgboost": metadata.version("xgboost"),
        }
        mismatches = []
        for package, current_version in expected_versions.items():
            trained_version = trained_environment.get(package)
            if trained_version and trained_version != current_version:
                mismatches.append(
                    f"{package}: trained={trained_version}, current={current_version}"
                )
        if mismatches:
            raise RuntimeError(
                "Serialized base-model environment mismatch. Install the pinned "
                "requirements before loading models:\n  - "
                + "\n  - ".join(mismatches)
            )


def create_monitor(
    name: str,
    reference_X: pd.DataFrame,
    reference_y: pd.Series,
    model: Any,
    *,
    batch_size: int,
    null_calibration_batches: int,
    alarm_quantile: float,
) -> Any:
    factories: dict[str, Callable[[], Any]] = {
        "drift_shap": lambda: DriftSHAPMonitor(
            reference_X,
            reference_y,
            model,
            batch_size=batch_size,
            null_calibration_batches=null_calibration_batches,
            alarm_quantile=alarm_quantile,
        ),
        "confidence": lambda: ConfidenceShiftMonitor(
            reference_X,
            reference_y,
            model,
            batch_size=batch_size,
            null_calibration_batches=null_calibration_batches,
            alarm_quantile=alarm_quantile,
        ),
    }
    return factories[name]()


def generate_stream(
    generator: TabularShiftGenerator,
    scenario: Scenario,
    dataset_config: dict[str, Any],
) -> tuple[list[pd.DataFrame], dict[str, float]]:
    return _generate_stream_shared(generator, scenario, dataset_config)


def _shift_fraction(mode: str, batch_index: int, num_batches: int) -> float:
    return _shift_fraction_shared(mode, batch_index, num_batches)


def _realized_shift_diagnostics(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Measure the realized feature shift independently of nominal severity."""
    mean_changes = []
    wasserstein_distances = []
    support_violation_rates = []
    for feature in features:
        reference_values = pd.to_numeric(reference[feature], errors="coerce").to_numpy(
            dtype=float
        )
        target_values = pd.to_numeric(target[feature], errors="coerce").to_numpy(
            dtype=float
        )
        reference_values = reference_values[np.isfinite(reference_values)]
        target_values = target_values[np.isfinite(target_values)]
        if not len(reference_values) or not len(target_values):
            continue

        scale = float(np.std(reference_values)) + 1e-12
        mean_changes.append(
            abs(float(np.mean(target_values)) - float(np.mean(reference_values)))
            / scale
        )
        wasserstein_distances.append(
            float(wasserstein_distance(reference_values, target_values)) / scale
        )
        reference_min = float(np.min(reference_values))
        reference_max = float(np.max(reference_values))
        support_violation_rates.append(
            float(
                np.mean(
                    (target_values < reference_min) | (target_values > reference_max)
                )
            )
        )

    def safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "realized_mean_shift_std": safe_mean(mean_changes),
        "realized_wasserstein_std": safe_mean(wasserstein_distances),
        "realized_support_violation_rate": safe_mean(support_violation_rates),
    }


def _observable_reliability_diagnostics(
    reference_X: pd.DataFrame,
    target_X: pd.DataFrame,
    model: Any,
    *,
    random_seed: int,
    max_rows: int,
    density_ratio_clip: float,
) -> dict[str, float]:
    """Compute target-label-free signals that may predict monitor failure."""
    rng = np.random.default_rng(random_seed)
    n_reference = min(max_rows, len(reference_X))
    n_target = min(max_rows, len(target_X))
    reference_positions = rng.choice(len(reference_X), n_reference, replace=False)
    target_positions = rng.choice(len(target_X), n_target, replace=False)
    reference_sample = reference_X.iloc[reference_positions].reset_index(drop=True)
    target_sample = target_X.iloc[target_positions].reset_index(drop=True)

    combined = pd.concat([reference_sample, target_sample], ignore_index=True)
    encoded = pd.get_dummies(combined, dummy_na=True, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    domain_labels = np.concatenate(
        [np.zeros(n_reference, dtype=int), np.ones(n_target, dtype=int)]
    )
    indices = np.arange(len(encoded))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.3,
        random_state=random_seed,
        stratify=domain_labels,
    )
    classifier = LogisticRegression(
        max_iter=250, solver="liblinear", random_state=random_seed
    )
    classifier.fit(encoded.iloc[train_indices], domain_labels[train_indices])
    test_probabilities = classifier.predict_proba(encoded.iloc[test_indices])[:, 1]
    domain_auc = float(
        roc_auc_score(domain_labels[test_indices], test_probabilities)
    )

    target_encoded = encoded.iloc[n_reference:]
    target_domain_probability = np.clip(
        classifier.predict_proba(target_encoded)[:, 1], 1e-6, 1.0 - 1e-6
    )
    density_ratio = target_domain_probability / (1.0 - target_domain_probability)
    clipped_ratio = np.minimum(density_ratio, density_ratio_clip)
    effective_sample_size = (
        float(np.square(clipped_ratio.sum()) / np.square(clipped_ratio).sum())
        if np.square(clipped_ratio).sum() > 0
        else 0.0
    )

    reference_probability = np.asarray(
        model.predict_proba(reference_sample), dtype=float
    )
    target_probability = np.asarray(model.predict_proba(target_sample), dtype=float)

    def entropy(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-12, 1.0)
        return -np.sum(clipped * np.log(clipped), axis=1)

    reference_entropy = float(np.mean(entropy(reference_probability)))
    target_entropy = float(np.mean(entropy(target_probability)))
    reference_confidence = float(np.mean(np.max(reference_probability, axis=1)))
    target_confidence = float(np.mean(np.max(target_probability, axis=1)))

    violation_cells = 0
    total_cells = len(target_sample) * len(reference_sample.columns)
    row_violations = np.zeros(len(target_sample), dtype=bool)
    for feature in reference_sample.columns:
        reference_values = reference_sample[feature]
        target_values = target_sample[feature]
        if pd.api.types.is_numeric_dtype(reference_values):
            finite_reference = pd.to_numeric(
                reference_values, errors="coerce"
            ).to_numpy(dtype=float)
            finite_reference = finite_reference[np.isfinite(finite_reference)]
            target_numeric = pd.to_numeric(
                target_values, errors="coerce"
            ).to_numpy(dtype=float)
            if len(finite_reference):
                violation = (
                    (target_numeric < np.min(finite_reference))
                    | (target_numeric > np.max(finite_reference))
                    | (~np.isfinite(target_numeric) & reference_values.notna().all())
                )
            else:
                violation = np.zeros(len(target_sample), dtype=bool)
        else:
            known = set(reference_values.astype("string").fillna("<NA>"))
            target_tokens = target_values.astype("string").fillna("<NA>")
            violation = ~target_tokens.isin(known).to_numpy()
        violation_cells += int(np.sum(violation))
        row_violations |= np.asarray(violation, dtype=bool)

    return {
        "domain_classifier_auc": domain_auc,
        "effective_sample_size": effective_sample_size,
        "effective_sample_size_fraction": effective_sample_size / n_target,
        "density_ratio_clipping_rate": float(
            np.mean(density_ratio > density_ratio_clip)
        ),
        "prediction_entropy": target_entropy,
        "prediction_entropy_shift": target_entropy - reference_entropy,
        "prediction_confidence": target_confidence,
        "prediction_confidence_shift": target_confidence - reference_confidence,
        "novel_support_rate": violation_cells / total_cells if total_cells else 0.0,
        "novel_row_rate": float(np.mean(row_violations)),
    }


def _attribution_stability(
    previous: dict[str, float] | None,
    current: dict[str, float],
    k: int,
) -> dict[str, float]:
    """Compare consecutive label-free attribution vectors."""
    if previous is None:
        return {
            "attribution_l1_instability": float("nan"),
            "attribution_topk_jaccard": float("nan"),
        }
    features = sorted(set(previous) | set(current))
    previous_values = np.asarray([previous.get(f, 0.0) for f in features])
    current_values = np.asarray([current.get(f, 0.0) for f in features])
    if np.allclose(previous_values, 0.0) or np.allclose(current_values, 0.0):
        return {
            "attribution_l1_instability": float("nan"),
            "attribution_topk_jaccard": float("nan"),
        }
    ranking_k = min(k, len(features))
    previous_top = set(np.asarray(features)[np.argsort(np.abs(previous_values))[-ranking_k:]])
    current_top = set(np.asarray(features)[np.argsort(np.abs(current_values))[-ranking_k:]])
    union = previous_top | current_top
    return {
        "attribution_l1_instability": float(
            0.5 * np.sum(np.abs(previous_values - current_values))
        ),
        "attribution_topk_jaccard": (
            float(len(previous_top & current_top) / len(union)) if union else 1.0
        ),
    }


def _resolve_alarm_target(
    capabilities: dict[str, Any],
    *,
    true_risk_event: bool,
    distribution_shift_event: bool,
) -> tuple[str | None, bool | None]:
    """Return the declared alarm estimand and its offline event label."""
    supports_alarm = bool(capabilities.get("alarm", False))
    if not supports_alarm:
        return None, None

    alarm_target = capabilities.get("alarm_target")
    if alarm_target not in ALARM_TARGETS:
        raise ValueError(
            "Alarm-capable monitors must declare capabilities['alarm_target'] "
            f"as one of {sorted(ALARM_TARGETS)}; got {alarm_target!r}"
        )
    if alarm_target == "risk_event":
        return alarm_target, bool(true_risk_event)
    return alarm_target, bool(distribution_shift_event)


def run_scenario(
    scenario: Scenario,
    test_pool: pd.DataFrame,
    target_col: str,
    evaluator: OracleEvaluator,
    monitor: Any,
    monitor_init_seconds: float,
    protocol: TABMONProtocol,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started_at = time.perf_counter()
    generator = TabularShiftGenerator(
        test_pool, target_col, random_seed=scenario.seed
    )
    stream, ground_truth_attr = generate_stream(
        generator, scenario, DATASET_CONFIG[scenario.dataset]
    )
    config = DATASET_CONFIG[scenario.dataset]
    diagnostic_features = (
        list(config["corr_features"])
        if scenario.shift == "correlated"
        else [config["shift_feature"]]
    )

    batch_records: list[dict[str, Any]] = []
    alarms: list[bool] = []
    true_risk_events: list[bool] = []
    distribution_shift_events: list[bool] = []
    alarm_target_events: list[bool] = []
    previous_attribution: dict[str, float] | None = None
    for batch_index, target_batch in enumerate(stream):
        shift_fraction = (
            0.0
            if scenario.shift == "no_shift"
            else _shift_fraction(scenario.mode, batch_index, scenario.num_batches)
        )
        batch_ground_truth_attr = {
            feature: value * shift_fraction
            for feature, value in ground_truth_attr.items()
        }
        target_X = target_batch.drop(columns=[target_col])
        if target_col in target_X.columns:
            raise RuntimeError("Label isolation failure: target reached monitor features")

        true_delta_loss = evaluator.calculate_true_excess_risk(target_batch)
        report, efficiency = protocol.profile_monitor(
            monitor.analyze_batch, target_X
        )
        scores = protocol.evaluate_batch(
            report, batch_ground_truth_attr, true_delta_loss
        )
        realized_shift = _realized_shift_diagnostics(
            test_pool, target_batch, diagnostic_features
        )
        observable = (
            _observable_reliability_diagnostics(
                monitor.reference_X,
                target_X,
                monitor.model,
                random_seed=scenario.seed + 10_007 * batch_index,
                max_rows=args.diagnostic_max_rows,
                density_ratio_clip=args.density_ratio_clip,
            )
            if args.observable_diagnostics
            else {}
        )

        estimated_risk = report.get("estimated_risk_change")
        monitor_score = report.get("monitor_score", estimated_risk)
        alarm = report.get("alarm")
        capabilities = report.get("capabilities", {})
        supports_risk = bool(capabilities.get("risk_estimation", False))
        supports_attribution = bool(capabilities.get("attribution", False))
        supports_alarm = bool(capabilities.get("alarm", alarm is not None))
        if supports_alarm and alarm is None:
            raise ValueError("Registered monitors must return a calibrated alarm")
        alarms.append(bool(alarm) if alarm is not None else False)

        predicted_attribution = report.get("feature_attribution", {})
        stability = (
            _attribution_stability(
                previous_attribution, predicted_attribution, protocol.k_features
            )
            if supports_attribution
            else {
                "attribution_l1_instability": np.nan,
                "attribution_topk_jaccard": np.nan,
            }
        )
        if supports_attribution:
            previous_attribution = dict(predicted_attribution)

        risk_abs_error = scores.get("risk_abs_error")
        risk_failure = (
            bool(float(risk_abs_error) > args.risk_failure_tolerance)
            if supports_risk
            and risk_abs_error is not None
            and np.isfinite(risk_abs_error)
            else None
        )
        attribution_recall = scores.get(f"top{protocol.k_features}_recall")
        attribution_failure = (
            bool(attribution_recall < args.attribution_failure_threshold)
            if supports_attribution
            and attribution_recall is not None
            and np.isfinite(attribution_recall)
            else None
        )
        true_risk_event = bool(true_delta_loss > args.risk_event_threshold)
        distribution_shift_event = bool(
            scenario.shift != "no_shift" and shift_fraction > 0.0
        )
        alarm_target, alarm_target_event = _resolve_alarm_target(
            capabilities,
            true_risk_event=true_risk_event,
            distribution_shift_event=distribution_shift_event,
        )
        alarm_failure = (
            bool(bool(alarm) != alarm_target_event) if supports_alarm else None
        )
        # Filled after the full trajectory is available. A valid early warning
        # must precede an oracle risk event that actually occurs later.
        early_warning = False
        true_risk_events.append(true_risk_event)
        distribution_shift_events.append(distribution_shift_event)
        alarm_target_events.append(
            bool(alarm_target_event) if alarm_target_event is not None else False
        )
        applicable_failures = [
            value
            for value in (risk_failure, attribution_failure, alarm_failure)
            if value is not None
        ]

        record: dict[str, Any] = {
            "batch_index": batch_index,
            "shift_fraction": shift_fraction,
            "true_excess_risk": true_delta_loss,
            "monitor_score": monitor_score,
            "estimated_risk_change": estimated_risk,
            "estimated_risk_ci_lower": report.get("estimated_risk_ci_lower"),
            "estimated_risk_ci_upper": report.get("estimated_risk_ci_upper"),
            "alarm": alarm,
            "alarm_threshold": report.get("alarm_threshold"),
            "alarm_p_value": report.get("alarm_p_value"),
            "supports_risk_estimation": supports_risk,
            "supports_attribution": supports_attribution,
            "supports_alarm": supports_alarm,
            "alarm_target": alarm_target,
            "alarm_direction": capabilities.get(
                "alarm_direction", report.get("alarm_direction")
            ),
            "true_risk_event": true_risk_event,
            "distribution_shift_event": distribution_shift_event,
            "alarm_target_event": alarm_target_event,
            "early_warning": early_warning,
            "risk_failure": risk_failure,
            "attribution_failure": attribution_failure,
            "alarm_failure": alarm_failure,
            "monitor_failure": (
                bool(any(applicable_failures)) if applicable_failures else None
            ),
            "ground_truth_attribution_json": json.dumps(
                _json_ready(batch_ground_truth_attr), sort_keys=True
            ),
            "predicted_attribution_json": json.dumps(
                _json_ready(predicted_attribution), sort_keys=True
            ),
            "raw_predicted_attribution_json": json.dumps(
                _json_ready(report.get("raw_feature_attribution", {})), sort_keys=True
            ),
            "attribution_abs_mass": report.get("attribution_abs_mass"),
            "attribution_max_abs": report.get("attribution_max_abs"),
            "attribution_mass_relative_to_null": report.get(
                "attribution_mass_relative_to_null"
            ),
            "null_score_median": report.get("null_score_median"),
            "risk_calibration_residual_std": report.get(
                "risk_calibration_residual_std"
            ),
            **scores,
            **efficiency,
            **realized_shift,
            **observable,
            **stability,
        }
        batch_records.append(record)

    batch_frame = pd.DataFrame(batch_records)
    risk_event_indices = np.flatnonzero(np.asarray(true_risk_events, dtype=bool))
    if len(risk_event_indices):
        first_risk_event_batch = int(risk_event_indices[0])
        early_warning_flags = (
            batch_frame["alarm"].astype(bool)
            & batch_frame["distribution_shift_event"].astype(bool)
            & batch_frame["batch_index"].lt(first_risk_event_batch)
        )
        batch_frame["early_warning"] = early_warning_flags
        for record, is_early in zip(batch_records, early_warning_flags):
            record["early_warning"] = bool(is_early)
    true_risks = batch_frame["true_excess_risk"].to_numpy(dtype=float)
    predicted_risks = pd.to_numeric(
        batch_frame["estimated_risk_change"], errors="coerce"
    ).to_numpy(dtype=float)
    valid_risk = np.isfinite(true_risks) & np.isfinite(predicted_risks)

    aggregate: dict[str, Any] = {
        "scenario_runtime_seconds": time.perf_counter() - started_at,
        "monitor_init_seconds": monitor_init_seconds,
        "n_batches_completed": len(batch_records),
        "true_excess_risk_mean": float(np.mean(true_risks)),
        "true_excess_risk_std": float(np.std(true_risks)),
        "estimated_risk_mean": (
            float(np.mean(predicted_risks[valid_risk])) if valid_risk.any() else np.nan
        ),
        "estimated_risk_std": (
            float(np.std(predicted_risks[valid_risk])) if valid_risk.any() else np.nan
        ),
        "supports_risk_estimation": bool(
            batch_frame["supports_risk_estimation"].all()
        ),
        "supports_attribution": bool(batch_frame["supports_attribution"].all()),
        "supports_alarm": bool(batch_frame["supports_alarm"].all()),
        "alarm_target": (
            str(batch_frame["alarm_target"].dropna().iloc[0])
            if batch_frame["alarm_target"].notna().any()
            else None
        ),
        "attribution_scored_batches": (
            int(
                pd.to_numeric(
                    batch_frame[f"top{protocol.k_features}_recall"],
                    errors="coerce",
                ).notna().sum()
            )
            if f"top{protocol.k_features}_recall" in batch_frame
            else 0
        ),
        "risk_failure_tolerance": args.risk_failure_tolerance,
        "attribution_failure_threshold": args.attribution_failure_threshold,
        "risk_event_threshold": args.risk_event_threshold,
    }
    if valid_risk.any():
        aggregate.update(
            compute_risk_metrics(true_risks[valid_risk], predicted_risks[valid_risk])
        )
    else:
        aggregate.update(
            {key: np.nan for key in ("risk_mae", "risk_rmse", "risk_pearson", "risk_spearman")}
        )

    mean_columns = [
        "risk_abs_error", f"ndcg@{protocol.k_features}",
        f"top{protocol.k_features}_recall", "sign_accuracy",
        "false_attribution_mass", "false_attribution_mass_normalized",
        "coverage", "interval_width",
        "runtime_seconds", "peak_ram_mb", "realized_mean_shift_std",
        "realized_wasserstein_std", "realized_support_violation_rate",
        "risk_failure", "attribution_failure", "alarm_failure",
        "monitor_failure", "alarm", "alarm_p_value", "true_risk_event",
        "distribution_shift_event", "alarm_target_event", "early_warning",
        "monitor_score",
        "attribution_abs_mass", "attribution_max_abs",
        "attribution_mass_relative_to_null", "attribution_l1_instability",
        "attribution_topk_jaccard", "domain_classifier_auc",
        "effective_sample_size", "effective_sample_size_fraction",
        "density_ratio_clipping_rate", "prediction_entropy",
        "prediction_entropy_shift", "prediction_confidence",
        "prediction_confidence_shift", "novel_support_rate", "novel_row_rate",
    ]
    for column in mean_columns:
        if column in batch_frame:
            aggregate[f"{column}_mean"] = float(
                pd.to_numeric(batch_frame[column], errors="coerce").mean()
            )

    pre_mask = batch_frame["shift_fraction"] == 0.0
    post_mask = batch_frame["shift_fraction"] > 0.0
    phase_columns = [
        "true_excess_risk",
        "estimated_risk_change",
        "realized_mean_shift_std",
        "realized_wasserstein_std",
        "realized_support_violation_rate",
        "false_attribution_mass",
        "false_attribution_mass_normalized",
        "attribution_abs_mass",
        "attribution_max_abs",
        "attribution_mass_relative_to_null",
        "risk_failure",
        "attribution_failure",
        "alarm_failure",
        "monitor_failure",
        "alarm",
        "alarm_p_value",
        "true_risk_event",
        "distribution_shift_event",
        "alarm_target_event",
        "early_warning",
        "monitor_score",
        "domain_classifier_auc",
        "effective_sample_size_fraction",
        "density_ratio_clipping_rate",
        "prediction_entropy_shift",
        "prediction_confidence_shift",
        "novel_support_rate",
        "novel_row_rate",
    ]
    for phase_name, phase_mask in (("pre", pre_mask), ("post", post_mask)):
        for column in phase_columns:
            if column not in batch_frame:
                continue
            values = pd.to_numeric(
                batch_frame.loc[phase_mask, column], errors="coerce"
            )
            aggregate[f"{phase_name}_{column}_mean"] = (
                float(values.mean()) if values.notna().any() else np.nan
            )
    aggregate["realized_true_risk_increase"] = (
        aggregate["post_true_excess_risk_mean"]
        - aggregate["pre_true_excess_risk_mean"]
    )
    aggregate["realized_estimated_risk_response"] = (
        aggregate["post_estimated_risk_change_mean"]
        - aggregate["pre_estimated_risk_change_mean"]
    )
    pre_mass = aggregate.get("pre_attribution_abs_mass_mean", np.nan)
    post_mass = aggregate.get("post_attribution_abs_mass_mean", np.nan)
    aggregate["pre_to_post_attribution_mass_ratio"] = (
        pre_mass / post_mass
        if np.isfinite(pre_mass) and np.isfinite(post_mass) and post_mass > 0
        else np.nan
    )
    target_metrics = compute_alarm_event_metrics(alarms, alarm_target_events)
    aggregate.update(
        {
            "alarm_target_event_batch": target_metrics["event_batch"],
            "first_alarm_batch": target_metrics["first_alarm_batch"],
            "false_alarm_rate": target_metrics["false_alarm_rate"],
            "power": target_metrics["power"],
            "detection_delay": target_metrics["detection_delay"],
            "missed_alarm": target_metrics["missed_alarm"],
        }
    )
    for prefix, event_flags in (
        ("risk_alarm", true_risk_events),
        ("shift_alarm", distribution_shift_events),
    ):
        metrics = compute_alarm_event_metrics(alarms, event_flags)
        for name, value in metrics.items():
            aggregate[f"{prefix}_{name}"] = value

    # Keep the v5 oracle-risk event timestamp as an explicitly named
    # compatibility field. Generic alarm metrics above follow alarm_target.
    risk_metrics = compute_event_based_sequential_metrics(
        alarms, true_risks, args.risk_event_threshold
    )
    aggregate["true_risk_event_batch"] = risk_metrics["true_risk_event_batch"]
    aggregate["early_warning_count"] = int(batch_frame["early_warning"].sum())

    return aggregate, batch_records


def write_manifest(args: argparse.Namespace, scenarios: list[Scenario]) -> None:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "shap"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "created_at_utc": _utc_now(),
        "schema_version": SCHEMA_VERSION,
        "scenario_count": len(scenarios),
        "configuration": {
            key: _json_ready(value)
            for key, value in vars(args).items()
            if key not in {"no_resume"}
        },
        "dataset_config": DATASET_CONFIG,
        "methodology": {
            "target_labels_visible_to_monitors": False,
            "alarm_calibration_source": "resampled calibration features only",
            "interval_estimand": (
                "calibrated excess log loss for risk-capable monitors; "
                "not applicable to proxy-only monitors"
            ),
            "capability_aware_scoring": True,
            "alarm_target_aware_scoring": True,
            "alarm_direction_aware_scoring": True,
            "alarm_targets": {
                "confidence": "risk_event",
                "drift_shap": "distribution_shift",
            },
            "alarm_directions": {
                "confidence": "increase",
                "drift_shap": "increase",
            },
            "early_warning_definition": (
                "alarm after controlled shift onset but before oracle risk event"
            ),
            "monitor_failure_components": [
                "risk_failure when risk_estimation is supported",
                "attribution_failure when attribution is supported and defined",
                "alarm_failure against the monitor-declared alarm target",
            ],
            "failure_labels_use_oracle_offline_only": True,
            "null_scenario": {
                "severity": "none",
                "mode": "static",
                "deduplicated_across_requested_severities_and_modes": True,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "packages": packages,
        },
    }
    _write_json_atomic(manifest, args.results_dir / "benchmark_manifest.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir = args.data_dir.resolve()
    args.base_models_dir = args.base_models_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    validate_inputs(args)

    scenarios = build_scenarios(args)
    write_manifest(args, scenarios)
    connection = open_checkpoint(args.results_dir / "benchmark_checkpoint.sqlite")
    if args.no_resume:
        reset_scenarios(connection, scenarios)

    requested_ids = {scenario.scenario_id for scenario in scenarios}
    already_completed = completed_scenario_ids(connection) & requested_ids
    total = len(scenarios)
    started_at = time.monotonic()
    deadline = (
        started_at + args.max_hours * 3600
        if args.max_hours is not None
        else None
    )
    stopped_for_time = False
    interrupted = False
    errors_this_run = 0

    print("=== TABMON-BENCH CORE RUNNER ===", flush=True)
    print(f"Scenarios: {total}", flush=True)
    print(f"Already complete: {len(already_completed)}", flush=True)
    print(f"Data: {args.data_dir}", flush=True)
    print(f"Base models: {args.base_models_dir}", flush=True)
    print(f"Results: {args.results_dir}", flush=True)
    if args.max_hours is not None:
        print(f"Soft time limit: {args.max_hours:.2f} hours", flush=True)

    current_resource_key: tuple[str, str, str] | None = None
    resource_objects: tuple[Any, ...] | None = None
    fatal_error = False

    try:
        for scenario in scenarios:
            if scenario.scenario_id in already_completed:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                stopped_for_time = True
                print("[time-limit] Stopping before the next scenario.", flush=True)
                break

            resource_key = (scenario.dataset, scenario.model, scenario.monitor)
            if resource_key != current_resource_key:
                if resource_objects is not None:
                    del resource_objects
                    gc.collect()

                config = DATASET_CONFIG[scenario.dataset]
                target_col = config["target"]
                dataset_dir = args.data_dir / scenario.dataset
                reference = pd.read_parquet(dataset_dir / "calibration.parquet")
                test_pool = pd.read_parquet(dataset_dir / "test_pool.parquet")
                required_features = {target_col}
                if set(args.shifts) & {
                    "no_shift", "covariate", "support", "pipeline", "concept"
                }:
                    required_features.add(config["shift_feature"])
                if "correlated" in args.shifts:
                    required_features.update(config["corr_features"])
                missing_features = required_features - set(test_pool.columns)
                if missing_features:
                    raise ValueError(
                        f"{scenario.dataset} is missing configured columns: "
                        f"{sorted(missing_features)}"
                    )
                model_path = (
                    args.base_models_dir / "models" / scenario.dataset
                    / f"{scenario.model}_calibrated.pkl"
                )
                model = joblib.load(model_path)
                evaluator = OracleEvaluator(model, reference, target_col)
                reference_X = reference.drop(columns=[target_col])
                reference_y = reference[target_col]
                monitor_started = time.perf_counter()
                monitor = create_monitor(
                    scenario.monitor,
                    reference_X,
                    reference_y,
                    model,
                    batch_size=args.batch_size,
                    null_calibration_batches=args.null_calibration_batches,
                    alarm_quantile=args.alarm_quantile,
                )
                monitor_init_seconds = time.perf_counter() - monitor_started
                protocol = TABMONProtocol(k_features=args.k_features)
                resource_objects = (
                    test_pool, target_col, evaluator, monitor,
                    monitor_init_seconds, protocol,
                )
                current_resource_key = resource_key

            (
                test_pool, target_col, evaluator, monitor,
                monitor_init_seconds, protocol,
            ) = resource_objects
            completed_now = len(completed_scenario_ids(connection) & requested_ids)
            print(
                f"[{completed_now + 1}/{total}] {scenario.dataset}/{scenario.model}/"
                f"{scenario.monitor}/{scenario.shift}/{scenario.severity}/"
                f"{scenario.mode}/seed={scenario.seed}",
                flush=True,
            )
            try:
                aggregate, batches = run_scenario(
                    scenario,
                    test_pool,
                    target_col,
                    evaluator,
                    monitor,
                    monitor_init_seconds,
                    protocol,
                    args,
                )
                checkpoint_success(
                    connection, scenario, aggregate, batches, args.save_batch_results
                )
                completed_now += 1
                print(
                    f"[ok] Experiment progress: {completed_now}/{total} "
                    f"({100 * completed_now / total:.1f}%)",
                    flush=True,
                )
            except Exception as exc:
                errors_this_run += 1
                checkpoint_error(connection, scenario, exc)
                print(
                    f"[error] {scenario.scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = True
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("[interrupt] Exporting completed checkpoints.", flush=True)
    except Exception as exc:
        errors_this_run += 1
        fatal_error = True
        if "scenario" in locals():
            checkpoint_error(connection, scenario, exc)
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        export_checkpoint(connection, args.results_dir)
        completed_final = len(completed_scenario_ids(connection) & requested_ids)
        status = {
            "updated_at_utc": _utc_now(),
            "complete": completed_final == total and errors_this_run == 0,
            "stopped_for_time_limit": stopped_for_time,
            "interrupted": interrupted,
            "completed_scenarios": completed_final,
            "requested_scenarios": total,
            "progress_percent": 100 * completed_final / total if total else 100.0,
            "errors_this_run": errors_this_run,
            "elapsed_hours": (time.monotonic() - started_at) / 3600,
        }
        _write_json_atomic(status, args.results_dir / "benchmark_status.json")
        connection.close()

    print(
        f"Final experiment progress: {completed_final}/{total} "
        f"({100 * completed_final / total:.1f}%)",
        flush=True,
    )
    if stopped_for_time:
        print("Run the same command again to resume remaining scenarios.")
    return 1 if fatal_error or errors_this_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
