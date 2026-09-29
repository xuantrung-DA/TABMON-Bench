"""Leakage-safe cache primitives for reusable TABMON predictor streams.

Target features and model probabilities are physically separated from target
labels and oracle losses.  Monitor implementations receive an
``ObservableCacheReader`` only; offline scoring receives a separate
``OracleCacheReader``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.stream_protocol import generate_stream, shift_fraction
from src.shift_generator import TabularShiftGenerator
from src.evaluation.risk import binary_log_losses, multiclass_log_losses


CACHE_SCHEMA_VERSION = 1
BATCH_INDEX = "_tabmon_batch_index"
ROW_INDEX = "_tabmon_row_index"
KEY_COLUMNS = [BATCH_INDEX, ROW_INDEX]
TARGET_LABEL = "target_label"
SOURCE_LABEL = "source_calibration_label"
FORBIDDEN_OBSERVABLE_COLUMNS = {
    TARGET_LABEL,
    "target_labels",
    "y_true",
    "oracle_risk",
    "true_excess_risk",
    "sample_log_loss",
    "risk_failure",
    "attribution_failure",
    "alarm_failure",
    "monitor_failure",
    "ground_truth_attribution_json",
    "shift_fraction",
}


def _stable_id(payload: dict[str, Any], length: int = 24) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class StreamSpec:
    dataset: str
    shift: str
    severity: str
    mode: str
    seed: int
    num_batches: int
    batch_size: int

    @property
    def stream_id(self) -> str:
        return _stable_id(
            {"cache_schema_version": CACHE_SCHEMA_VERSION, **asdict(self)}
        )


@dataclass(frozen=True)
class PredictorStreamSpec:
    stream_id: str
    dataset: str
    model: str

    @property
    def predictor_stream_id(self) -> str:
        return _stable_id(
            {"cache_schema_version": CACHE_SCHEMA_VERSION, **asdict(self)}
        )


def build_stream_specs(
    datasets: Iterable[str],
    shifts: Iterable[str],
    severities: Iterable[str],
    modes: Iterable[str],
    seeds: Iterable[int],
    num_batches: int,
    batch_size: int,
) -> list[StreamSpec]:
    """Build monitor-independent streams with one deduplicated null per seed."""
    shifts = list(shifts)
    specs: list[StreamSpec] = []
    for dataset, seed in product(datasets, seeds):
        if "no_shift" in shifts:
            specs.append(
                StreamSpec(
                    dataset,
                    "no_shift",
                    "none",
                    "static",
                    int(seed),
                    num_batches,
                    batch_size,
                )
            )
        for shift, severity, mode in product(
            [name for name in shifts if name != "no_shift"], severities, modes
        ):
            specs.append(
                StreamSpec(
                    dataset,
                    shift,
                    severity,
                    mode,
                    int(seed),
                    num_batches,
                    batch_size,
                )
            )
    return specs


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def assert_observable_columns(columns: Iterable[str]) -> None:
    columns = set(columns)
    forbidden = sorted(columns & FORBIDDEN_OBSERVABLE_COLUMNS)
    if forbidden:
        raise ValueError(f"Oracle columns reached observable cache: {forbidden}")


def materialize_stream(
    spec: StreamSpec,
    test_pool: pd.DataFrame,
    target_column: str,
    dataset_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate one deterministic stream and split observable/oracle records."""
    generator = TabularShiftGenerator(test_pool, target_column, spec.seed)
    batches, intervention = generate_stream(generator, spec, dataset_config)
    observable_parts = []
    oracle_parts = []
    batch_records = []
    for batch_index, batch in enumerate(batches):
        target_X = batch.drop(columns=[target_column]).reset_index(drop=True)
        assert_observable_columns(target_X.columns)
        keys = pd.DataFrame(
            {
                BATCH_INDEX: np.full(len(batch), batch_index, dtype=np.int16),
                ROW_INDEX: np.arange(len(batch), dtype=np.int32),
            }
        )
        observable_parts.append(pd.concat([keys, target_X], axis=1))
        oracle_parts.append(
            pd.concat(
                [
                    keys,
                    batch[target_column]
                    .reset_index(drop=True)
                    .rename(TARGET_LABEL),
                ],
                axis=1,
            )
        )
        fraction = shift_fraction(spec.mode, batch_index, spec.num_batches)
        batch_records.append(
            {
                BATCH_INDEX: batch_index,
                "shift_fraction": fraction,
                "ground_truth_attribution_json": json.dumps(
                    {
                        feature: float(value) * fraction
                        for feature, value in intervention.items()
                    },
                    sort_keys=True,
                ),
            }
        )
    observable = pd.concat(observable_parts, ignore_index=True)
    oracle_targets = pd.concat(oracle_parts, ignore_index=True)
    oracle_batches = pd.DataFrame(batch_records)
    if len(observable) != spec.num_batches * spec.batch_size:
        raise RuntimeError("Stream cache row count does not match configuration")
    return observable, oracle_targets, oracle_batches


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in KEY_COLUMNS]


def predict_frame(
    model: Any, observable: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute reusable probabilities without copying target features."""
    assert_observable_columns(observable.columns)
    X = observable[feature_columns(observable)]
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba output does not match model classes")
    # Persist probabilities as float32 to keep the reusable cache compact.
    # Derive the stored class from that exact representation so a near-0.5
    # float64 value cannot change its argmax after serialization.
    stored_probabilities = probabilities.astype(np.float32)
    predictions = classes[np.argmax(stored_probabilities, axis=1)]
    result = observable[KEY_COLUMNS].copy()
    for index in range(stored_probabilities.shape[1]):
        result[f"probability_class_{index}"] = stored_probabilities[:, index]
    result["predicted_class"] = predictions
    assert_observable_columns(result.columns)
    return result, probabilities, classes


def losses_for_labels(
    probabilities: np.ndarray, classes: np.ndarray, labels: pd.Series
) -> np.ndarray:
    if len(classes) == 2:
        return binary_log_losses(probabilities, classes, labels.to_numpy())
    return multiclass_log_losses(probabilities, classes, labels.to_numpy())


def oracle_batch_risk(
    oracle_targets: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
    reference_risk: float,
) -> pd.DataFrame:
    losses = losses_for_labels(probabilities, classes, oracle_targets[TARGET_LABEL])
    working = oracle_targets[KEY_COLUMNS].copy()
    working["sample_log_loss"] = losses
    result = (
        working.groupby(BATCH_INDEX, sort=True)["sample_log_loss"]
        .agg(["mean", "std", "size"])
        .reset_index()
        .rename(
            columns={
                "mean": "target_log_loss",
                "std": "target_log_loss_std",
                "size": "batch_size",
            }
        )
    )
    result["true_excess_risk"] = result["target_log_loss"] - reference_risk
    return result


class ObservableCacheReader:
    """Read-only monitor-facing API with no method for target oracle access."""

    def __init__(self, cache_root: Path):
        self.root = Path(cache_root).resolve() / "observable"
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def load_target_features(self, stream_id: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.root / "streams" / f"{stream_id}.parquet")
        assert_observable_columns(frame.columns)
        return frame[feature_columns(frame)]

    def load_target_features_with_keys(self, stream_id: str) -> pd.DataFrame:
        """Return monitor-visible target features with stable batch/row keys.

        Window-adaptive monitors such as PAPE must fit separately on each
        deployment batch.  The keys expose that partition without opening any
        target label, intervention target, or oracle quantity.
        """

        frame = pd.read_parquet(self.root / "streams" / f"{stream_id}.parquet")
        assert_observable_columns(frame.columns)
        missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Observable feature stream is missing keys: {missing}")
        return frame[KEY_COLUMNS + feature_columns(frame)]

    def load_target_probabilities(self, predictor_stream_id: str) -> pd.DataFrame:
        frame = pd.read_parquet(
            self.root / "predictions" / f"{predictor_stream_id}.parquet"
        )
        assert_observable_columns(frame.columns)
        return frame

    def load_reference_features(self, dataset: str) -> pd.DataFrame:
        frame = pd.read_parquet(
            self.root / "references" / dataset / "features.parquet"
        )
        assert_observable_columns(frame.columns)
        return frame.drop(columns=["_tabmon_reference_row"])

    def load_source_calibration_labels(self, dataset: str) -> pd.Series:
        frame = pd.read_parquet(
            self.root / "references" / dataset / "source_labels.parquet"
        )
        if list(frame.columns) != ["_tabmon_reference_row", SOURCE_LABEL]:
            raise ValueError("Invalid source-calibration label schema")
        return frame[SOURCE_LABEL]

    def load_reference_probabilities(self, dataset: str, model: str) -> pd.DataFrame:
        frame = pd.read_parquet(
            self.root
            / "reference_predictions"
            / dataset
            / f"{model}.parquet"
        )
        assert_observable_columns(frame.columns)
        return frame


class OracleCacheReader:
    """Offline-only reader for hidden target labels and evaluation quantities."""

    def __init__(self, cache_root: Path):
        self.root = Path(cache_root).resolve() / "oracle"
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def load_target_labels(self, stream_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / "targets" / f"{stream_id}.parquet")

    def load_intervention_targets(self, stream_id: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root / "interventions" / f"{stream_id}.parquet"
        )

    def load_batch_risk(self, predictor_stream_id: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root / "batch_risk" / f"{predictor_stream_id}.parquet"
        )
