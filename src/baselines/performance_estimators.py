"""Label-free classification-performance estimators used in Phase 5.

All estimators are calibrated exclusively from labeled source-calibration data
and consume only target prediction probabilities at monitoring time.  Their
estimand is 0--1 classification error (and its complement, accuracy), never
log loss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


METHODS = ("ac", "doc", "atc", "cot", "cott")
ENDPOINT = "classification_error"
MULTICLASS_COTT_CALIBRATION_LIMIT = 5000
PREDICTION_AUXILIARY_COLUMNS = {
    "_tabmon_batch_index",
    "_tabmon_row_index",
    "_tabmon_reference_row",
    "predicted_class",
}


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [name for name in frame if name.startswith("probability_class_")]
    columns.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if len(columns) < 2:
        raise ValueError("At least two class-probability columns are required")
    return columns


def probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = probability_columns(frame)
    unexpected = sorted(set(frame.columns) - set(columns) - PREDICTION_AUXILIARY_COLUMNS)
    if unexpected:
        raise ValueError(f"Non-allowlisted prediction columns: {unexpected}")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities contain non-finite values")
    if np.any(values < -1e-7) or np.any(values > 1.0 + 1e-7):
        raise ValueError("Probabilities fall outside [0, 1]")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Probability rows must have positive mass")
    return np.clip(values / row_sums, 0.0, 1.0)


@dataclass(frozen=True)
class TailThreshold:
    """Upper-tail threshold with deterministic fractional handling of ties."""

    threshold: float
    tie_weight: float
    requested_rate: float
    fitted_rate: float

    def rate(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            raise ValueError("Cannot evaluate an empty score vector")
        if np.isposinf(self.threshold):
            return 0.0
        if np.isneginf(self.threshold):
            return 1.0
        above = scores > self.threshold
        tied = scores == self.threshold
        return float(np.mean(above) + self.tie_weight * np.mean(tied))


def fit_upper_tail_threshold(scores: np.ndarray, rate: float) -> TailThreshold:
    """Fit P(score > t) + w P(score = t) to an empirical target rate.

    Fractional tie weighting is the deterministic equivalent of randomized
    thresholding and prevents isotonic-calibration plateaus from making the
    calibration result depend on row order.
    """

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Threshold scores must be a non-empty vector")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Threshold scores contain non-finite values")
    rate = float(np.clip(rate, 0.0, 1.0))
    if rate == 0.0:
        return TailThreshold(float("inf"), 0.0, rate, 0.0)
    if rate == 1.0:
        return TailThreshold(float("-inf"), 0.0, rate, 1.0)

    target_count = rate * len(scores)
    unique, counts = np.unique(scores, return_counts=True)
    cumulative_above = 0.0
    for threshold, count in zip(unique[::-1], counts[::-1]):
        if cumulative_above + count >= target_count:
            tie_weight = (target_count - cumulative_above) / count
            fitted = (cumulative_above + tie_weight * count) / len(scores)
            return TailThreshold(
                float(threshold), float(tie_weight), rate, float(fitted)
            )
        cumulative_above += count
    raise RuntimeError("Unable to fit empirical threshold")


def binary_transport_costs(
    probabilities: np.ndarray, source_class_mass: np.ndarray
) -> np.ndarray:
    """Return per-sample optimal L-infinity transport costs for two classes.

    The target confidence vectors have equal mass.  Sorting the difference
    between transport-to-class costs gives the exact solution of the binary
    transportation problem, including at most one fractionally split row.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    source_class_mass = np.asarray(source_class_mass, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("The current TABMON COT implementation requires binary data")
    if source_class_mass.shape != (2,):
        raise ValueError("Expected two source class masses")
    if not np.isclose(source_class_mass.sum(), 1.0, atol=1e-10):
        raise ValueError("Source class masses must sum to one")

    one_hot = np.eye(2, dtype=np.float64)
    costs = np.max(
        np.abs(probabilities[:, None, :] - one_hot[None, :, :]), axis=2
    )
    allocation_to_zero = np.zeros(len(probabilities), dtype=np.float64)
    remaining = float(source_class_mass[0] * len(probabilities))
    order = np.argsort(costs[:, 0] - costs[:, 1], kind="stable")
    for index in order:
        if remaining <= 0:
            break
        assigned = min(1.0, remaining)
        allocation_to_zero[index] = assigned
        remaining -= assigned
    if remaining > 1e-8:
        raise RuntimeError("Binary transport allocation did not satisfy class mass")
    return (
        allocation_to_zero * costs[:, 0]
        + (1.0 - allocation_to_zero) * costs[:, 1]
    )


def multiclass_transport_costs(
    probabilities: np.ndarray, source_class_mass: np.ndarray
) -> np.ndarray:
    """Return per-sample COT costs for an arbitrary class count.

    For two classes the frozen exact implementation is retained.  For three
    or more classes, POT solves the empirical transport problem from target
    probability vectors to the source-label class proportions.  Multiplying
    row transport mass by ``n`` converts the coupling contribution back to a
    per-observation cost.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    source_class_mass = np.asarray(source_class_mass, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected an n-by-C probability matrix with C >= 2")
    if source_class_mass.shape != (probabilities.shape[1],):
        raise ValueError("Source class masses do not match probability columns")
    if np.any(source_class_mass < 0) or not np.isclose(
        source_class_mass.sum(), 1.0, atol=1e-10
    ):
        raise ValueError("Source class masses must be nonnegative and sum to one")
    if probabilities.shape[1] == 2:
        return binary_transport_costs(probabilities, source_class_mass)

    try:
        import ot
    except ImportError as exc:  # pragma: no cover - exercised by env validation
        raise ImportError("Multiclass COT requires the 'pot' package") from exc
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)
    costs = np.max(
        np.abs(probabilities[:, None, :] - one_hot[None, :, :]), axis=2
    )
    source_weights = np.full(len(probabilities), 1.0 / len(probabilities))
    coupling = ot.emd(
        source_weights,
        source_class_mass,
        costs,
        numItermax=10_000_000,
        center_dual=False,
    )
    if not np.allclose(coupling.sum(axis=1), source_weights, atol=1e-7):
        raise RuntimeError("COT coupling does not preserve target sample mass")
    if not np.allclose(coupling.sum(axis=0), source_class_mass, atol=1e-7):
        raise RuntimeError("COT coupling does not preserve source class mass")
    return len(probabilities) * np.sum(coupling * costs, axis=1)


@dataclass
class PerformanceEstimatorSuite:
    classes: np.ndarray
    source_accuracy: float
    source_mean_confidence: float
    source_class_mass: np.ndarray
    atc_threshold: TailThreshold
    cott_threshold: TailThreshold

    @classmethod
    def fit(
        cls,
        reference_predictions: pd.DataFrame,
        source_labels: pd.Series,
        classes: list[Any] | np.ndarray,
    ) -> "PerformanceEstimatorSuite":
        classes = np.asarray(classes)
        probabilities = probability_matrix(reference_predictions)
        if probabilities.shape[1] != len(classes):
            raise ValueError("Class labels do not match probability columns")
        if len(classes) < 2:
            raise ValueError("At least two classes are required")
        labels = np.asarray(source_labels)
        if len(labels) != len(reference_predictions):
            raise ValueError("Reference labels and predictions are not aligned")
        predicted = classes[np.argmax(probabilities, axis=1)]
        correct = predicted == labels
        source_accuracy = float(np.mean(correct))
        source_error = 1.0 - source_accuracy
        confidence = probabilities.max(axis=1)
        atc = fit_upper_tail_threshold(1.0 - confidence, source_error)

        class_mass = np.asarray(
            [np.mean(labels == class_label) for class_label in classes],
            dtype=np.float64,
        )
        if np.any(class_mass == 0):
            raise ValueError("COT requires every class in source calibration data")
        transport_probabilities = probabilities
        if (
            probabilities.shape[1] > 2
            and len(probabilities) > MULTICLASS_COTT_CALIBRATION_LIMIT
        ):
            # Class masses use the full calibration set.  Only the empirical
            # COTT score distribution is deterministically subsampled to keep
            # the exact multiclass EMD problem tractable and reproducible.
            positions = np.linspace(
                0,
                len(probabilities) - 1,
                MULTICLASS_COTT_CALIBRATION_LIMIT,
                dtype=int,
            )
            transport_probabilities = probabilities[positions]
        reference_transport = multiclass_transport_costs(
            transport_probabilities, class_mass
        )
        cott = fit_upper_tail_threshold(reference_transport, source_error)
        return cls(
            classes=classes,
            source_accuracy=source_accuracy,
            source_mean_confidence=float(np.mean(confidence)),
            source_class_mass=class_mass,
            atc_threshold=atc,
            cott_threshold=cott,
        )

    @property
    def source_error(self) -> float:
        return 1.0 - self.source_accuracy

    def calibration_record(self) -> dict[str, Any]:
        return {
            "endpoint": ENDPOINT,
            "source_accuracy": self.source_accuracy,
            "source_error": self.source_error,
            "source_mean_confidence": self.source_mean_confidence,
            "source_class_mass_json": json.dumps(self.source_class_mass.tolist()),
            **{f"atc_{key}": value for key, value in asdict(self.atc_threshold).items()},
            **{
                f"cott_{key}": value
                for key, value in asdict(self.cott_threshold).items()
            },
        }

    def estimate(self, target_predictions: pd.DataFrame) -> dict[str, float]:
        probabilities = probability_matrix(target_predictions)
        confidence = probabilities.max(axis=1)
        target_mean_confidence = float(np.mean(confidence))

        ac_accuracy = target_mean_confidence
        doc_accuracy = float(
            np.clip(
                self.source_accuracy
                + target_mean_confidence
                - self.source_mean_confidence,
                0.0,
                1.0,
            )
        )
        atc_error = self.atc_threshold.rate(1.0 - confidence)
        transport_costs = multiclass_transport_costs(
            probabilities, self.source_class_mass
        )
        cot_error = float(np.mean(transport_costs))
        cott_error = self.cott_threshold.rate(transport_costs)

        estimates = {
            "ac": 1.0 - ac_accuracy,
            "doc": 1.0 - doc_accuracy,
            "atc": atc_error,
            "cot": cot_error,
            "cott": cott_error,
        }
        for method, value in estimates.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"Invalid {method} error estimate: {value}")
        return estimates
