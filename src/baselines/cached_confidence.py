"""Cache-native form of the calibrated Confidence log-loss estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in frame.columns if column.startswith("probability_class_")
    ]
    if not columns:
        raise ValueError("Prediction cache contains no probability columns")
    return sorted(columns, key=lambda value: int(value.rsplit("_", 1)[-1]))


def probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[probability_columns(frame)].to_numpy(dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid cached probability matrix")
    return values


@dataclass
class CachedConfidenceEstimator:
    classes: np.ndarray
    calibrator: Ridge
    reference_predicted_losses: np.ndarray
    reference_predicted_risk: float
    reference_observed_risk: float
    calibration_residual_variance: float

    @staticmethod
    def _probability_features(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-12, 1.0)
        confidence = np.max(clipped, axis=1)
        entropy = -np.sum(clipped * np.log(clipped), axis=1)
        ordered = np.sort(clipped, axis=1)
        margin = (
            ordered[:, -1] - ordered[:, -2]
            if ordered.shape[1] > 1
            else confidence
        )
        self_information = -np.log(confidence)
        return np.column_stack(
            [clipped, confidence, entropy, margin, self_information]
        )

    @classmethod
    def fit(
        cls,
        reference_predictions: pd.DataFrame,
        reference_y: pd.Series,
        classes: list[Any] | np.ndarray,
    ) -> "CachedConfidenceEstimator":
        probabilities = probability_matrix(reference_predictions)
        class_values = np.asarray(classes)
        if probabilities.shape[1] != len(class_values):
            raise ValueError("Cached probabilities do not match declared classes")
        if len(reference_y) != len(probabilities):
            raise ValueError("Reference predictions and labels have unequal length")
        class_to_index = {
            value: index for index, value in enumerate(class_values.tolist())
        }
        try:
            positions = np.asarray(
                [class_to_index[value] for value in reference_y.tolist()]
            )
        except KeyError as exc:
            raise ValueError(f"Unknown reference class: {exc.args[0]}") from exc
        selected = probabilities[np.arange(len(probabilities)), positions]
        observed_losses = -np.log(np.clip(selected, 1e-12, 1.0))
        features = cls._probability_features(probabilities)
        calibrator = Ridge(alpha=1.0).fit(features, observed_losses)
        predicted_losses = np.asarray(calibrator.predict(features), dtype=float)
        residuals = observed_losses - predicted_losses
        return cls(
            classes=class_values,
            calibrator=calibrator,
            reference_predicted_losses=predicted_losses,
            reference_predicted_risk=float(predicted_losses.mean()),
            reference_observed_risk=float(observed_losses.mean()),
            calibration_residual_variance=float(
                np.var(residuals, ddof=1) if len(residuals) > 1 else 0.0
            ),
        )

    def predicted_losses(self, predictions: pd.DataFrame) -> np.ndarray:
        probabilities = probability_matrix(predictions)
        if probabilities.shape[1] != len(self.classes):
            raise ValueError("Target probabilities do not match declared classes")
        return np.asarray(
            self.calibrator.predict(self._probability_features(probabilities)),
            dtype=float,
        )

    def estimate_excess_log_loss(self, predictions: pd.DataFrame) -> float:
        return float(self.predicted_losses(predictions).mean() - self.reference_predicted_risk)

    def score_position_batches(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions)
        if positions.ndim != 2:
            raise ValueError("positions must have shape (batches, batch_size)")
        return (
            self.reference_predicted_losses[positions].mean(axis=1)
            - self.reference_predicted_risk
        )
