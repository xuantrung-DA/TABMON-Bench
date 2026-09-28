"""Calibrated confidence-based label-free risk estimator."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from .abstract_monitor import AbstractMonitor


class ConfidenceShiftMonitor(AbstractMonitor):
    """Estimate excess log loss from model-output statistics.

    A ridge calibration model is fitted on labeled reference predictions. At
    monitoring time it consumes only model probabilities. This makes the
    target-side estimator label-free while keeping its output on the same
    log-loss scale as the oracle.
    """

    capabilities = {
        "risk_estimation": True,
        "attribution": False,
        "alarm": True,
        "alarm_target": "risk_event",
        "alarm_direction": "increase",
    }

    def __init__(
        self,
        reference_X: pd.DataFrame,
        reference_y: pd.Series,
        model: Any,
        *,
        batch_size: int = 1000,
        null_calibration_batches: int = 100,
        alarm_quantile: float = 0.99,
        random_seed: int = 17_021,
    ) -> None:
        super().__init__(reference_X, reference_y, model)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if null_calibration_batches < 5:
            raise ValueError("null_calibration_batches must be at least 5")
        if not 0.5 < alarm_quantile < 1.0:
            raise ValueError("alarm_quantile must be between 0.5 and 1.0")

        self.batch_size = batch_size
        self.null_calibration_batches = null_calibration_batches
        self.alarm_quantile = alarm_quantile
        self.reference_probabilities = np.asarray(
            self.model.predict_proba(self.reference_X), dtype=float
        )
        self._fit_risk_calibrator(np.asarray(self.reference_y))
        self._calibrate_null_distribution(random_seed)

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

    def _observed_log_losses(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        classes = np.asarray(self.model.classes_)
        class_to_index = {label: index for index, label in enumerate(classes)}
        try:
            positions = np.asarray([class_to_index[label] for label in labels])
        except KeyError as exc:
            raise ValueError(f"Unknown reference class: {exc.args[0]}") from exc
        selected = probabilities[np.arange(len(labels)), positions]
        return -np.log(np.clip(selected, 1e-12, 1.0))

    def _fit_risk_calibrator(self, reference_y: np.ndarray) -> None:
        features = self._probability_features(self.reference_probabilities)
        observed_losses = self._observed_log_losses(
            self.reference_probabilities, reference_y
        )
        self.risk_calibrator = Ridge(alpha=1.0)
        self.risk_calibrator.fit(features, observed_losses)
        predicted = self.risk_calibrator.predict(features)
        self.reference_predicted_losses = np.asarray(predicted, dtype=float)
        self.reference_predicted_risk = float(
            np.mean(self.reference_predicted_losses)
        )
        residuals = observed_losses - self.reference_predicted_losses
        self.calibration_residual_variance = float(
            np.var(residuals, ddof=1) if len(residuals) > 1 else 0.0
        )

    def _predicted_losses(self, target_X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.model.predict_proba(target_X), dtype=float)
        return np.asarray(
            self.risk_calibrator.predict(self._probability_features(probabilities)),
            dtype=float,
        )

    def _calibrate_null_distribution(self, random_seed: int) -> None:
        rng = np.random.default_rng(random_seed)
        scores = []
        for _ in range(self.null_calibration_batches):
            positions = rng.integers(
                0, len(self.reference_predicted_losses), size=self.batch_size
            )
            score = float(
                np.mean(self.reference_predicted_losses[positions])
                - self.reference_predicted_risk
            )
            scores.append(score)
        self.null_scores = np.asarray(scores, dtype=float)
        self.null_score_median = float(np.median(self.null_scores))
        self.alarm_threshold = float(
            np.quantile(
                self.null_scores, self.alarm_quantile, method="higher"
            )
        )

    def analyze_batch(self, target_X: pd.DataFrame) -> Dict[str, Any]:
        predicted_losses = self._predicted_losses(target_X)
        estimated_delta = float(
            np.mean(predicted_losses) - self.reference_predicted_risk
        )
        target_variance = float(
            np.var(predicted_losses, ddof=1) if len(predicted_losses) > 1 else 0.0
        )
        reference_variance = float(
            np.var(self.reference_predicted_losses, ddof=1)
            if len(self.reference_predicted_losses) > 1
            else 0.0
        )
        standard_error = np.sqrt(
            (target_variance + self.calibration_residual_variance)
            / max(len(predicted_losses), 1)
            + reference_variance / max(len(self.reference_predicted_losses), 1)
        )
        z_value = float(norm.ppf((1.0 + self.alarm_quantile) / 2.0))
        half_width = z_value * standard_error
        alarm_p_value = float(
            (1 + np.sum(self.null_scores >= estimated_delta))
            / (len(self.null_scores) + 1)
        )
        alarm_alpha = 1.0 - self.alarm_quantile

        return {
            "capabilities": dict(self.capabilities),
            "monitor_score": estimated_delta,
            "estimated_risk_change": estimated_delta,
            "estimated_risk_ci_lower": estimated_delta - half_width,
            "estimated_risk_ci_upper": estimated_delta + half_width,
            "alarm": bool(alarm_p_value <= alarm_alpha),
            "alarm_p_value": alarm_p_value,
            "alarm_threshold": self.alarm_threshold,
            "alarm_direction": "increase",
            "feature_attribution": {},
            "raw_feature_attribution": {},
            "attribution_abs_mass": None,
            "attribution_max_abs": None,
            "attribution_mass_relative_to_null": None,
            "null_score_median": self.null_score_median,
            "risk_calibration_residual_std": float(
                np.sqrt(self.calibration_residual_variance)
            ),
        }
