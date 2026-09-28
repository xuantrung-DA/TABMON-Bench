"""DriftSHAP label-free monitoring baseline.

The monitor exposes both normalized feature rankings and raw attribution
magnitude. Alarm thresholds and proxy intervals are calibrated exclusively
from resampled reference features; target labels are never accepted by
``analyze_batch``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .abstract_monitor import AbstractMonitor
from .drift_importance import (
    IMPORTANCE_BACKENDS,
    compute_feature_discrepancies,
    compute_global_importance,
    weight_drift_discrepancies,
)


class DriftSHAPMonitor(AbstractMonitor):
    capabilities = {
        "risk_estimation": False,
        "attribution": True,
        "alarm": True,
        "alarm_target": "distribution_shift",
        "alarm_direction": "increase",
    }

    def __init__(
        self,
        reference_X: pd.DataFrame,
        reference_y: pd.Series,
        model: Any,
        *,
        batch_size: int = 1000,
        null_calibration_batches: int = 50,
        alarm_quantile: float = 0.99,
        random_seed: int = 17_021,
        importance_backend: str = "permutation",
    ) -> None:
        super().__init__(reference_X, reference_y, model)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if null_calibration_batches < 5:
            raise ValueError("null_calibration_batches must be at least 5")
        if not 0.5 < alarm_quantile < 1.0:
            raise ValueError("alarm_quantile must be between 0.5 and 1.0")
        if importance_backend not in IMPORTANCE_BACKENDS:
            raise ValueError(
                f"importance_backend must be one of {IMPORTANCE_BACKENDS}"
            )

        self.batch_size = batch_size
        self.null_calibration_batches = null_calibration_batches
        self.alarm_quantile = alarm_quantile
        self.random_seed = random_seed
        self.importance_backend = importance_backend
        self._compute_reference_importance()
        self._calibrate_null_distribution()

    def _compute_reference_importance(self) -> None:
        """Compute global model importance on the labeled reference set."""
        result = compute_global_importance(
            self.reference_X,
            self.reference_y,
            self.model,
            self.importance_backend,
        )
        self.global_shap = result.values
        self.shap_dict = result.as_dict()
        self.importance_backend_used = result.backend
        self.transformed_feature_count = result.transformed_feature_count

    def _score_batch(
        self, target_X: pd.DataFrame
    ) -> tuple[float, Dict[str, float], Dict[str, float]]:
        discrepancies = compute_feature_discrepancies(self.reference_X, target_X)
        return weight_drift_discrepancies(discrepancies, self.shap_dict)

    def _calibrate_null_distribution(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scores = []
        for _ in range(self.null_calibration_batches):
            positions = rng.integers(0, len(self.reference_X), size=self.batch_size)
            null_batch = self.reference_X.iloc[positions]
            score, _, _ = self._score_batch(null_batch)
            scores.append(score)
        self.null_scores = np.asarray(scores, dtype=float)
        self.null_score_median = float(np.median(self.null_scores))
        self.alarm_threshold = float(
            np.quantile(
                self.null_scores, self.alarm_quantile, method="higher"
            )
        )
        centered = np.abs(self.null_scores - self.null_score_median)
        self.proxy_ci_half_width = float(
            np.quantile(centered, self.alarm_quantile)
        )

    def analyze_batch(self, target_X: pd.DataFrame) -> Dict[str, Any]:
        score, raw_attribution, normalized_attribution = self._score_batch(target_X)
        denominator = max(self.null_score_median, np.finfo(float).eps)
        alarm_p_value = float(
            (1 + np.sum(self.null_scores >= score))
            / (len(self.null_scores) + 1)
        )
        alarm_alpha = 1.0 - self.alarm_quantile
        return {
            "capabilities": dict(self.capabilities),
            "monitor_score": score,
            "estimated_risk_change": None,
            "estimated_risk_ci_lower": None,
            "estimated_risk_ci_upper": None,
            "alarm": bool(alarm_p_value <= alarm_alpha),
            "alarm_p_value": alarm_p_value,
            "alarm_threshold": self.alarm_threshold,
            "alarm_direction": "increase",
            "feature_attribution": normalized_attribution,
            "raw_feature_attribution": raw_attribution,
            "attribution_abs_mass": score,
            "attribution_max_abs": float(
                max((abs(value) for value in raw_attribution.values()), default=0.0)
            ),
            "attribution_mass_relative_to_null": score / denominator,
            "null_score_median": self.null_score_median,
            "importance_backend": self.importance_backend_used,
            "transformed_feature_count": self.transformed_feature_count,
        }
