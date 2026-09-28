r"""Paper-faithful Sequential Harmful Shift Detection (SHD).

This module implements the quantile detector proposed by Amoukou et al.
(NeurIPS 2024), including the higher-power :math:`\Phi_q^2` statistic used in
their main experiments. An RF regressor learns sample error from source
features, an independent source split calibrates true/predicted-error
quantiles under an FDP constraint, and deployment uses no target labels.

The target lower confidence sequence is the predictably-mixed empirical-
Bernstein (PM-EB) construction used by Podkopaev and Ramdas (ICLR 2022).
Fixed-source terms use the Hoeffding interval specified in the SHD appendix.
The implementation fails closed when no selector satisfies the predeclared
FDP limit; relaxing that limit is a separate sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.baselines.performance_estimators import probability_matrix


METHOD = "shd_quantile_phi2_pm_eb"
ENDPOINT = "sequential_selected_high_error_prevalence"


def hoeffding_width(sample_size: int, alpha: float) -> float:
    """Two-sided Hoeffding half-width used for fixed source quantities."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * sample_size)))


def pm_eb_lower_bound(
    observations: np.ndarray | list[float], alpha: float, *, cap: float = 0.5
) -> np.ndarray:
    """One-sided PM-EB lower confidence sequence for bounded observations.

    This is the closed-form predictable plug-in construction reported in
    Appendix E of Podkopaev and Ramdas (2022). The regularized running mean
    starts at 1/2 and the variance estimate at 1/4, with predictable betting
    rates capped at 1/2, matching their experimental specification.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("PM-EB observations must be finite and lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie in (0, 1)")

    log_term = np.log(1.0 / alpha)
    weighted_sum = 0.0
    weight_sum = 0.0
    penalty_sum = 0.0
    cumulative = 0.0
    variance_numerator = 0.25
    mean_previous = 0.5
    variance_previous = 0.25
    lower = np.empty(len(values), dtype=np.float64)

    for index, value in enumerate(values, start=1):
        denominator = variance_previous * index * np.log1p(index)
        betting_rate = min(np.sqrt(2.0 * log_term / denominator), cap)
        variance_increment = 4.0 * (value - mean_previous) ** 2
        psi = (-np.log1p(-betting_rate) - betting_rate) / 4.0
        weighted_sum += betting_rate * value
        weight_sum += betting_rate
        penalty_sum += variance_increment * psi
        lower[index - 1] = (
            weighted_sum - log_term - penalty_sum
        ) / weight_sum

        cumulative += value
        mean_current = (0.5 + cumulative) / (index + 1.0)
        variance_numerator += (value - mean_current) ** 2
        variance_previous = variance_numerator / (index + 1.0)
        mean_previous = mean_current

    return np.clip(lower, 0.0, 1.0)


def _coerce_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize extension dtypes without learning from target data."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce")
        else:
            result[column] = result[column].astype(object)
            result[column] = result[column].where(result[column].notna(), np.nan)
    return result


def _error_regression_pipeline(frame: pd.DataFrame, seed: int, trees: int) -> Pipeline:
    numeric = [
        column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [column for column in frame.columns if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
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
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    regressor = RandomForestRegressor(
        n_estimators=trees,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


@dataclass(frozen=True)
class SHDCalibration:
    true_error_quantile_probability: float
    predicted_error_quantile_probability: float
    true_error_threshold: float
    predicted_error_threshold: float
    selector_fdp: float
    selector_power: float
    selector_feasible: bool
    feasible_candidate_count: int
    evaluated_candidate_count: int
    minimum_grid_fdp: float
    power_at_minimum_grid_fdp: float
    error_estimator_r2: float
    source_high_error_rate: float
    source_high_error_upper: float
    source_selected_high_error_rate: float
    source_selected_high_error_upper: float
    source_false_discovery_joint_rate: float
    source_false_discovery_joint_upper: float
    source_selected_rate: float
    source_size: int
    error_estimator_type: str
    error_estimator_trees: int
    fdp_limit: float
    alpha_total: float
    alpha_source: float
    alpha_target: float
    alpha_false_discovery: float
    epsilon: float


class SHDMonitor:
    """SHD quantile monitor whose deployment API accepts features only."""

    def __init__(self, estimator: Any, calibration: SHDCalibration):
        self.estimator = estimator
        self.calibration = calibration

    @classmethod
    def fit(
        cls,
        reference_X: pd.DataFrame,
        reference_predictions: pd.DataFrame,
        source_labels: pd.Series,
        classes: list[Any] | np.ndarray,
        *,
        random_seed: int = 42,
        fdp_limit: float = 0.20,
        alpha: float = 0.01,
        epsilon: float = 0.02,
        n_estimators: int = 200,
    ) -> "SHDMonitor":
        if not 0.0 <= fdp_limit < 1.0:
            raise ValueError("fdp_limit must lie in [0, 1)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        features = _coerce_features(reference_X.reset_index(drop=True))
        probabilities = probability_matrix(reference_predictions)
        labels = np.asarray(source_labels)
        if not (len(features) == len(labels) == len(probabilities)):
            raise ValueError("Reference features, predictions, and labels are not aligned")
        classes = np.asarray(classes)
        predicted = classes[np.argmax(probabilities, axis=1)]
        errors = (predicted != labels).astype(np.float64)

        rng = np.random.default_rng(random_seed)
        order = rng.permutation(len(features))
        split = len(order) // 2
        train_idx, calibration_idx = order[:split], order[split:]
        if len(train_idx) < 20 or len(calibration_idx) < 20:
            raise ValueError("SHD requires at least 40 source calibration rows")

        estimator = _error_regression_pipeline(
            features.iloc[train_idx], random_seed, n_estimators
        )
        estimator.fit(features.iloc[train_idx], errors[train_idx])
        predicted_error = np.clip(
            np.asarray(estimator.predict(features.iloc[calibration_idx]), dtype=float),
            0.0,
            1.0,
        )
        actual_error = errors[calibration_idx]
        estimator_r2 = float(r2_score(actual_error, predicted_error))

        candidates: list[tuple[float, float, float, float, float, float]] = []
        evaluated: list[tuple[float, float]] = []
        for true_p in np.arange(0.50, 0.951, 0.05):
            true_threshold = float(np.quantile(actual_error, true_p))
            high_error = actual_error > true_threshold
            if not high_error.any():
                continue
            for predicted_p in np.arange(0.10, 0.901, 0.10):
                predicted_threshold = float(np.quantile(predicted_error, predicted_p))
                selected = predicted_error > predicted_threshold
                if not selected.any():
                    continue
                false_discovery = float(np.mean(~high_error[selected]))
                power = float(np.mean(selected[high_error]))
                evaluated.append((false_discovery, power))
                if false_discovery < fdp_limit:
                    candidates.append(
                        (
                            power,
                            -false_discovery,
                            true_p,
                            predicted_p,
                            true_threshold,
                            predicted_threshold,
                        )
                    )

        feasible = bool(candidates)
        if feasible:
            power, negative_fdp, true_p, predicted_p, true_q, predicted_q = max(
                candidates
            )
            fdp = -negative_fdp
        else:
            true_p = 0.50
            true_q = float(np.quantile(actual_error, true_p))
            predicted_p = 0.90
            predicted_q = float("inf")
            fdp = float("nan")
            power = 0.0

        if evaluated:
            minimum_fdp, power_at_minimum_fdp = min(
                evaluated, key=lambda pair: (pair[0], -pair[1])
            )
        else:
            minimum_fdp, power_at_minimum_fdp = float("nan"), 0.0

        selected = predicted_error > predicted_q
        high_error = actual_error > true_q
        selected_high = selected & high_error
        false_discovery_joint = selected & ~high_error
        n_source = len(calibration_idx)
        alpha_source = alpha / 2.0
        alpha_target = alpha / 4.0
        alpha_false_discovery = alpha / 4.0
        source_width = hoeffding_width(n_source, alpha_source)
        false_discovery_width = hoeffding_width(n_source, alpha_false_discovery)
        source_high_rate = float(np.mean(high_error))
        source_selected_high_rate = float(np.mean(selected_high))
        false_discovery_rate = float(np.mean(false_discovery_joint))

        calibration = SHDCalibration(
            true_error_quantile_probability=float(true_p),
            predicted_error_quantile_probability=float(predicted_p),
            true_error_threshold=float(true_q),
            predicted_error_threshold=float(predicted_q),
            selector_fdp=float(fdp),
            selector_power=float(power),
            selector_feasible=feasible,
            feasible_candidate_count=len(candidates),
            evaluated_candidate_count=len(evaluated),
            minimum_grid_fdp=float(minimum_fdp),
            power_at_minimum_grid_fdp=float(power_at_minimum_fdp),
            error_estimator_r2=estimator_r2,
            source_high_error_rate=source_high_rate,
            source_high_error_upper=min(1.0, source_high_rate + source_width),
            source_selected_high_error_rate=source_selected_high_rate,
            source_selected_high_error_upper=min(
                1.0, source_selected_high_rate + source_width
            ),
            source_false_discovery_joint_rate=false_discovery_rate,
            source_false_discovery_joint_upper=min(
                1.0, false_discovery_rate + false_discovery_width
            ),
            source_selected_rate=float(np.mean(selected)),
            source_size=n_source,
            error_estimator_type="RandomForestRegressor",
            error_estimator_trees=int(n_estimators),
            fdp_limit=float(fdp_limit),
            alpha_total=float(alpha),
            alpha_source=float(alpha_source),
            alpha_target=float(alpha_target),
            alpha_false_discovery=float(alpha_false_discovery),
            epsilon=float(epsilon),
        )
        return cls(estimator, calibration)

    def predicted_error(self, target_X: pd.DataFrame) -> np.ndarray:
        features = _coerce_features(target_X.reset_index(drop=True))
        return np.clip(
            np.asarray(self.estimator.predict(features), dtype=float), 0.0, 1.0
        )

    def with_inference_parameters(
        self, *, alpha: float, epsilon: float
    ) -> "SHDMonitor":
        """Reuse the fitted source error model under a new alarm boundary.

        The SHD quantile selector and error regressor do not depend on
        ``alpha`` or ``epsilon``.  These parameters enter only through the
        source Hoeffding bounds, the target PM-EB confidence sequence, and the
        final harmful-change margin.  Reconfiguring those terms therefore
        gives an exact cached sensitivity analysis without repeatedly fitting
        the same random forest or changing the selected source subgroup.
        """

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        calibration = self.calibration
        alpha_source = alpha / 2.0
        alpha_target = alpha / 4.0
        alpha_false_discovery = alpha / 4.0
        source_width = hoeffding_width(calibration.source_size, alpha_source)
        false_discovery_width = hoeffding_width(
            calibration.source_size, alpha_false_discovery
        )
        updated = replace(
            calibration,
            source_high_error_upper=min(
                1.0, calibration.source_high_error_rate + source_width
            ),
            source_selected_high_error_upper=min(
                1.0, calibration.source_selected_high_error_rate + source_width
            ),
            source_false_discovery_joint_upper=min(
                1.0,
                calibration.source_false_discovery_joint_rate
                + false_discovery_width,
            ),
            alpha_total=float(alpha),
            alpha_source=float(alpha_source),
            alpha_target=float(alpha_target),
            alpha_false_discovery=float(alpha_false_discovery),
            epsilon=float(epsilon),
        )
        return SHDMonitor(self.estimator, updated)

    def monitor(self, target_X: pd.DataFrame) -> pd.DataFrame:
        """Return SHD Eq. 16/20/24 statistics and absorbing alarms."""
        predicted_error = self.predicted_error(target_X)
        selected = predicted_error > self.calibration.predicted_error_threshold
        selected_float = selected.astype(np.float64)
        running_rate = np.cumsum(selected_float) / np.arange(
            1, len(selected) + 1, dtype=np.float64
        )
        target_selected_lower = pm_eb_lower_bound(
            selected_float, self.calibration.alpha_target
        )
        high_error_lower = np.maximum(
            0.0,
            target_selected_lower
            - self.calibration.source_false_discovery_joint_upper,
        )
        phi_q_boundary = min(
            1.0,
            self.calibration.source_high_error_upper + self.calibration.epsilon,
        )
        phi_q2_boundary = min(
            1.0,
            self.calibration.source_selected_high_error_upper
            + self.calibration.epsilon,
        )
        phi_q_alarm = np.maximum.accumulate(high_error_lower > phi_q_boundary)
        phi_q2_alarm = np.maximum.accumulate(high_error_lower > phi_q2_boundary)
        return pd.DataFrame(
            {
                "predicted_error": predicted_error,
                "high_error_selector": selected,
                "running_selected_rate": running_rate,
                "target_selected_pm_eb_lower": target_selected_lower,
                "estimated_high_error_lower": high_error_lower,
                "phi_q_boundary": phi_q_boundary,
                "phi_q2_boundary": phi_q2_boundary,
                "alarm_phi_q": phi_q_alarm.astype(bool),
                "alarm_phi_q2": phi_q2_alarm.astype(bool),
                "alarm": phi_q2_alarm.astype(bool),
            }
        )

    def calibration_record(self) -> dict[str, Any]:
        return {"method": METHOD, "endpoint": ENDPOINT, **asdict(self.calibration)}
