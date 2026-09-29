"""Probabilistic Adaptive Performance Estimation (PAPE).

This is an independent implementation of the algorithm described by Bialek
et al. (NeurIPS 2025).  It preserves the published monitor boundary: labeled
reference data configure a target-window-specific calibrator, while target
labels are never consumed by the estimator.

The implementation follows the paper/code-supplement defaults that affect the
estimand: a LightGBM domain classifier, density-ratio denominator floor 0.05,
weight floor 0.001, and a weighted LightGBM probability calibrator.  The
official supplement is CC BY-NC-SA and is not copied or vendored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from src.baselines.performance_estimators import probability_matrix


METHOD = "pape"
ENDPOINT = "classification_error"
RANDOM_STATE = 42
MINIMUM_DOMAIN_DENOMINATOR = 0.05
MINIMUM_DENSITY_WEIGHT = 0.001


def _lightgbm_classes() -> tuple[type, type]:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment validation
        raise ImportError(
            "PAPE requires lightgbm; install the project requirements"
        ) from exc
    return LGBMClassifier, LGBMRegressor


def _validate_feature_frames(
    reference_features: pd.DataFrame, target_features: pd.DataFrame
) -> None:
    if reference_features.empty or target_features.empty:
        raise ValueError("PAPE requires non-empty reference and target features")
    if reference_features.columns.tolist() != target_features.columns.tolist():
        raise ValueError("Reference and target feature schemas are not identical")
    if reference_features.columns.duplicated().any():
        raise ValueError("PAPE feature names must be unique")


def density_ratio_from_domain_probabilities(
    target_domain_probability_on_reference: np.ndarray,
    reference_size: int,
    target_size: int,
    minimum_denominator: float = MINIMUM_DOMAIN_DENOMINATOR,
    minimum_weight: float = MINIMUM_DENSITY_WEIGHT,
) -> np.ndarray:
    """Convert P(target-domain | x_ref) to target/reference density ratios."""

    probabilities = np.asarray(
        target_domain_probability_on_reference, dtype=np.float64
    )
    if probabilities.ndim != 1 or len(probabilities) != reference_size:
        raise ValueError("Domain probabilities must align with reference rows")
    if reference_size <= 0 or target_size <= 0:
        raise ValueError("Reference and target sizes must be positive")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Domain probabilities contain non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Domain probabilities fall outside [0, 1]")
    if not 0.0 < minimum_denominator <= 1.0:
        raise ValueError("minimum_denominator must fall in (0, 1]")
    if minimum_weight <= 0.0:
        raise ValueError("minimum_weight must be positive")

    denominator = np.maximum(minimum_denominator, 1.0 - probabilities)
    weights = (reference_size / target_size) * probabilities / denominator
    weights = np.maximum(weights, minimum_weight)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("PAPE produced invalid density-ratio weights")
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(np.square(weights)))
    if denominator <= 0.0:
        return 0.0
    return float(np.square(np.sum(weights)) / denominator)


def estimated_accuracy_from_calibrated_probabilities(
    positive_class_probabilities: np.ndarray,
    predicted_positive: np.ndarray,
) -> float:
    """Expected accuracy from PAPE's calibrated positive-class probabilities."""

    probabilities = np.asarray(positive_class_probabilities, dtype=np.float64)
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    if probabilities.ndim != 1 or probabilities.shape != predicted_positive.shape:
        raise ValueError("Calibrated probabilities and predictions must align")
    if len(probabilities) == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Calibrated probabilities must be finite and non-empty")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Calibrated probabilities fall outside [0, 1]")
    correctness_probability = np.where(
        predicted_positive, probabilities, 1.0 - probabilities
    )
    return float(np.mean(correctness_probability))


@dataclass(frozen=True)
class DensityRatioDiagnostics:
    reference_rows: int
    target_rows: int
    categorical_features: int
    continuous_features: int
    weight_min: float
    weight_median: float
    weight_mean: float
    weight_max: float
    weight_p99: float
    effective_sample_size: float
    effective_sample_size_fraction: float
    denominator_floor_fraction: float
    minimum_weight_fraction: float


@dataclass(frozen=True)
class DensityRatioResult:
    weights: np.ndarray
    diagnostics: DensityRatioDiagnostics


class PAPEDensityRatioEstimator:
    """Fit PAPE's target-vs-reference classifier for one target window."""

    def __init__(
        self,
        *,
        random_state: int = RANDOM_STATE,
        minimum_denominator: float = MINIMUM_DOMAIN_DENOMINATOR,
        minimum_weight: float = MINIMUM_DENSITY_WEIGHT,
        n_jobs: int | None = 1,
        n_estimators: int = 100,
    ) -> None:
        self.random_state = int(random_state)
        self.minimum_denominator = float(minimum_denominator)
        self.minimum_weight = float(minimum_weight)
        self.n_jobs = n_jobs
        self.n_estimators = int(n_estimators)

    def estimate_weights(
        self,
        reference_features: pd.DataFrame,
        target_features: pd.DataFrame,
    ) -> DensityRatioResult:
        _validate_feature_frames(reference_features, target_features)
        categorical = [
            column
            for column in reference_features.columns
            if not is_numeric_dtype(reference_features[column].dtype)
        ]
        continuous = [
            column for column in reference_features.columns if column not in categorical
        ]

        combined = pd.concat(
            [reference_features, target_features], ignore_index=True
        )
        if categorical:
            transformer = ColumnTransformer(
                [
                    (
                        "categorical",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                        categorical,
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
            transformed = transformer.fit_transform(combined)
            feature_names = transformer.get_feature_names_out().tolist()
        else:
            transformed = combined.to_numpy(dtype=np.float64)
            feature_names = combined.columns.tolist()
        transformed = pd.DataFrame(transformed, columns=feature_names)

        domain_labels = np.concatenate(
            [
                np.zeros(len(reference_features), dtype=np.int8),
                np.ones(len(target_features), dtype=np.int8),
            ]
        )
        classifier_type, _ = _lightgbm_classes()
        classifier = classifier_type(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            verbosity=-1,
        )
        fit_kwargs: dict[str, Any] = {}
        if categorical:
            fit_kwargs["categorical_feature"] = categorical
        classifier.fit(transformed, domain_labels, **fit_kwargs)
        reference_domain_probability = classifier.predict_proba(
            transformed.iloc[: len(reference_features)]
        )[:, 1]
        weights = density_ratio_from_domain_probabilities(
            reference_domain_probability,
            len(reference_features),
            len(target_features),
            self.minimum_denominator,
            self.minimum_weight,
        )
        denominator_floor = (1.0 - reference_domain_probability) < (
            self.minimum_denominator
        )
        minimum_weight_mask = weights <= self.minimum_weight * (1.0 + 1e-12)
        ess = effective_sample_size(weights)
        diagnostics = DensityRatioDiagnostics(
            reference_rows=len(reference_features),
            target_rows=len(target_features),
            categorical_features=len(categorical),
            continuous_features=len(continuous),
            weight_min=float(np.min(weights)),
            weight_median=float(np.median(weights)),
            weight_mean=float(np.mean(weights)),
            weight_max=float(np.max(weights)),
            weight_p99=float(np.quantile(weights, 0.99)),
            effective_sample_size=ess,
            effective_sample_size_fraction=ess / len(reference_features),
            denominator_floor_fraction=float(np.mean(denominator_floor)),
            minimum_weight_fraction=float(np.mean(minimum_weight_mask)),
        )
        return DensityRatioResult(weights=weights, diagnostics=diagnostics)

    def configuration(self) -> dict[str, Any]:
        return {
            "random_state": self.random_state,
            "minimum_denominator": self.minimum_denominator,
            "minimum_weight": self.minimum_weight,
            "n_jobs": self.n_jobs,
            "n_estimators": self.n_estimators,
        }


@dataclass(frozen=True)
class PAPEEstimate:
    estimated_error: float
    estimated_accuracy: float
    calibrated_probability_min: float
    calibrated_probability_mean: float
    calibrated_probability_max: float


class PAPEClassificationErrorEstimator:
    """PAPE estimator for the binary classification-error endpoint."""

    def __init__(
        self,
        reference_predictions: pd.DataFrame,
        source_labels: pd.Series,
        classes: list[Any] | np.ndarray,
        *,
        random_state: int = RANDOM_STATE,
        n_jobs: int | None = 1,
        n_estimators: int = 100,
    ) -> None:
        self.classes = np.asarray(classes)
        if self.classes.shape != (2,):
            raise ValueError("PAPE Phase 1 supports binary classifiers only")
        probabilities = probability_matrix(reference_predictions)
        labels = np.asarray(source_labels)
        if len(labels) != len(reference_predictions):
            raise ValueError("Reference predictions and labels are not aligned")
        self.reference_positive_probability = probabilities[:, 1]
        self.reference_positive_label = (labels == self.classes[1]).astype(float)
        reference_predicted = self.classes[np.argmax(probabilities, axis=1)]
        self.source_error = float(np.mean(reference_predicted != labels))
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.n_estimators = int(n_estimators)

    def estimate(
        self,
        target_predictions: pd.DataFrame,
        reference_density_weights: np.ndarray,
    ) -> PAPEEstimate:
        weights = np.asarray(reference_density_weights, dtype=np.float64)
        if weights.shape != self.reference_positive_probability.shape:
            raise ValueError("Density weights do not align with reference predictions")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("Density weights must be finite and positive")

        _, regressor_type = _lightgbm_classes()
        calibrator = regressor_type(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            verbosity=-1,
        )
        calibrator.fit(
            self.reference_positive_probability.reshape(-1, 1),
            self.reference_positive_label,
            sample_weight=weights,
        )
        target_probabilities = probability_matrix(target_predictions)
        calibrated = calibrator.predict(target_probabilities[:, 1].reshape(-1, 1))
        calibrated = np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)
        target_predicted = self.classes[np.argmax(target_probabilities, axis=1)]
        accuracy = estimated_accuracy_from_calibrated_probabilities(
            calibrated, target_predicted == self.classes[1]
        )
        error = 1.0 - accuracy
        if not np.isfinite(error) or not 0.0 <= error <= 1.0:
            raise RuntimeError(f"Invalid PAPE error estimate: {error}")
        return PAPEEstimate(
            estimated_error=error,
            estimated_accuracy=accuracy,
            calibrated_probability_min=float(np.min(calibrated)),
            calibrated_probability_mean=float(np.mean(calibrated)),
            calibrated_probability_max=float(np.max(calibrated)),
        )

    def calibration_record(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "endpoint": ENDPOINT,
            "positive_class": str(self.classes[1]),
            "reference_rows": len(self.reference_positive_probability),
            "source_error": self.source_error,
            "source_positive_rate": float(np.mean(self.reference_positive_label)),
            "source_mean_positive_probability": float(
                np.mean(self.reference_positive_probability)
            ),
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "n_estimators": self.n_estimators,
        }


def diagnostics_record(result: DensityRatioResult) -> dict[str, Any]:
    return asdict(result.diagnostics)
