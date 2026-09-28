"""Explicit global-importance backends for DriftSHAP-style monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


IMPORTANCE_BACKENDS = ("tree_shap", "permutation")


@dataclass(frozen=True)
class ImportanceResult:
    backend: str
    raw_features: tuple[str, ...]
    values: np.ndarray
    transformed_feature_count: int

    def as_dict(self) -> dict[str, float]:
        return {
            feature: float(value)
            for feature, value in zip(self.raw_features, self.values)
        }


class PreparedDriftReference:
    """Reusable exact KS/TV reference for many target batches.

    Numeric reference columns are sorted once.  This avoids repeatedly sorting
    a large calibration set while generating calibration-only null streams.
    """

    def __init__(self, reference_X: pd.DataFrame):
        self.reference_X = reference_X
        self.features = tuple(str(column) for column in reference_X.columns)
        self._numeric: dict[str, np.ndarray] = {}
        self._categorical: dict[str, pd.Series] = {}
        for feature in self.features:
            values = reference_X[feature]
            if pd.api.types.is_numeric_dtype(values):
                finite = values.dropna().to_numpy()
                self._numeric[feature] = np.sort(finite)
            else:
                self._categorical[feature] = values.value_counts(
                    normalize=True, dropna=False
                )

    @staticmethod
    def _ks_against_sorted_reference(
        sorted_reference: np.ndarray, target_values: pd.Series
    ) -> float:
        target = target_values.dropna().to_numpy()
        if not len(sorted_reference) or not len(target):
            return float("nan")
        values, counts = np.unique(target, return_counts=True)
        cumulative = np.cumsum(counts)
        target_right = cumulative / len(target)
        target_left = (cumulative - counts) / len(target)
        reference_right = (
            np.searchsorted(sorted_reference, values, side="right")
            / len(sorted_reference)
        )
        reference_left = (
            np.searchsorted(sorted_reference, values, side="left")
            / len(sorted_reference)
        )
        return float(
            max(
                np.max(np.abs(target_right - reference_right)),
                np.max(np.abs(target_left - reference_left)),
            )
        )

    def score_batch(self, target_X: pd.DataFrame) -> dict[str, float]:
        missing = set(self.features) - set(target_X.columns)
        if missing:
            raise ValueError(f"Target batch is missing features: {sorted(missing)}")
        result: dict[str, float] = {}
        for feature in self.features:
            if feature in self._numeric:
                statistic = self._ks_against_sorted_reference(
                    self._numeric[feature], target_X[feature]
                )
            else:
                target_frequency = target_X[feature].value_counts(
                    normalize=True, dropna=False
                )
                reference_frequency = self._categorical[feature]
                categories = reference_frequency.index.union(target_frequency.index)
                statistic = 0.5 * np.abs(
                    reference_frequency.reindex(categories, fill_value=0.0)
                    - target_frequency.reindex(categories, fill_value=0.0)
                ).sum()
            result[feature] = float(statistic)
        return result

    def score_position_batches(
        self, positions: np.ndarray
    ) -> np.ndarray:
        """Return rows=batch draws, columns=raw features."""

        positions = np.asarray(positions)
        if positions.ndim != 2:
            raise ValueError("positions must have shape (batches, batch_size)")
        matrix = np.empty((positions.shape[0], len(self.features)), dtype=float)
        for index, batch_positions in enumerate(positions):
            scores = self.score_batch(self.reference_X.iloc[batch_positions])
            matrix[index] = [scores[feature] for feature in self.features]
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite drift discrepancy in calibration batches")
        return matrix


def _unwrap_fitted_pipeline(model: Any) -> Pipeline:
    """Extract the fitted raw-feature pipeline from a calibrated estimator."""

    candidates: list[Any] = [model]
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        candidates.insert(0, calibrated[0].estimator)

    for candidate in candidates:
        current = candidate
        visited: set[int] = set()
        while id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, Pipeline):
                return current
            inner = getattr(current, "estimator", None)
            if inner is None or inner is current:
                break
            current = inner
    raise TypeError(
        "Could not extract a fitted preprocessing/model Pipeline from the "
        "calibrated estimator"
    )


def _positive_class_shap_values(values: Any, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == n_features:
        return array
    if array.ndim == 3:
        if array.shape[1] == n_features:
            return array[:, :, -1]
        if array.shape[2] == n_features:
            return array[-1, :, :]
    raise ValueError(f"Unsupported TreeSHAP value shape: {array.shape}")


def _raw_feature_for_transformed(
    transformed_name: str, raw_features: list[str]
) -> str:
    suffix = transformed_name.split("__", 1)[-1]
    candidates = [
        feature
        for feature in raw_features
        if suffix == feature or suffix.startswith(feature + "_")
    ]
    if not candidates:
        raise ValueError(
            f"Cannot map transformed feature '{transformed_name}' to a raw feature"
        )
    return max(candidates, key=len)


def _tree_shap_importance(
    reference_X: pd.DataFrame,
    model: Any,
    sample_size: int,
    random_seed: int,
) -> ImportanceResult:
    pipeline = _unwrap_fitted_pipeline(model)
    if len(pipeline.steps) < 2:
        raise ValueError("Expected preprocessing and estimator pipeline steps")
    preprocessor = pipeline[:-1]
    estimator = pipeline.steps[-1][1]
    estimator_name = estimator.__class__.__name__.lower()
    if "forest" not in estimator_name and "xgb" not in estimator_name:
        raise TypeError(
            "tree_shap backend is restricted to RF/XGB tree estimators; "
            f"found {estimator.__class__.__name__}"
        )

    sample = reference_X.sample(
        n=min(sample_size, len(reference_X)), random_state=random_seed
    )
    transformed = preprocessor.transform(sample)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    names = list(preprocessor.get_feature_names_out())
    if transformed.shape[1] != len(names):
        raise ValueError("Transformed feature names do not match matrix width")

    explainer = shap.TreeExplainer(estimator)
    values = _positive_class_shap_values(
        explainer.shap_values(transformed), transformed.shape[1]
    )
    transformed_importance = np.abs(values).mean(axis=0)
    raw_features = list(reference_X.columns)
    raw_importance = {feature: 0.0 for feature in raw_features}
    for name, value in zip(names, transformed_importance):
        raw = _raw_feature_for_transformed(name, raw_features)
        raw_importance[raw] += float(value)
    result = np.asarray([raw_importance[name] for name in raw_features], dtype=float)
    return ImportanceResult(
        backend="tree_shap",
        raw_features=tuple(raw_features),
        values=result,
        transformed_feature_count=len(names),
    )


def _permutation_importance(
    reference_X: pd.DataFrame,
    reference_y: pd.Series,
    model: Any,
    sample_size: int,
    random_seed: int,
    repeats: int,
    n_jobs: int,
) -> ImportanceResult:
    sample = reference_X.sample(
        n=min(sample_size, len(reference_X)), random_state=random_seed
    )
    sample_y = reference_y.loc[sample.index]
    result = permutation_importance(
        model,
        sample,
        sample_y,
        scoring="neg_log_loss",
        n_repeats=repeats,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    importance = np.abs(np.asarray(result.importances_mean, dtype=float))
    return ImportanceResult(
        backend="permutation",
        raw_features=tuple(reference_X.columns),
        values=importance,
        transformed_feature_count=len(reference_X.columns),
    )


def compute_global_importance(
    reference_X: pd.DataFrame,
    reference_y: pd.Series,
    model: Any,
    backend: str,
    *,
    sample_size: int = 1000,
    random_seed: int = 42,
    permutation_repeats: int = 3,
    n_jobs: int = -1,
) -> ImportanceResult:
    """Compute an explicit backend without silent fallback."""

    if backend not in IMPORTANCE_BACKENDS:
        raise ValueError(
            f"Unknown importance backend '{backend}'; expected {IMPORTANCE_BACKENDS}"
        )
    if sample_size <= 0 or permutation_repeats <= 0:
        raise ValueError("sample_size and permutation_repeats must be positive")
    if backend == "tree_shap":
        result = _tree_shap_importance(
            reference_X, model, sample_size, random_seed
        )
    else:
        result = _permutation_importance(
            reference_X,
            reference_y,
            model,
            sample_size,
            random_seed,
            permutation_repeats,
            n_jobs,
        )
    if len(result.values) != reference_X.shape[1]:
        raise ValueError("Importance dimension does not match raw features")
    if not np.all(np.isfinite(result.values)) or np.any(result.values < 0):
        raise ValueError("Global importance must be finite and non-negative")
    if not np.any(result.values > 0):
        raise ValueError("Global importance is identically zero")
    return result


def compute_feature_discrepancies(
    reference_X: pd.DataFrame,
    target_X: pd.DataFrame,
) -> dict[str, float]:
    """Measure raw-feature drift without consulting labels or model outputs."""

    missing = set(reference_X.columns) - set(target_X.columns)
    if missing:
        raise ValueError(f"Target batch is missing features: {sorted(missing)}")
    discrepancies: dict[str, float] = {}
    for feature in reference_X.columns:
        reference_values = reference_X[feature]
        target_values = target_X[feature]
        if pd.api.types.is_numeric_dtype(reference_values):
            statistic, _ = ks_2samp(
                reference_values.to_numpy(),
                target_values.to_numpy(),
                nan_policy="omit",
            )
        else:
            reference_frequency = reference_values.value_counts(
                normalize=True, dropna=False
            )
            target_frequency = target_values.value_counts(
                normalize=True, dropna=False
            )
            categories = reference_frequency.index.union(target_frequency.index)
            statistic = 0.5 * np.abs(
                reference_frequency.reindex(categories, fill_value=0.0)
                - target_frequency.reindex(categories, fill_value=0.0)
            ).sum()
        discrepancies[str(feature)] = float(statistic)
    return discrepancies


def weight_drift_discrepancies(
    discrepancies: dict[str, float],
    importance: ImportanceResult | dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Combine feature drift with an explicit global-importance backend."""

    importance_by_feature = (
        importance.as_dict() if isinstance(importance, ImportanceResult) else importance
    )
    if set(discrepancies) != set(importance_by_feature):
        raise ValueError("Discrepancy and importance feature sets do not match")
    raw = {
        feature: float(discrepancies[feature] * importance_by_feature[feature])
        for feature in discrepancies
    }
    absolute_mass = float(sum(abs(value) for value in raw.values()))
    normalized = (
        {feature: value / absolute_mass for feature, value in raw.items()}
        if absolute_mass > 0.0
        else {feature: 0.0 for feature in raw}
    )
    return absolute_mass, raw, normalized
