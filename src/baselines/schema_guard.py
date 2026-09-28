"""Label-free schema and support guard for cached tabular streams.

The guard is fit exclusively from source-calibration features.  It reports
observable data-quality evidence; it neither consumes labels nor attempts to
correct a model-risk estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd


MISSING_TOKEN = "__TABMON_MISSING__"


def _categorical_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_TOKEN)


@dataclass(frozen=True)
class NumericFeatureProfile:
    missing_rate: float
    hard_min: float
    hard_max: float
    lower_tail: float
    upper_tail: float
    reference_tail_rate: float


@dataclass(frozen=True)
class CategoricalFeatureProfile:
    missing_rate: float
    levels: frozenset[str]


@dataclass(frozen=True)
class SchemaGuardScore:
    score: float
    top_feature: str
    top_component: str
    feature_scores_json: str
    component_scores_json: str


class SchemaGuardProfile:
    """Reference-only profile for observable schema/support violations."""

    def __init__(
        self,
        columns: tuple[str, ...],
        numeric: dict[str, NumericFeatureProfile],
        categorical: dict[str, CategoricalFeatureProfile],
    ) -> None:
        self.columns = columns
        self.numeric = numeric
        self.categorical = categorical

    @classmethod
    def fit(
        cls,
        reference_X: pd.DataFrame,
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
    ) -> "SchemaGuardProfile":
        if reference_X.empty:
            raise ValueError("Reference features must not be empty")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("Invalid robust-tail quantiles")
        numeric: dict[str, NumericFeatureProfile] = {}
        categorical: dict[str, CategoricalFeatureProfile] = {}
        for column in reference_X.columns:
            values = reference_X[column]
            missing_rate = float(values.isna().mean())
            if pd.api.types.is_numeric_dtype(values.dtype):
                finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                finite = finite[np.isfinite(finite)]
                if not len(finite):
                    # An all-missing numeric column is handled by missingness only.
                    numeric[column] = NumericFeatureProfile(
                        missing_rate, np.nan, np.nan, np.nan, np.nan, 0.0
                    )
                    continue
                lower, upper = np.quantile(finite, [lower_quantile, upper_quantile])
                reference_tail_rate = float(
                    np.mean((finite < lower) | (finite > upper))
                )
                numeric[column] = NumericFeatureProfile(
                    missing_rate=missing_rate,
                    hard_min=float(np.min(finite)),
                    hard_max=float(np.max(finite)),
                    lower_tail=float(lower),
                    upper_tail=float(upper),
                    reference_tail_rate=reference_tail_rate,
                )
            else:
                levels = frozenset(_categorical_values(values).astype(str).unique())
                categorical[column] = CategoricalFeatureProfile(
                    missing_rate=missing_rate,
                    levels=levels,
                )
        return cls(tuple(reference_X.columns), numeric, categorical)

    def score_batch(self, batch_X: pd.DataFrame) -> SchemaGuardScore:
        if tuple(batch_X.columns) != self.columns:
            missing = sorted(set(self.columns) - set(batch_X.columns))
            extra = sorted(set(batch_X.columns) - set(self.columns))
            raise ValueError(
                f"Feature schema mismatch; missing={missing}, extra={extra}, "
                "or column order changed"
            )
        if batch_X.empty:
            raise ValueError("Target batch must not be empty")
        per_feature: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for column in self.columns:
            values = batch_X[column]
            missing_rate = float(values.isna().mean())
            if column in self.numeric:
                profile = self.numeric[column]
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                finite_mask = np.isfinite(numeric)
                finite = numeric[finite_mask]
                if len(finite) and np.isfinite(profile.hard_min):
                    hard_rate = float(
                        np.mean((finite < profile.hard_min) | (finite > profile.hard_max))
                    )
                    tail_rate = float(
                        np.mean((finite < profile.lower_tail) | (finite > profile.upper_tail))
                    )
                else:
                    hard_rate = 0.0
                    tail_rate = 0.0
                column_components = {
                    "missing_rate_increase": max(
                        0.0, missing_rate - profile.missing_rate
                    ),
                    "hard_range_violation_rate": hard_rate,
                    "tail_rate_increase": max(
                        0.0, tail_rate - profile.reference_tail_rate
                    ),
                    "novel_category_rate": 0.0,
                }
            else:
                profile = self.categorical[column]
                categorical = _categorical_values(values).astype(str)
                novel_rate = float((~categorical.isin(profile.levels)).mean())
                column_components = {
                    "missing_rate_increase": max(
                        0.0, missing_rate - profile.missing_rate
                    ),
                    "hard_range_violation_rate": 0.0,
                    "tail_rate_increase": 0.0,
                    "novel_category_rate": novel_rate,
                }
            components[column] = column_components
            per_feature[column] = float(max(column_components.values()))

        top_feature = max(self.columns, key=lambda name: per_feature[name])
        top_component = max(
            components[top_feature], key=lambda name: components[top_feature][name]
        )
        return SchemaGuardScore(
            score=per_feature[top_feature],
            top_feature=top_feature,
            top_component=top_component,
            feature_scores_json=json.dumps(per_feature, sort_keys=True),
            component_scores_json=json.dumps(components, sort_keys=True),
        )

    def score_position_batches(
        self, reference_X: pd.DataFrame, positions: np.ndarray
    ) -> np.ndarray:
        positions = np.asarray(positions)
        if positions.ndim != 2:
            raise ValueError("positions must have shape (batches, batch_size)")
        if positions.size and (positions.min() < 0 or positions.max() >= len(reference_X)):
            raise IndexError("Reference resampling position is out of bounds")
        return np.asarray(
            [self.score_batch(reference_X.iloc[row]).score for row in positions],
            dtype=float,
        )

