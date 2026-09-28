"""Tests for explicit DriftSHAP global-importance backends."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.baselines.drift_importance import (
    PreparedDriftReference,
    compute_feature_discrepancies,
    compute_global_importance,
    weight_drift_discrepancies,
)


def fitted_model(estimator):
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        {
            "numeric_a": rng.normal(size=180),
            "numeric_b": rng.normal(size=180),
            "category": np.where(rng.random(180) > 0.5, "left", "right"),
        }
    )
    target = pd.Series(
        (
            frame["numeric_a"]
            + 0.6 * frame["numeric_b"]
            + (frame["category"] == "left").astype(float)
            + rng.normal(scale=0.4, size=len(frame))
            > 0.5
        ).astype(int),
        index=frame.index,
    )
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["numeric_a", "numeric_b"],
            ),
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
                ["category"],
            ),
        ],
        sparse_threshold=1.0,
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("estimator", estimator)]
    )
    pipeline.fit(frame.iloc[:120], target.iloc[:120])
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="isotonic")
    calibrated.fit(frame.iloc[120:], target.iloc[120:])
    return frame.iloc[120:].copy(), target.iloc[120:].copy(), calibrated


class DriftImportanceBackendTest(unittest.TestCase):
    def test_tree_and_permutation_are_explicit_raw_feature_backends(self):
        frame, target, model = fitted_model(
            RandomForestClassifier(n_estimators=20, max_depth=4, random_state=7)
        )
        results = {
            backend: compute_global_importance(
                frame,
                target,
                model,
                backend,
                sample_size=50,
                permutation_repeats=2,
                n_jobs=1,
            )
            for backend in ("tree_shap", "permutation")
        }
        for backend, result in results.items():
            self.assertEqual(result.backend, backend)
            self.assertEqual(result.raw_features, tuple(frame.columns))
            self.assertEqual(len(result.values), frame.shape[1])
            self.assertTrue(np.isfinite(result.values).all())
            self.assertGreater(float(result.values.sum()), 0.0)
        self.assertGreater(results["tree_shap"].transformed_feature_count, 3)
        self.assertEqual(results["permutation"].transformed_feature_count, 3)

    def test_tree_backend_does_not_silently_fall_back(self):
        frame, target, model = fitted_model(LogisticRegression(max_iter=500))
        with self.assertRaisesRegex(TypeError, "restricted to RF/XGB"):
            compute_global_importance(
                frame, target, model, "tree_shap", sample_size=40, n_jobs=1
            )

    def test_discrepancy_weighting_preserves_raw_feature_schema(self):
        reference = pd.DataFrame(
            {"number": [0.0, 0.0, 1.0, 1.0], "category": ["a", "a", "b", "b"]}
        )
        target = pd.DataFrame(
            {"number": [2.0, 2.0, 3.0, 3.0], "category": ["b", "b", "b", "b"]}
        )
        discrepancies = compute_feature_discrepancies(reference, target)
        mass, raw, normalized = weight_drift_discrepancies(
            discrepancies, {"number": 2.0, "category": 1.0}
        )
        self.assertEqual(set(raw), {"number", "category"})
        self.assertGreater(mass, 0.0)
        self.assertAlmostEqual(sum(abs(value) for value in normalized.values()), 1.0)

    def test_prepared_reference_matches_standard_discrepancies(self):
        reference = pd.DataFrame(
            {
                "number": [0.0, 0.0, 1.0, 2.0, np.nan],
                "category": ["a", "a", "b", None, "b"],
            }
        )
        target = pd.DataFrame(
            {
                "number": [0.0, 1.0, 1.0, 3.0, np.nan],
                "category": ["a", "b", "b", None, "c"],
            }
        )
        expected = compute_feature_discrepancies(reference, target)
        actual = PreparedDriftReference(reference).score_batch(target)
        for feature in expected:
            self.assertAlmostEqual(actual[feature], expected[feature])


if __name__ == "__main__":
    unittest.main()
