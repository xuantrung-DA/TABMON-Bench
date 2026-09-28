import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.build_stream_cache import main as build_stream_cache
from experiments.run_phase5_performance_estimators import main as run_phase5
from src.baselines.performance_estimators import (
    METHODS,
    PerformanceEstimatorSuite,
    binary_transport_costs,
    fit_upper_tail_threshold,
    probability_matrix,
)


class Phase5EstimatorTest(unittest.TestCase):
    def test_target_prediction_allowlist_rejects_oracle_columns(self):
        frame = pd.DataFrame(
            {
                "probability_class_0": [0.6],
                "probability_class_1": [0.4],
                "predicted_class": [0],
                "target_label": [0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Non-allowlisted"):
            probability_matrix(frame)

    def test_fractional_tie_threshold_matches_requested_rate(self):
        scores = np.asarray([0.1, 0.2, 0.2, 0.9])
        threshold = fit_upper_tail_threshold(scores, 0.5)
        self.assertAlmostEqual(threshold.rate(scores), 0.5)
        self.assertAlmostEqual(threshold.threshold, 0.2)
        self.assertAlmostEqual(threshold.tie_weight, 0.5)

    def test_binary_transport_obeys_source_class_mass(self):
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]
        )
        balanced = binary_transport_costs(probabilities, np.asarray([0.5, 0.5]))
        all_class_zero = binary_transport_costs(
            probabilities, np.asarray([1.0, 0.0])
        )
        self.assertAlmostEqual(float(np.mean(balanced)), 0.2)
        self.assertAlmostEqual(float(np.mean(all_class_zero)), 0.45)

    def test_all_estimators_return_classification_error(self):
        reference = pd.DataFrame(
            {
                "probability_class_0": [0.9, 0.8, 0.4, 0.1],
                "probability_class_1": [0.1, 0.2, 0.6, 0.9],
                "predicted_class": [0, 0, 1, 1],
            }
        )
        labels = pd.Series([0, 0, 1, 0])
        suite = PerformanceEstimatorSuite.fit(reference, labels, [0, 1])
        estimates = suite.estimate(reference)

        self.assertEqual(set(estimates), set(METHODS))
        self.assertAlmostEqual(suite.source_error, 0.25)
        self.assertAlmostEqual(estimates["doc"], suite.source_error)
        self.assertAlmostEqual(estimates["atc"], suite.source_error)
        self.assertAlmostEqual(estimates["cott"], suite.source_error)
        for value in estimates.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_cached_runner_produces_complete_paired_grid(self):
        rng = np.random.default_rng(19)
        frame = pd.DataFrame(
            {
                "age": rng.normal(40, 10, 180),
                "education-num": rng.integers(1, 16, 180),
            }
        )
        frame["income"] = (
            frame["age"] + 1.5 * frame["education-num"] > 55
        ).astype(int)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / "data" / "adult"
            model_dir = root / "models" / "models" / "adult"
            cache_dir = root / "cache"
            output_dir = root / "phase5"
            data_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            calibration = frame.iloc[:80].reset_index(drop=True)
            test_pool = frame.iloc[80:].reset_index(drop=True)
            calibration.to_parquet(data_dir / "calibration.parquet", index=False)
            test_pool.to_parquet(data_dir / "test_pool.parquet", index=False)
            model = LogisticRegression(max_iter=500).fit(
                calibration[["age", "education-num"]], calibration["income"]
            )
            joblib.dump(model, model_dir / "lr_calibrated.pkl")

            cache_exit = build_stream_cache(
                [
                    "--datasets",
                    "adult",
                    "--models",
                    "lr",
                    "--shifts",
                    "covariate",
                    "--severities",
                    "high",
                    "--modes",
                    "abrupt",
                    "--seeds",
                    "42",
                    "--num-batches",
                    "2",
                    "--batch-size",
                    "16",
                    "--data-dir",
                    str(root / "data"),
                    "--base-models-dir",
                    str(root / "models"),
                    "--cache-dir",
                    str(cache_dir),
                ]
            )
            self.assertEqual(cache_exit, 0)
            status_path = cache_dir / "cache_status.json"
            status = json.loads(status_path.read_text())
            status["cache_revision"] = 1
            status_path.write_text(json.dumps(status), encoding="utf-8")

            phase5_exit = run_phase5(
                [
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(output_dir),
                    "--limit-streams",
                    "1",
                ]
            )
            self.assertEqual(phase5_exit, 0)
            batches = pd.read_parquet(output_dir / "batch_metrics.parquet")
            scenarios = pd.read_parquet(output_dir / "scenario_metrics.parquet")
            phase5_status = json.loads(
                (output_dir / "phase5_status.json").read_text()
            )

            self.assertEqual(len(batches), 2 * len(METHODS))
            self.assertEqual(len(scenarios), len(METHODS))
            self.assertEqual(set(scenarios["method"]), set(METHODS))
            self.assertEqual(set(scenarios["endpoint"]), {"classification_error"})
            self.assertFalse(any("log_loss" in column for column in batches))
            self.assertTrue(phase5_status["complete"])
            self.assertEqual(
                phase5_status["completed_method_stream_evaluations"], len(METHODS)
            )


if __name__ == "__main__":
    unittest.main()
