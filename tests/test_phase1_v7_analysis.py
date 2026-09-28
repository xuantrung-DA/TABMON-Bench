import json
import unittest

import numpy as np
import pandas as pd

from experiments.analyze_phase1_v7 import (
    attribution_metric_audit,
    attribution_recall_from_json,
    cluster_bootstrap_ratio,
)


class Phase1V7AnalysisTest(unittest.TestCase):
    def test_attribution_recall_handles_multiple_active_features(self):
        truth = json.dumps({"a": 0.5, "b": 0.5, "c": 0.0})
        prediction = json.dumps({"a": 0.9, "c": 0.8, "d": 0.7, "b": 0.1})
        self.assertEqual(attribution_recall_from_json(truth, prediction, 1), 0.5)
        self.assertEqual(attribution_recall_from_json(truth, prediction, 3), 0.5)
        self.assertEqual(attribution_recall_from_json(truth, prediction, 5), 1.0)

    def test_cluster_ratio_uses_ratio_of_sums(self):
        frame = pd.DataFrame(
            {
                "dataset": ["a", "a", "b", "b"],
                "seed": [1, 2, 1, 2],
                "reversed": [1, 0, 2, 1],
                "harmful": [2, 1, 4, 1],
            }
        )
        estimate, lower, upper = cluster_bootstrap_ratio(
            frame,
            "reversed",
            "harmful",
            ("dataset", "seed"),
            200,
            np.random.default_rng(7),
        )
        self.assertAlmostEqual(estimate, 4 / 8)
        self.assertLessEqual(lower, estimate)
        self.assertGreaterEqual(upper, estimate)

    def test_attribution_audit_explains_half_recall_as_non_failure(self):
        truth = json.dumps({"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0})
        prediction = json.dumps({"a": 0.9, "c": 0.8, "d": 0.7, "b": 0.1})
        frame = pd.DataFrame(
            {
                "scenario_id": ["s1"],
                "dataset": ["adult"],
                "model": ["lr"],
                "monitor": ["drift_shap"],
                "shift": ["correlated"],
                "severity": ["high"],
                "mode": ["abrupt"],
                "seed": [42],
                "batch_index": [5],
                "shift_fraction": [1.0],
                "ground_truth_attribution_json": [truth],
                "predicted_attribution_json": [prediction],
                "top3_recall": [0.5],
                "attribution_failure": [False],
            }
        )
        summary, distribution, scenario = attribution_metric_audit(frame)
        self.assertEqual(summary.loc[0, "recall_at_3"], 0.5)
        self.assertEqual(summary.loc[0, "failure_rate"], 0.0)
        self.assertEqual(distribution.loc[0, "recall_at_3_recomputed"], 0.5)
        self.assertEqual(scenario.loc[0, "scenario_mean_failure_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
