import unittest

import numpy as np
import pandas as pd

from src.baselines.cached_confidence import CachedConfidenceEstimator
from src.evaluation.alarm_calibration import stream_max_threshold


class SchemaV8Test(unittest.TestCase):
    def test_cached_confidence_is_probability_only_at_inference(self):
        predictions = pd.DataFrame(
            {
                "probability_class_0": [0.9, 0.8, 0.3, 0.2, 0.55, 0.45],
                "probability_class_1": [0.1, 0.2, 0.7, 0.8, 0.45, 0.55],
            }
        )
        labels = pd.Series([0, 0, 1, 1, 0, 1])
        estimator = CachedConfidenceEstimator.fit(predictions, labels, [0, 1])
        score = estimator.estimate_excess_log_loss(predictions.iloc[:3])
        self.assertTrue(np.isfinite(score))
        self.assertTrue(np.isfinite(estimator.reference_observed_risk))
        self.assertFalse(hasattr(estimator, "target_labels"))

    def test_stream_threshold_uses_maximum_of_each_trajectory(self):
        scores = np.arange(1000, dtype=float)
        threshold, maxima, rank = stream_max_threshold(scores, 100, 10, 0.01)
        np.testing.assert_array_equal(maxima, np.arange(9, 1000, 10))
        self.assertEqual(rank, 100)
        self.assertEqual(threshold, 999.0)


if __name__ == "__main__":
    unittest.main()
