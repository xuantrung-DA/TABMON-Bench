import inspect
import json
import unittest

import numpy as np
import pandas as pd

from src.baselines.schema_guard import SchemaGuardProfile


class SchemaGuardTest(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame(
            {
                "numeric": np.linspace(0.0, 10.0, 2000),
                "category": pd.Series(["a", "b"] * 1000, dtype="string"),
            }
        )
        self.profile = SchemaGuardProfile.fit(self.reference)

    def test_pipeline_sentinel_is_a_hard_range_violation(self):
        batch = self.reference.iloc[:100].copy()
        batch.loc[:79, "numeric"] = -999.0
        result = self.profile.score_batch(batch)
        self.assertEqual(result.top_feature, "numeric")
        self.assertEqual(result.top_component, "hard_range_violation_rate")
        self.assertAlmostEqual(result.score, 0.8)

    def test_exact_reference_has_zero_positive_deviation(self):
        result = self.profile.score_batch(self.reference)
        self.assertAlmostEqual(result.score, 0.0)
        components = json.loads(result.component_scores_json)
        self.assertEqual(components["numeric"]["hard_range_violation_rate"], 0.0)

    def test_novel_category_is_detected(self):
        batch = self.reference.iloc[:100].copy()
        batch.loc[:49, "category"] = "unseen"
        result = self.profile.score_batch(batch)
        self.assertEqual(result.top_feature, "category")
        self.assertEqual(result.top_component, "novel_category_rate")
        self.assertAlmostEqual(result.score, 0.5)

    def test_api_has_no_label_or_oracle_argument(self):
        for method in (SchemaGuardProfile.fit, SchemaGuardProfile.score_batch):
            parameters = {name.lower() for name in inspect.signature(method).parameters}
            forbidden = {"label", "labels", "target", "y", "oracle", "risk", "failure"}
            self.assertTrue(parameters.isdisjoint(forbidden))

    def test_column_order_change_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self.profile.score_batch(self.reference[["category", "numeric"]])


if __name__ == "__main__":
    unittest.main()

