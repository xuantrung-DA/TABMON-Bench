import unittest

import numpy as np
import pandas as pd

from experiments.analyze_v8_step9 import (
    _classification_frame,
    _design_level_counts,
    _outcome_equivalent_methods,
    paired_classification_comparisons,
    two_way_cell_bootstrap_mean,
)


class Step9AnalysisTest(unittest.TestCase):
    def test_outcome_equivalent_methods_detect_exact_streamwise_match(self):
        rows = []
        for stream, ac_value, cot_value in (("s1", 0, 1), ("s2", 1, 1)):
            for method, value in (
                ("ac", ac_value),
                ("doc", ac_value),
                ("cot", cot_value),
            ):
                rows.append(
                    {
                        "predictor_stream_id": stream,
                        "method": method,
                        "majority_failure": value,
                    }
                )
        self.assertEqual(
            _outcome_equivalent_methods(
                pd.DataFrame(rows), "majority_failure"
            ),
            ["doc"],
        )

    def test_design_level_counts_are_derived_from_the_frame(self):
        frame = pd.DataFrame(
            {
                "dataset": ["a", "a", "a", "b"],
                "seed": [1, 1, 2, 1],
                "predictor_stream_id": ["s1", "s1", "s2", "s3"],
            }
        )
        self.assertEqual(
            _design_level_counts(frame),
            {
                "datasets": 2,
                "seeds": 2,
                "dataset_seed_cells": 3,
                "predictor_streams": 3,
            },
        )

    def test_two_way_cell_bootstrap_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "dataset": ["a", "a", "b", "b"],
                "seed": [1, 2, 1, 2],
                "value": [0.0, 1.0, 2.0, 3.0],
            }
        )
        first = two_way_cell_bootstrap_mean(
            frame, "value", 1000, 0.95, np.random.default_rng(7)
        )[:4]
        second = two_way_cell_bootstrap_mean(
            frame, "value", 1000, 0.95, np.random.default_rng(7)
        )[:4]
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.5)

    def test_classification_pairing_rejects_different_oracles(self):
        rows = []
        for method_index, method in enumerate(("ac", "doc", "atc", "cot", "cott")):
            rows.append(
                {
                    "predictor_stream_id": "stream",
                    "endpoint": "classification_error",
                    "method": method,
                    "mean_true_value": float(method_index == 4),
                }
            )
        with self.assertRaisesRegex(ValueError, "same oracle endpoint"):
            _classification_frame(pd.DataFrame(rows))

    def test_paired_difference_direction_is_a_minus_b(self):
        rows = []
        methods = ("ac", "doc", "atc", "cot", "cott")
        for dataset in ("a", "b"):
            for seed in (1, 2):
                stream = f"{dataset}-{seed}"
                for index, method in enumerate(methods):
                    rows.append(
                        {
                            "predictor_stream_id": stream,
                            "dataset": dataset,
                            "model": "lr",
                            "shift": "no_shift",
                            "severity": "none",
                            "mode": "static",
                            "seed": seed,
                            "method": method,
                            "mean_absolute_error": float(index),
                            "risk_failure_rate": float(index) / 10.0,
                        }
                    )
        result = paired_classification_comparisons(
            pd.DataFrame(rows), 1000, np.random.default_rng(9)
        )
        row = result[
            result["scope"].eq("overall")
            & result["outcome"].eq("mean_absolute_error")
            & result["method_a"].eq("ac")
            & result["method_b"].eq("cot")
        ].iloc[0]
        self.assertEqual(row["difference_a_minus_b"], -3.0)
        self.assertEqual(row["ci_winner"], "ac")


if __name__ == "__main__":
    unittest.main()
