"""Tests for cross-fitted alarm recalibration."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np
import pandas as pd

from experiments.run_step7_alarm_recalibration import (
    _fit_cross_fitted_thresholds,
    _score_scenarios,
    _streamwise_feasibility,
    reference_normalized_score,
    split_conformal_threshold,
)


class AlarmRecalibrationTest(unittest.TestCase):
    def test_split_conformal_uses_finite_sample_rank(self):
        threshold, rank, count = split_conformal_threshold(range(160), 0.01)
        self.assertEqual(count, 160)
        self.assertEqual(rank, 160)
        self.assertEqual(threshold, 159.0)

        threshold, rank, count = split_conformal_threshold(range(40), 0.01)
        self.assertEqual((rank, count), (41, 40))
        self.assertTrue(math.isinf(threshold))

    def test_reference_normalization_uses_pair_specific_null_scale(self):
        frame = pd.DataFrame(
            {
                "monitor_score": [2.0, 5.0],
                "null_score_median": [1.0, 1.0],
                "alarm_threshold": [2.0, 3.0],
            }
        )
        np.testing.assert_allclose(reference_normalized_score(frame), [1.0, 2.0])

    def test_cross_fit_excludes_held_out_seed(self):
        records = []
        for seed in [42, 43, 44, 45, 46]:
            for batch_index in range(10):
                records.append(
                    {
                        "scenario_id": f"null-{seed}",
                        "dataset": "adult",
                        "model": "rf",
                        "monitor": "confidence",
                        "shift": "no_shift",
                        "severity": "none",
                        "mode": "static",
                        "seed": seed,
                        "batch_index": batch_index,
                        "normalized_score": seed + batch_index / 10,
                    }
                )
        thresholds = _fit_cross_fitted_thresholds(pd.DataFrame(records), [0.05])
        self.assertEqual(len(thresholds), 10)
        self.assertTrue((thresholds["calibration_seed_count"] == 4).all())
        self.assertTrue((thresholds["calibration_batches"] == 40).all())
        for row in thresholds.itertuples(index=False):
            self.assertNotIn(
                row.held_out_seed, json.loads(row.calibration_seeds_json)
            )

    def test_alarm_scoring_joins_events_only_after_decisions(self):
        alarms = pd.DataFrame(
            {
                "scenario_id": ["s"] * 4,
                "dataset": ["adult"] * 4,
                "model": ["rf"] * 4,
                "monitor": ["confidence"] * 4,
                "shift": ["pipeline"] * 4,
                "severity": ["high"] * 4,
                "mode": ["abrupt"] * 4,
                "seed": [42] * 4,
                "batch_index": [0, 1, 2, 3],
                "policy": ["dataset_monitor"] * 4,
                "alpha": [0.01] * 4,
                "normalized_score": [0.0, 0.0, 2.0, 2.0],
                "normalized_threshold": [1.0] * 4,
                "recalibrated_alarm": [False, False, True, True],
            }
        )
        events = pd.DataFrame(
            {
                "scenario_id": ["s"] * 4,
                "batch_index": [0, 1, 2, 3],
                "alarm_target_event": [False, False, True, True],
            }
        )
        _, scenarios = _score_scenarios(alarms, events)
        row = scenarios.iloc[0]
        self.assertEqual(row["false_alarm_rate"], 0.0)
        self.assertEqual(row["detection_delay"], 0.0)
        self.assertTrue(row["detected_event"])

    def test_one_percent_streamwise_control_requires_more_null_trajectories(self):
        records = []
        for dataset in ["adult", "covertype"]:
            for seed in [42, 43, 44, 45, 46]:
                for model in ["rf", "xgb"]:
                    records.append(
                        {
                            "scenario_id": f"{dataset}-{seed}-{model}",
                            "dataset": dataset,
                            "model": model,
                            "monitor": "confidence",
                            "shift": "no_shift",
                            "seed": seed,
                        }
                    )
        feasibility = _streamwise_feasibility(pd.DataFrame(records), [0.01])
        self.assertFalse(feasibility["streamwise_control_feasible"].any())
        self.assertTrue(
            (
                feasibility[
                    "minimum_for_finite_split_conformal_threshold"
                ]
                == 99
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
