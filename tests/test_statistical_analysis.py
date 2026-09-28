import unittest

import numpy as np
import pandas as pd

from experiments.analyze_schema_v7 import (
    balanced_variance_components,
    benjamini_hochberg,
    build_scenario_metrics,
    severity_correlations,
    student_t_interval,
    two_way_cluster_bootstrap_mean,
)
from experiments.audit_reliability_output import _prepare_meta_monitor_split
from src.evaluation.meta_monitor_schema import (
    META_MONITOR_FEATURES,
    meta_monitor_allowlist_audit,
    meta_monitor_predictors,
    validate_meta_monitor_features,
)


class StatisticalAnalysisTest(unittest.TestCase):
    def test_meta_monitor_allowlist_rejects_every_oracle_class(self):
        forbidden = [
            "target_label",
            "true_excess_risk",
            "monitor_failure",
            "ground_truth_attribution_json",
            "label_derived_diagnostic",
        ]
        for column in forbidden:
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    validate_meta_monitor_features([column])

    def test_meta_monitor_predictors_cannot_include_offline_target(self):
        frame = pd.DataFrame(
            {
                "domain_classifier_auc": [0.5, 0.9],
                "prediction_confidence_shift": [0.0, 0.2],
                "monitor_failure": [False, True],
                "true_excess_risk": [0.0, 1.0],
                "target_label": [0, 1],
            }
        )
        predictors = meta_monitor_predictors(frame)
        self.assertEqual(
            list(predictors.columns),
            ["domain_classifier_auc", "prediction_confidence_shift"],
        )
        self.assertNotIn("monitor_failure", predictors)
        self.assertNotIn("true_excess_risk", predictors)
        self.assertNotIn("target_label", predictors)

    def test_meta_monitor_split_keeps_failure_label_out_of_x(self):
        frame = pd.DataFrame(
            {
                "domain_classifier_auc": [0.5, 0.6, 0.8, 0.9],
                "monitor_score": [0.0, 0.1, 0.2, 0.3],
                "monitor_failure": [False, True, False, True],
                "true_excess_risk": [0.0, 4.0, 0.0, 4.0],
            }
        )
        train_X, test_X, train_y, test_y, features = (
            _prepare_meta_monitor_split(frame.iloc[:2], frame.iloc[2:])
        )
        self.assertEqual(train_X.shape, (2, 2))
        self.assertEqual(test_X.shape, (2, 2))
        self.assertEqual(features, ["domain_classifier_auc", "monitor_score"])
        self.assertEqual(train_y.tolist(), [0, 1])
        self.assertEqual(test_y.tolist(), [0, 1])

    def test_meta_monitor_allowlist_has_observable_only_provenance(self):
        audit = meta_monitor_allowlist_audit()
        self.assertTrue(audit["target_is_separate_from_predictors"])
        self.assertEqual(
            [feature["name"] for feature in audit["features"]],
            list(META_MONITOR_FEATURES),
        )
        for feature in audit["features"]:
            self.assertFalse(feature["uses_target_labels"])
            self.assertFalse(feature["uses_oracle_risk"])
            self.assertFalse(feature["uses_failure_labels"])

    def test_scenario_metrics_use_only_post_shift_batches(self):
        rows = []
        for batch_index, (fraction, risk) in enumerate(
            [(0.0, 100.0), (0.0, 100.0), (0.5, 1.0), (1.0, 3.0)]
        ):
            rows.append(
                {
                    "scenario_id": "scenario",
                    "dataset": "adult",
                    "model": "lr",
                    "monitor": "confidence",
                    "shift": "concept",
                    "severity": "high",
                    "mode": "gradual",
                    "seed": 42,
                    "batch_index": batch_index,
                    "shift_fraction": fraction,
                    "true_excess_risk": risk,
                }
            )
        result = build_scenario_metrics(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "analysis_batch_count"], 2)
        self.assertAlmostEqual(result.loc[0, "true_excess_risk"], 2.0)

    def test_student_t_interval_contains_the_mean(self):
        mean, std, lower, upper, n = student_t_interval([1, 2, 3, 4, 5])
        self.assertEqual(n, 5)
        self.assertAlmostEqual(mean, 3.0)
        self.assertGreater(std, 0.0)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)

    def test_two_way_cluster_bootstrap_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "dataset": ["a", "a", "b", "b"],
                "seed": [1, 2, 1, 2],
                "value": [0.0, 1.0, 2.0, 3.0],
            }
        )
        first = two_way_cluster_bootstrap_mean(
            frame, "value", 100, 0.95, np.random.default_rng(7)
        )
        second = two_way_cluster_bootstrap_mean(
            frame, "value", 100, 0.95, np.random.default_rng(7)
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.5)

    def test_balanced_variance_decomposition_finds_dataset_effect(self):
        rows = []
        for dataset, value in (("a", 0.0), ("b", 10.0)):
            for seed in (1, 2, 3):
                rows.append({"dataset": dataset, "seed": seed, "outcome": value})
        result = balanced_variance_components(
            pd.DataFrame(rows),
            "outcome",
            ["dataset", "seed"],
            [],
        ).set_index("term")
        self.assertAlmostEqual(result.loc["dataset", "variance_fraction"], 1.0)
        self.assertAlmostEqual(result.loc["seed", "variance_fraction"], 0.0)
        self.assertAlmostEqual(
            result.loc["residual_and_higher_order", "variance_fraction"], 0.0
        )

    def test_benjamini_hochberg_is_monotone_and_bounded(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
        self.assertTrue(np.isnan(adjusted[-1]))
        self.assertTrue(np.all((adjusted[:3] >= 0.0) & (adjusted[:3] <= 1.0)))
        order = np.argsort([0.01, 0.04, 0.03])
        self.assertTrue(np.all(np.diff(adjusted[:3][order]) >= 0.0))

    def test_constant_severity_signal_is_recorded_as_zero_correlation(self):
        frame = pd.DataFrame(
            {
                "dataset": ["adult"] * 3,
                "model": ["lr"] * 3,
                "monitor": ["drift_shap"] * 3,
                "shift": ["concept"] * 3,
                "mode": ["abrupt"] * 3,
                "seed": [42] * 3,
                "severity": ["low", "medium", "high"],
                "true_excess_risk": [1.0, 2.0, 3.0],
                "monitor_score": [0.0, 0.0, 0.0],
            }
        )
        by_group, summary = severity_correlations(
            frame, 100, 0.95, np.random.default_rng(7)
        )
        constant = by_group[by_group["metric"].eq("monitor_score")].iloc[0]
        self.assertEqual(constant["spearman_rho"], 0.0)
        self.assertTrue(constant["constant_across_severity"])
        risk = summary[summary["metric"].eq("true_excess_risk")].iloc[0]
        self.assertEqual(risk["mean_spearman_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
