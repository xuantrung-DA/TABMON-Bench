from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.baselines.performance_estimators import (
    PerformanceEstimatorSuite,
    multiclass_transport_costs,
)
from src.baselines.shd import SHDMonitor, hoeffding_width, pm_eb_lower_bound
from src.baselines.xpe import XPEExplainer, _restore_frame
from src.evaluation.risk import multiclass_log_losses
from src.shift_generator import TabularShiftGenerator
from experiments.merge_stream_caches import _json_equivalent, _parquet_equivalent
from experiments.analyze_step12_schema13 import (
    _classification,
    method_summaries,
    paired_comparisons,
)


def _predictions(probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            f"probability_class_{index}": probabilities[:, index]
            for index in range(probabilities.shape[1])
        }
    )
    frame["predicted_class"] = probabilities.argmax(axis=1)
    return frame


class _ToyProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        linear = 0.9 * np.asarray(frame["x"], dtype=float) - 0.4 * np.asarray(
            frame["z"], dtype=float
        )
        probability_one = expit(linear)
        return np.column_stack([1.0 - probability_one, probability_one])


class Step12Q1ExtensionTest(unittest.TestCase):
    @staticmethod
    def _toy_schema13_scenarios() -> pd.DataFrame:
        rows = []
        methods = ["ac", "doc", "atc", "cot", "cott"]
        for dataset in ["a", "b"]:
            for seed in [1, 2]:
                stream = f"{dataset}-{seed}"
                for position, method in enumerate(methods):
                    rows.append(
                        {
                            "predictor_stream_id": stream,
                            "dataset": dataset,
                            "model": "lr",
                            "shift": "covariate",
                            "severity": "low",
                            "mode": "abrupt",
                            "seed": seed,
                            "method": method,
                            "endpoint": "classification_error",
                            "mean_true_value": 0.2,
                            "mean_absolute_error": 0.01 * (position + 1),
                            "risk_failure_rate": float(position > 2),
                        }
                    )
        return pd.DataFrame(rows)

    def test_schema13_analysis_keeps_performance_estimators_paired(self):
        classification = _classification(self._toy_schema13_scenarios())
        summary = method_summaries(
            classification,
            "binary",
            100,
            np.random.default_rng(5),
        )
        pairs = paired_comparisons(
            classification,
            "binary",
            100,
            np.random.default_rng(5),
        )
        overall = summary[
            summary["scope"].eq("overall")
            & summary["outcome"].eq("mean_absolute_error")
        ].set_index("method")
        self.assertAlmostEqual(overall.loc["ac", "estimate"], 0.01)
        self.assertEqual(len(pairs[pairs["scope"].eq("overall")]), 20)
        self.assertTrue(
            set(pairs["method_a"]).union(pairs["method_b"])
            <= {"ac", "doc", "atc", "cot", "cott"}
        )

    def test_cache_merge_compares_shared_references_semantically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_json, right_json = root / "left.json", root / "right.json"
            left_json.write_text(
                json.dumps({"dataset": "adult", "reference_log_loss": 0.25}),
                encoding="utf-8",
            )
            right_json.write_text(
                json.dumps(
                    {"reference_log_loss": 0.25000000001, "dataset": "adult"},
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.assertTrue(_json_equivalent(left_json, right_json))
            left_parquet, right_parquet = root / "left.parquet", root / "right.parquet"
            pd.DataFrame({"p": np.asarray([0.1, 0.2], dtype=np.float32)}).to_parquet(
                left_parquet, index=False
            )
            pd.DataFrame({"p": np.asarray([0.1, 0.2], dtype=np.float64)}).to_parquet(
                right_parquet, index=False
            )
            self.assertTrue(_parquet_equivalent(left_parquet, right_parquet))

    def test_multiclass_log_loss_selects_declared_class(self):
        probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        losses = multiclass_log_losses(probabilities, [0, 1, 2], [0, 2])
        np.testing.assert_allclose(losses, -np.log([0.7, 0.6]))

    def test_multiclass_concept_shift_changes_labels_not_features(self):
        frame = pd.DataFrame(
            {"x": np.arange(30, dtype=float), "target": np.tile([0, 1, 2], 10)}
        )
        generator = TabularShiftGenerator(frame, "target", random_seed=7)
        stream, attribution = generator.shift_b6_concept_shift_negative_control(
            lambda data: pd.Series(True, index=data.index),
            "high",
            mode="static",
            num_batches=1,
            batch_size=30,
        )
        self.assertEqual(set(stream[0]["target"]), {0, 1, 2})
        self.assertTrue(all(value == 0 for value in attribution.values()))
        expected = stream[0]["target"].map({0: 2, 1: 0, 2: 1})
        # Reversing the cyclic map recovers the source label paired with each x.
        source_by_x = frame.set_index("x")["target"]
        observed_source = stream[0]["x"].map(source_by_x)
        pd.testing.assert_series_equal(
            expected.reset_index(drop=True),
            observed_source.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )

    def test_shd_inference_api_has_no_target_or_oracle_argument(self):
        parameters = set(inspect.signature(SHDMonitor.monitor).parameters)
        self.assertEqual(parameters, {"self", "target_X"})

    def test_shd_produces_absorbing_boolean_alarm(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(0.0, 1.0, 600)
        reference_X = pd.DataFrame({"x": x, "z": rng.normal(size=len(x))})
        probabilities = np.column_stack([np.full(len(x), 0.8), np.full(len(x), 0.2)])
        labels = (x > 0.8).astype(int)
        monitor = SHDMonitor.fit(
            reference_X,
            _predictions(probabilities),
            pd.Series(labels),
            [0, 1],
            random_seed=3,
            n_estimators=80,
        )
        output = monitor.monitor(reference_X.iloc[:80])
        self.assertTrue(monitor.calibration.selector_feasible)
        self.assertLessEqual(
            monitor.calibration.selector_fdp,
            monitor.calibration.fdp_limit + 1e-12,
        )
        self.assertEqual(len(output), 80)
        self.assertEqual(output["alarm"].dtype, bool)
        self.assertTrue(np.all(np.diff(output["alarm"].astype(int)) >= 0))
        self.assertTrue(
            (
                output["target_selected_pm_eb_lower"]
                <= output["running_selected_rate"] + 1e-12
            ).all()
        )

    def test_shd_bounds_follow_declared_constructions(self):
        self.assertGreater(hoeffding_width(100, 0.01), hoeffding_width(1000, 0.01))
        zeros = pm_eb_lower_bound(np.zeros(200), 0.005)
        ones = pm_eb_lower_bound(np.ones(200), 0.005)
        self.assertTrue(np.all((zeros >= 0) & (zeros <= 1)))
        self.assertTrue(np.all((ones >= 0) & (ones <= 1)))
        self.assertGreater(ones[-1], zeros[-1])

    def test_shd_inference_sensitivity_reuses_fitted_selector(self):
        rng = np.random.default_rng(14)
        x = rng.uniform(0.0, 1.0, 600)
        reference_X = pd.DataFrame({"x": x, "z": rng.normal(size=len(x))})
        probabilities = np.column_stack([np.full(len(x), 0.8), np.full(len(x), 0.2)])
        labels = (x > 0.8).astype(int)
        monitor = SHDMonitor.fit(
            reference_X,
            _predictions(probabilities),
            pd.Series(labels),
            [0, 1],
            random_seed=8,
            n_estimators=40,
        )
        relaxed = monitor.with_inference_parameters(alpha=0.10, epsilon=0.0)
        self.assertIs(relaxed.estimator, monitor.estimator)
        self.assertEqual(
            relaxed.calibration.predicted_error_threshold,
            monitor.calibration.predicted_error_threshold,
        )
        self.assertLessEqual(
            relaxed.calibration.source_selected_high_error_upper,
            monitor.calibration.source_selected_high_error_upper,
        )
        self.assertEqual(relaxed.calibration.epsilon, 0.0)

    def test_xpe_equal_mass_transport_and_shapley_efficiency(self):
        try:
            import ot  # noqa: F401
        except ImportError:
            self.skipTest("POT is not installed in the local test environment")
        source = pd.DataFrame(
            {"x": [-2.0, -1.0, 1.0, 2.0, 0.5, -0.5], "z": [0, 1, 0, 1, 2, 2]}
        )
        labels = pd.Series([0, 0, 1, 1, 1, 0])
        target = pd.DataFrame(
            {"x": [-1.8, -0.8, 1.2, 2.2, 0.7, -0.3], "z": [0, 1, 0, 1, 2, 2]}
        )
        result = XPEExplainer(source, labels).explain(
            _ToyProbabilityModel(),
            target,
            sample_size=6,
            permutations=32,
            random_seed=9,
        )
        self.assertEqual(result.source_transport_rows, result.target_transport_rows)
        self.assertAlmostEqual(result.coupling_retained_mass_fraction, 1.0, places=6)
        self.assertAlmostEqual(result.coupling_one_to_one_fraction, 1.0, places=6)
        self.assertLess(result.shapley_efficiency_max_abs_error, 1e-10)
        self.assertAlmostEqual(
            sum(result.attribution.values()), result.estimated_loss_change, places=10
        )

    def test_xpe_kernel_shap_agrees_with_exact_two_feature_shapley(self):
        try:
            import ot  # noqa: F401
            import shap  # noqa: F401
        except ImportError:
            self.skipTest("POT/SHAP is not installed in the local test environment")
        source = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "z": [0.0, 1.0, 0.0, 1.0]})
        labels = pd.Series([0, 0, 1, 1])
        target = pd.DataFrame({"x": [-1.7, -0.7, 1.3, 2.3], "z": [0.2, 1.2, 0.2, 1.2]})
        explainer = XPEExplainer(source, labels)
        permutation = explainer.explain(
            _ToyProbabilityModel(), target, sample_size=4, permutations=200, random_seed=13
        )
        kernel = explainer.explain(
            _ToyProbabilityModel(),
            target,
            sample_size=4,
            attribution_backend="kernel_shap",
            kernel_nsamples=3000,
            random_seed=13,
        )
        np.testing.assert_allclose(
            list(permutation.attribution.values()),
            list(kernel.attribution.values()),
            atol=1e-6,
        )

    def test_xpe_kernel_restore_preserves_pipeline_sentinel(self):
        template = pd.DataFrame(
            {
                "x": np.asarray([1, 2], dtype=np.int8),
                "z": pd.Categorical(["a", "b"]),
            }
        )
        restored = _restore_frame(
            np.asarray([[-999, "a"], [2, "b"]], dtype=object),
            template,
        )
        self.assertEqual(restored.loc[0, "x"], -999)
        self.assertEqual(restored["x"].dtype, np.dtype("int16"))

    def test_xpe_kernel_restore_preserves_float32_path(self):
        template = pd.DataFrame(
            {"x": np.asarray([1.25, 2.5], dtype=np.float32)}
        )
        restored = _restore_frame(
            np.asarray([[1.25], [-999.0]], dtype=object),
            template,
        )
        self.assertEqual(restored["x"].dtype, np.dtype("float32"))
        np.testing.assert_array_equal(
            restored["x"].to_numpy(),
            np.asarray([1.25, -999.0], dtype=np.float32),
        )

    def test_xpe_kernel_efficiency_survives_narrow_integer_promotion(self):
        try:
            import ot  # noqa: F401
            import shap  # noqa: F401
        except ImportError:
            self.skipTest("POT/SHAP is not installed in the local test environment")
        source = pd.DataFrame(
            {
                "x": np.asarray([-2, -1, 1, 2], dtype=np.int8),
                "z": np.asarray([0, 1, 0, 1], dtype=np.int8),
            }
        )
        labels = pd.Series([0, 0, 1, 1])
        target = source.astype({"x": np.int16}).copy()
        target.loc[0, "x"] = -999
        result = XPEExplainer(source, labels).explain(
            _ToyProbabilityModel(),
            target,
            sample_size=4,
            attribution_backend="kernel_shap",
            kernel_nsamples=3000,
            random_seed=13,
        )
        self.assertLess(result.shapley_efficiency_max_abs_error, 1e-10)

    def test_multiclass_performance_estimators_preserve_endpoint(self):
        try:
            import ot  # noqa: F401
        except ImportError:
            self.skipTest("POT is not installed in the local test environment")
        rng = np.random.default_rng(11)
        probabilities = rng.dirichlet([2, 2, 2], 180)
        labels = np.tile([0, 1, 2], 60)
        suite = PerformanceEstimatorSuite.fit(
            _predictions(probabilities), pd.Series(labels), [0, 1, 2]
        )
        estimates = suite.estimate(_predictions(probabilities[:60]))
        self.assertEqual(set(estimates), {"ac", "doc", "atc", "cot", "cott"})
        self.assertTrue(all(0 <= value <= 1 for value in estimates.values()))

    def test_multiclass_transport_conserves_per_sample_costs(self):
        try:
            import ot  # noqa: F401
        except ImportError:
            self.skipTest("POT is not installed in the local test environment")
        probabilities = np.asarray(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
        )
        costs = multiclass_transport_costs(probabilities, np.ones(3) / 3)
        self.assertEqual(costs.shape, (3,))
        self.assertTrue(np.all(costs >= 0))


if __name__ == "__main__":
    unittest.main()
