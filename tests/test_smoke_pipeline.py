import inspect
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.base_models import BaseModelsTrainer
from src.prepare_data import optimize_dtypes
from src.baselines.b0_confidence import ConfidenceShiftMonitor
from src.baselines.b1_drift_shap import DriftSHAPMonitor
from src.evaluation.oracle_evaluator import OracleEvaluator
from src.evaluation.protocol import TABMONProtocol
from src.evaluation.metrics import (
    compute_alarm_event_metrics,
    compute_attribution_metrics,
)
from src.shift_generator import TabularShiftGenerator
from experiments.run_core_benchmark import (
    _realized_shift_diagnostics,
    _resolve_alarm_target,
    build_scenarios,
    main as run_core_benchmark,
    parse_args,
)
from experiments.rescore_schema_v6 import rescore_batches
from experiments.rescore_schema_v7 import _recalibrate_confidence_alarms


class EndToEndSmokeTest(unittest.TestCase):
    def test_label_free_monitor_api_has_no_target_label_argument(self):
        forbidden = {"target_y", "target_labels", "labels", "oracle_risk"}
        for monitor_class in (ConfidenceShiftMonitor, DriftSHAPMonitor):
            parameters = set(
                inspect.signature(monitor_class.analyze_batch).parameters
            )
            self.assertTrue(parameters.isdisjoint(forbidden))

    def test_confidence_alarm_is_one_sided_for_risk_increase(self):
        monitor = object.__new__(ConfidenceShiftMonitor)
        monitor.reference_predicted_risk = 0.0
        monitor.reference_predicted_losses = np.zeros(5)
        monitor.calibration_residual_variance = 0.0
        monitor.alarm_quantile = 0.8
        monitor.null_scores = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        monitor.null_score_median = 0.0
        monitor.alarm_threshold = 0.2

        target = pd.DataFrame({'x': [0.0]})
        monitor._predicted_losses = lambda _: np.array([1.0])
        increase = monitor.analyze_batch(target)
        monitor._predicted_losses = lambda _: np.array([-1.0])
        decrease = monitor.analyze_batch(target)

        self.assertTrue(increase['alarm'])
        self.assertFalse(decrease['alarm'])
        self.assertEqual(increase['alarm_direction'], 'increase')
        self.assertEqual(
            ConfidenceShiftMonitor.capabilities['alarm_target'], 'risk_event'
        )

    def test_schema_v6_uses_monitor_specific_alarm_targets(self):
        target, event = _resolve_alarm_target(
            ConfidenceShiftMonitor.capabilities,
            true_risk_event=False,
            distribution_shift_event=True,
        )
        self.assertEqual(target, 'risk_event')
        self.assertFalse(event)

        target, event = _resolve_alarm_target(
            DriftSHAPMonitor.capabilities,
            true_risk_event=False,
            distribution_shift_event=True,
        )
        self.assertEqual(target, 'distribution_shift')
        self.assertTrue(event)

    def test_schema_v6_rescore_separates_early_warning_from_false_alarm(self):
        batches = pd.DataFrame({
            'scenario_id': ['confidence', 'confidence', 'drift', 'drift'],
            'batch_index': [0, 1, 0, 1],
            'monitor': ['confidence', 'confidence', 'drift_shap', 'drift_shap'],
            'shift': ['covariate'] * 4,
            'shift_fraction': [0.5, 1.0, 0.5, 1.0],
            'alarm': [True, False, True, False],
            'supports_alarm': [True] * 4,
            'risk_failure': [False, False, np.nan, np.nan],
            'attribution_failure': [np.nan, np.nan, False, False],
            'true_excess_risk': [0.01, 0.10, 0.01, 0.10],
            'true_risk_event': [False, True, False, True],
        })
        rescored = rescore_batches(batches, risk_event_threshold=0.05)

        self.assertTrue(rescored.loc[[0, 2], 'early_warning'].all())
        self.assertFalse(rescored.loc[[1, 3], 'early_warning'].any())
        self.assertTrue(bool(rescored.loc[0, 'alarm_failure']))
        self.assertFalse(bool(rescored.loc[2, 'alarm_failure']))
        self.assertEqual(rescored.loc[0, 'alarm_target'], 'risk_event')
        self.assertEqual(
            rescored.loc[2, 'alarm_target'], 'distribution_shift'
        )

    def test_target_aware_sequential_metrics_use_explicit_event_onset(self):
        metrics = compute_alarm_event_metrics(
            [False, False, True, True], [False, True, True, True]
        )
        self.assertEqual(metrics['event_batch'], 1.0)
        self.assertEqual(metrics['detection_delay'], 1.0)
        self.assertAlmostEqual(metrics['power'], 2 / 3)
        self.assertEqual(metrics['false_alarm_rate'], 0.0)

    def test_false_attribution_uses_raw_not_normalized_mass(self):
        protocol = TABMONProtocol(k_features=1)
        scores = protocol.evaluate_batch(
            {
                'estimated_risk_change': 0.0,
                'feature_attribution': {'age': 1.0, 'hours': 0.0},
                'raw_feature_attribution': {'age': 0.001, 'hours': 0.0},
            },
            {'age': 0.0, 'hours': 0.0},
            0.0,
        )

        self.assertAlmostEqual(scores['false_attribution_mass'], 0.001)
        self.assertAlmostEqual(scores['false_attribution_mass_normalized'], 1.0)

    def test_null_control_has_zero_intervention_attribution(self):
        frame = pd.DataFrame({
            'age': np.linspace(20.0, 70.0, 100),
            'income': np.tile([0, 1], 50),
        })
        generator = TabularShiftGenerator(frame, 'income', random_seed=123)
        batches, attribution = generator.shift_b0_no_shift(
            num_batches=4, batch_size=30
        )

        self.assertEqual(len(batches), 4)
        self.assertTrue(all(len(batch) == 30 for batch in batches))
        self.assertTrue(all(value == 0.0 for value in attribution.values()))

    def test_null_scenario_is_not_duplicated_by_severity_or_mode(self):
        args = parse_args([
            '--datasets', 'adult',
            '--models', 'lr',
            '--monitors', 'confidence',
            '--shifts', 'no_shift', 'covariate',
            '--severities', 'low', 'high',
            '--modes', 'abrupt', 'gradual',
            '--seeds', '42',
        ])
        scenarios = build_scenarios(args)
        null_scenarios = [s for s in scenarios if s.shift == 'no_shift']

        self.assertEqual(len(scenarios), 5)
        self.assertEqual(len(null_scenarios), 1)
        self.assertEqual(null_scenarios[0].severity, 'none')
        self.assertEqual(null_scenarios[0].mode, 'static')

    def test_pre_shift_batches_are_paired_across_shift_families(self):
        frame = pd.DataFrame({
            'age': np.linspace(20.0, 70.0, 100),
            'income': np.tile([0, 1], 50),
        })
        covariate_generator = TabularShiftGenerator(
            frame, 'income', random_seed=123
        )
        support_generator = TabularShiftGenerator(
            frame, 'income', random_seed=123
        )
        covariate_batches, _ = covariate_generator.shift_b1_single_covariate(
            'age', 'high', mode='abrupt', num_batches=4, batch_size=30
        )
        support_batches, _ = support_generator.shift_b4_support_violation(
            'age', 'high', mode='abrupt', num_batches=4, batch_size=30
        )

        pd.testing.assert_frame_equal(covariate_batches[0], support_batches[0])
        pd.testing.assert_frame_equal(covariate_batches[1], support_batches[1])

    def test_realized_diagnostics_preserve_support_shift_magnitude(self):
        reference = pd.DataFrame({'age': np.linspace(20.0, 70.0, 100)})
        scale = reference['age'].max() + reference['age'].std()
        low = reference.copy()
        high = reference.copy()
        low['age'] += 1.2 * scale
        high['age'] += 2.0 * scale

        low_metrics = _realized_shift_diagnostics(reference, low, ['age'])
        high_metrics = _realized_shift_diagnostics(reference, high, ['age'])

        self.assertGreater(
            high_metrics['realized_wasserstein_std'],
            low_metrics['realized_wasserstein_std'],
        )
        self.assertEqual(low_metrics['realized_support_violation_rate'], 1.0)
        self.assertEqual(high_metrics['realized_support_violation_rate'], 1.0)

    def test_attribution_recall_uses_the_requested_prediction_k(self):
        ground_truth = {'shifted': 1.0, 'first': 0.0, 'second': 0.0, 'third': 0.0}
        prediction = {'shifted': 0.5, 'first': 1.0, 'second': 0.8, 'third': 0.1}

        top1 = compute_attribution_metrics(ground_truth, prediction, k=1)
        top3 = compute_attribution_metrics(ground_truth, prediction, k=3)

        self.assertEqual(top1['top1_recall'], 0.0)
        self.assertEqual(top3['top3_recall'], 1.0)
        self.assertGreater(top3['ndcg@3'], 0.0)

    def test_optimize_dtypes_handles_arrow_strings(self):
        frame = pd.DataFrame({
            'numeric': pd.Series([1, 2, 3], dtype='int64'),
            'text': pd.Series(['State-gov', 'Private', None], dtype='string[pyarrow]'),
        })
        optimized = optimize_dtypes(frame)
        self.assertEqual(optimized['text'].dtype.name, 'category')
        self.assertTrue(np.issubdtype(optimized['numeric'].dtype, np.integer))

    def test_train_shift_monitor_and_oracle(self):
        rng = np.random.default_rng(7)
        n_rows = 300
        age = rng.normal(40, 12, n_rows)
        hours = rng.normal(40, 8, n_rows)
        workclass = rng.choice(['private', 'public', 'self'], n_rows).astype(object)
        workclass[::29] = np.nan
        income = (age + 0.6 * hours + 5 * (workclass == 'self') > 65).astype(int)
        frame = pd.DataFrame({
            'age': age,
            'hours': hours,
            'workclass': pd.Series(workclass, dtype='category'),
            'income': income,
        })

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / 'data' / 'adult'
            results_dir = root / 'results'
            data_dir.mkdir(parents=True)

            frame.iloc[:140].to_parquet(data_dir / 'train.parquet', index=False)
            frame.iloc[140:220].to_parquet(data_dir / 'calibration.parquet', index=False)
            frame.iloc[220:].to_parquet(data_dir / 'test_pool.parquet', index=False)

            trainer = BaseModelsTrainer('adult', str(root / 'data'), str(results_dir))
            metrics = trainer.train_and_calibrate(model_names=['lr', 'rf', 'xgb', 'mlp'])
            self.assertEqual(set(metrics['model']), {'LR', 'RF', 'XGB', 'MLP'})

            model_path = results_dir / 'models' / 'adult' / 'lr_calibrated.pkl'
            model = joblib.load(model_path)
            reference = frame.iloc[140:220].copy()
            test_pool = frame.iloc[220:].copy()

            generator = TabularShiftGenerator(test_pool, 'income', random_seed=11)
            batches, attribution = generator.shift_b1_single_covariate(
                'age', 'medium', mode='static', num_batches=1, batch_size=50
            )
            oracle = OracleEvaluator(model, reference, 'income')
            true_delta = oracle.calculate_true_excess_risk(batches[0])
            self.assertTrue(np.isfinite(true_delta))

            monitor = DriftSHAPMonitor(
                reference.drop(columns=['income']), reference['income'], model
            )
            protocol = TABMONProtocol(k_features=1)
            report, efficiency = protocol.profile_monitor(
                monitor.analyze_batch, batches[0].drop(columns=['income'])
            )
            scores = protocol.evaluate_batch(report, attribution, true_delta)

            self.assertNotIn('risk_abs_error', scores)
            self.assertIn('ndcg@1', scores)
            self.assertGreaterEqual(efficiency['runtime_seconds'], 0.0)

            confidence_monitor = ConfidenceShiftMonitor(
                reference.drop(columns=['income']),
                reference['income'],
                model,
                batch_size=50,
                null_calibration_batches=10,
            )
            confidence_report = confidence_monitor.analyze_batch(
                batches[0].drop(columns=['income'])
            )
            confidence_scores = protocol.evaluate_batch(
                confidence_report, attribution, true_delta
            )
            self.assertIn('risk_abs_error', confidence_scores)
            self.assertIn('coverage', confidence_scores)
            self.assertNotIn('ndcg@1', confidence_scores)

            v6_batches = pd.DataFrame({
                'scenario_id': ['synthetic', 'synthetic'],
                'batch_index': [0, 1],
                'dataset': ['adult', 'adult'],
                'model': ['lr', 'lr'],
                'monitor': ['confidence', 'confidence'],
                'estimated_risk_change': [
                    float(confidence_monitor.null_scores.min() - 1.0),
                    float(confidence_monitor.null_scores.max() + 1.0),
                ],
                'null_score_median': [
                    confidence_monitor.null_score_median,
                    confidence_monitor.null_score_median,
                ],
                'alarm': [True, False],
                'supports_alarm': [True, True],
                'true_risk_event': [False, True],
                'distribution_shift_event': [True, True],
                'alarm_target_event': [False, True],
                'risk_failure': [False, False],
                'attribution_failure': [np.nan, np.nan],
            })
            v7_batches = _recalibrate_confidence_alarms(
                v6_batches,
                {
                    'configuration': {
                        'batch_size': 50,
                        'null_calibration_batches': 10,
                        'alarm_quantile': 0.8,
                    },
                    'dataset_config': {'adult': {'target': 'income'}},
                },
                root / 'data',
                results_dir,
            )
            self.assertFalse(bool(v7_batches.loc[0, 'alarm']))
            self.assertTrue(bool(v7_batches.loc[1, 'alarm']))
            self.assertEqual(
                set(v7_batches['alarm_direction']), {'increase'}
            )

            core_results = root / 'core_results'
            exit_code = run_core_benchmark([
                '--smoke-test',
                '--data-dir', str(root / 'data'),
                '--base-models-dir', str(results_dir),
                '--results-dir', str(core_results),
                '--batch-size', '32',
                '--no-resume',
            ])
            self.assertEqual(exit_code, 0)

            aggregate = pd.read_parquet(core_results / 'aggregate_metrics.parquet')
            batch_metrics = pd.read_parquet(core_results / 'batch_metrics.parquet')
            status = pd.read_json(core_results / 'benchmark_status.json', typ='series')
            manifest = json.loads(
                (core_results / 'benchmark_manifest.json').read_text()
            )
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(len(batch_metrics), 4)
            self.assertTrue(bool(status['complete']))
            self.assertEqual(int(status['completed_scenarios']), 1)
            self.assertEqual(manifest['schema_version'], 7)

            pre_shift = batch_metrics[batch_metrics['shift_fraction'] == 0.0]
            post_shift = batch_metrics[batch_metrics['shift_fraction'] > 0.0]
            self.assertTrue(pre_shift['top3_recall'].isna().all())
            self.assertTrue(post_shift['top3_recall'].notna().all())
            for payload in pre_shift['ground_truth_attribution_json']:
                self.assertTrue(all(value == 0.0 for value in json.loads(payload).values()))
            self.assertIn('realized_wasserstein_std', batch_metrics.columns)
            self.assertIn('realized_true_risk_increase', aggregate.columns)
            self.assertEqual(int(aggregate.loc[0, 'attribution_scored_batches']), 2)
            self.assertTrue(batch_metrics['alarm'].notna().all())
            self.assertTrue(batch_metrics['estimated_risk_ci_lower'].isna().all())
            self.assertTrue(batch_metrics['estimated_risk_ci_upper'].isna().all())
            self.assertTrue(batch_metrics['risk_failure'].isna().all())
            self.assertTrue(batch_metrics['monitor_failure'].notna().all())
            self.assertTrue(batch_metrics['supports_attribution'].all())
            self.assertFalse(batch_metrics['supports_risk_estimation'].any())
            self.assertTrue(
                batch_metrics['alarm_target'].eq('distribution_shift').all()
            )
            self.assertTrue(
                batch_metrics['alarm_direction'].eq('increase').all()
            )
            expected_shift_event = batch_metrics['shift_fraction'].gt(0.0)
            self.assertTrue(
                batch_metrics['distribution_shift_event'].eq(
                    expected_shift_event
                ).all()
            )
            self.assertTrue(
                batch_metrics['alarm_target_event'].eq(
                    expected_shift_event
                ).all()
            )
            self.assertIn('shift_alarm_power', aggregate.columns)
            self.assertIn('risk_alarm_power', aggregate.columns)
            self.assertIn('domain_classifier_auc', batch_metrics.columns)
            self.assertIn('attribution_abs_mass', batch_metrics.columns)

            resume_exit_code = run_core_benchmark([
                '--smoke-test',
                '--data-dir', str(root / 'data'),
                '--base-models-dir', str(results_dir),
                '--results-dir', str(core_results),
                '--batch-size', '32',
            ])
            self.assertEqual(resume_exit_code, 0)
            resumed = pd.read_parquet(core_results / 'aggregate_metrics.parquet')
            self.assertEqual(len(resumed), 1)


if __name__ == '__main__':
    unittest.main()
