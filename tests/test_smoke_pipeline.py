import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.base_models import BaseModelsTrainer
from src.baselines.b1_drift_shap import DriftSHAPMonitor
from src.evaluation.oracle_evaluator import OracleEvaluator
from src.evaluation.protocal import TABMONProtocol
from src.shift_generator import TabularShiftGenerator


class EndToEndSmokeTest(unittest.TestCase):
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

            self.assertIn('risk_abs_error', scores)
            self.assertIn('ndcg@1', scores)
            self.assertGreaterEqual(efficiency['runtime_seconds'], 0.0)


if __name__ == '__main__':
    unittest.main()
