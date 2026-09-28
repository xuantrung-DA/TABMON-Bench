import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from src.shift_generator import TabularShiftGenerator
from src.baselines.b1_drift_shap import DriftSHAPMonitor
from src.evaluation.oracle_evaluator import OracleEvaluator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "adult")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "base_models",
    "models",
    "adult",
    "lr_calibrated.pkl",
)

if __name__ == "__main__":
    print("=== B7: UNIT TEST MONITORING BASELINES ===")

    ref_df = pd.read_parquet(os.path.join(DATA_DIR, "calibration.parquet"))
    test_pool = pd.read_parquet(os.path.join(DATA_DIR, "test_pool.parquet"))
    model = joblib.load(MODEL_PATH)

    target_col = "income"
    ref_X = ref_df.drop(columns=[target_col])
    ref_y = ref_df[target_col]

    print("[*] Generating high-severity covariate shift on 'age'...")
    generator = TabularShiftGenerator(test_pool, target_col=target_col)
    stream_batches, ground_truth = generator.shift_b1_single_covariate(
        feature="age",
        severity="high",
        mode="static",
        num_batches=1,
        batch_size=1000,
    )
    target_batch = stream_batches[0]

    print("[*] Initializing the DriftSHAP-style monitor...")
    monitor = DriftSHAPMonitor(ref_X, ref_y, model)

    target_X = target_batch.drop(columns=[target_col])
    report = monitor.analyze_batch(target_X)

    print("[*] Scoring intervention attribution...")
    evaluator = OracleEvaluator(model, ref_df, target_col)
    scores = evaluator.evaluate_attribution(
        report["feature_attribution"], ground_truth
    )

    print("\n=== SMOKE-TEST RESULT ===")
    print("Top three attributed features:")
    sorted_attr = sorted(
        report["feature_attribution"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    for k, v in sorted_attr:
        print(f"  - {k}: {v:.4f}")

    print("\nAttribution metrics:")
    print(f"  - Top-1 accuracy: {scores['top1_accuracy']}")
    print(f"  - Spearman rank correlation: {scores['rank_correlation']:.4f}")
