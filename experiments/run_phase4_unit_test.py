import os
import pandas as pd
import joblib
import sys

# Thêm đường dẫn project vào sys.path để chạy từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from src.shift_generator import TabularShiftGenerator
from src.baselines.b1_drift_shap import DriftSHAPMonitor
from src.evaluation.oracle_evaluator import OracleEvaluator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'adult')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'results', 'base_models', 'models', 'adult', 'lr_calibrated.pkl')

if __name__ == "__main__":
    print("=== B7: UNIT TEST MONITORING BASELINES ===")
    
    # 1. Tải dữ liệu và mô hình
    ref_df = pd.read_parquet(os.path.join(DATA_DIR, 'calibration.parquet'))
    test_pool = pd.read_parquet(os.path.join(DATA_DIR, 'test_pool.parquet'))
    model = joblib.load(MODEL_PATH)
    
    target_col = 'income'
    ref_X = ref_df.drop(columns=[target_col])
    ref_y = ref_df[target_col]
    
    # 2. Sinh lỗi (Single Covariate Shift) trên feature 'age'
    print("[*] Đang sinh lỗi Covariate Shift trên feature 'age' (Severity: High)...")
    generator = TabularShiftGenerator(test_pool, target_col=target_col)
    stream_batches, ground_truth = generator.shift_b1_single_covariate(
        feature='age', severity='high', mode='static', num_batches=1, batch_size=1000
    )
    target_batch = stream_batches[0]
    
    # 3. Chạy Monitor Baseline 1
    print("[*] Khởi tạo Drift + SHAP Monitor...")
    monitor = DriftSHAPMonitor(ref_X, ref_y, model)
    
    target_X = target_batch.drop(columns=[target_col])
    report = monitor.analyze_batch(target_X)
    
    # 4. Trọng tài chấm điểm
    print("[*] Đang chấm điểm Attribution...")
    evaluator = OracleEvaluator(model, ref_df, target_col)
    scores = evaluator.evaluate_attribution(report['feature_attribution'], ground_truth)
    
    print("\n=== KẾT QUẢ UNIT TEST ===")
    print(f"Top 3 Feature bị Monitor nghi ngờ:")
    sorted_attr = sorted(report['feature_attribution'].items(), key=lambda x: x[1], reverse=True)[:3]
    for k, v in sorted_attr:
        print(f"  - {k}: {v:.4f}")
        
    print(f"\nMetric Đánh Giá:")
    print(f"  - Top-1 Accuracy: {scores['top1_accuracy']}")
    print(f"  - Rank Correlation (Spearman): {scores['rank_correlation']:.4f}")
