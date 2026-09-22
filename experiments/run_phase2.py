import os
import sys
import pandas as pd
from src.base_models import BaseModelsTrainer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# Cấu hình đường dẫn thống nhất
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'base_models')

DATASETS = ['adult', 'bank_marketing', 'acs_income', 'covertype']

if __name__ == "__main__":
    print("=== KÍCH HOẠT PHASE 2: TRAIN BASE MODELS & CALIBRATION ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_metrics = []
    
    for ds in DATASETS:
        print(f"\n[*] Đang xử lý Dataset: {ds}")
        trainer = BaseModelsTrainer(dataset_name=ds, data_dir=DATA_DIR, results_dir=RESULTS_DIR)
        
        try:
            ds_metrics_df = trainer.train_and_calibrate()
            all_metrics.append(ds_metrics_df)
        except Exception as e:
            print(f"[!] Lỗi khi xử lý {ds}: {str(e)}")
            
    # Tổng hợp và lưu bảng Baseline Metrics
    if all_metrics:
        final_metrics_df = pd.concat(all_metrics, ignore_index=True)
        print("\n=== KẾT QUẢ ĐÁNH GIÁ BASELINE (TEST POOL) ===")
        print(final_metrics_df.to_string(index=False))
        
        metrics_file = os.path.join(RESULTS_DIR, 'baseline_metrics_summary.csv')
        final_metrics_df.to_csv(metrics_file, index=False)
        print(f"\n[v] Toàn bộ Model, Predictions, Logits và Metrics đã được lưu tại: {RESULTS_DIR}")
    else:
        print("\n[!] Không có kết quả nào được ghi nhận.")
