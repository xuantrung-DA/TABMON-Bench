import os
import gc
import sys
import traceback
import pandas as pd
import joblib
from itertools import product

# Import các module đã xây dựng từ Phase 1 đến 5
from src.shift_generator import TabularShiftGenerator
from src.evaluation.protocal import TABMONProtocol
from src.evaluation.oracle_evaluator import OracleEvaluator
from src.baselines.b1_drift_shap import DriftSHAPMonitor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# ==========================================
# CẤU HÌNH BENCHMARK (B2, B3, B4)
# ==========================================
# B1: Đổi thành True để chạy Smoke Test (1 dataset, 1 model, 1 shift, 1 seed)
SMOKE_TEST = True 

DATASETS = ['adult', 'bank_marketing', 'acs_income', 'covertype']
MODELS = ['lr', 'rf', 'xgb', 'mlp']
SHIFT_FAMILIES = ['covariate', 'correlated', 'support', 'pipeline', 'concept']
SEVERITIES = ['low', 'medium', 'high']
SEEDS = [42, 43, 44, 45, 46]

if SMOKE_TEST:
    DATASETS, MODELS = ['adult'], ['lr']
    SHIFT_FAMILIES, SEVERITIES, SEEDS = ['covariate'], ['high'], [42]

# Map target và feature cho từng dataset (tùy chỉnh theo thực tế)
DS_CONFIG = {
    'adult': {'target': 'income', 'shift_feature': 'age', 'corr_features': ['age', 'education']},
    'bank_marketing': {'target': 'y', 'shift_feature': 'balance', 'corr_features': ['balance', 'duration']},
    'acs_income': {'target': 'PINCP', 'shift_feature': 'AGEP', 'corr_features': ['AGEP', 'WKHP']},
    'covertype': {'target': 'Cover_Type', 'shift_feature': 'Elevation', 'corr_features': ['Elevation', 'Slope']}
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'results', 'tables', 'aggregate_metrics.csv')
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

def run_benchmark():
    if SMOKE_TEST:
        print("!!! SMOKE TEST MODE !!!")
    protocol = TABMONProtocol(k_features=3)
    results_records = []
    
    total_runs = len(DATASETS) * len(MODELS) * len(SHIFT_FAMILIES) * len(SEVERITIES) * len(SEEDS)
    current_run = 0

    for ds_name in DATASETS:
        target_col = DS_CONFIG[ds_name]['target']
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', ds_name)
        
        # Tải dữ liệu một lần cho mỗi dataset
        ref_df = pd.read_parquet(os.path.join(data_dir, 'calibration.parquet'))
        test_pool = pd.read_parquet(os.path.join(data_dir, 'test_pool.parquet'))
        ref_X, ref_y = ref_df.drop(columns=[target_col]), ref_df[target_col]

        for model_name in MODELS:
            model_path = os.path.join(PROJECT_ROOT, 'results', 'base_models', 'models', ds_name, f'{model_name}_calibrated.pkl')
            if not os.path.exists(model_path):
                print(f"[!] Bỏ qua {ds_name} - {model_name}: Không tìm thấy model.")
                continue
                
            model = joblib.load(model_path)
            evaluator = OracleEvaluator(model, ref_df, target_col)

            # Khởi tạo Baseline Monitor (Có thể mở rộng thêm XPE, ODD vào đây)
            # DriftSHAPMonitor chỉ cần khởi tạo 1 lần (tính reference SHAP)
            monitor = DriftSHAPMonitor(ref_X, ref_y, model)

            for seed in SEEDS:
                generator = TabularShiftGenerator(test_pool, target_col, random_seed=seed)
                
                for shift_type, severity in product(SHIFT_FAMILIES, SEVERITIES):
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] Running: {ds_name} | {model_name} | {shift_type} | {severity} | Seed {seed}")
                    
                    try: # B6: Khối try-except bảo vệ luồng chạy
                        # 1. Sinh Stream dựa trên loại Shift
                        f_shift = DS_CONFIG[ds_name]['shift_feature']
                        
                        if shift_type == 'covariate':
                            stream, gt_attr = generator.shift_b1_single_covariate(f_shift, severity)
                        elif shift_type == 'correlated':
                            stream, gt_attr = generator.shift_b2_correlated_multi(DS_CONFIG[ds_name]['corr_features'], severity)
                        elif shift_type == 'support':
                            stream, gt_attr = generator.shift_b4_support_violation(f_shift, severity)
                        elif shift_type == 'pipeline':
                            stream, gt_attr = generator.shift_b5_pipeline_corruption(f_shift, severity)
                        elif shift_type == 'concept':
                            # Ví dụ: Hàm điều kiện cho concept shift (X > mean)
                            cond = lambda df: df[f_shift] > df[f_shift].mean()
                            stream, gt_attr = generator.shift_b6_concept_shift_negative_control(cond, severity)
                        else:
                            raise ValueError(f"Unknown shift family: {shift_type}")

                        batch_metrics = []
                        
                        # 2. Xử lý từng batch trong Stream
                        for batch_idx, target_batch in enumerate(stream):
                            target_X = target_batch.drop(columns=[target_col])
                            
                            # Oracle tính True Risk
                            true_delta_loss = evaluator.calculate_true_excess_risk(target_batch)
                            
                            # Protocol đo lường Monitor
                            report, eff_metrics = protocol.profile_monitor(monitor.analyze_batch, target_X)
                            
                            # Protocol chấm điểm
                            eval_scores = protocol.evaluate_batch(report, gt_attr, true_delta_loss)
                            eval_scores.update(eff_metrics)
                            batch_metrics.append(eval_scores)

                        # 3. B5: Tính trung bình Aggregate (Không lưu file rác)
                        agg_metrics = pd.DataFrame(batch_metrics).mean().to_dict()
                        agg_metrics.update({
                            'dataset': ds_name, 'model': model_name, 
                            'shift': shift_type, 'severity': severity, 'seed': seed
                        })
                        results_records.append(agg_metrics)
                        
                    except Exception as e:
                        print(f"[ERROR] Thất bại tại kịch bản {shift_type}-{severity}: {str(e)}")
                        traceback.print_exc()

            # Giải phóng RAM sau mỗi mô hình (quan trọng cho máy 32GB RAM)
            del model, monitor, evaluator
            gc.collect()

    # B5: Lưu tổng hợp kết quả
    if results_records:
        df_results = pd.DataFrame(results_records)
        df_results.to_csv(RESULTS_FILE, index=False)
        print(f"\n[v] Hoàn thành! Aggregate Results được lưu an toàn tại {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark()
