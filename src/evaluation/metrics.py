import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple

# ==========================================
# B1: Risk Estimation Metrics
# ==========================================
def compute_risk_metrics(true_risks: np.ndarray, pred_risks: np.ndarray) -> Dict[str, float]:
    """Đo lường độ chính xác của việc dự đoán Delta Loss tổng thể."""
    mae = mean_absolute_error(true_risks, pred_risks)
    rmse = np.sqrt(mean_squared_error(true_risks, pred_risks))
    
    # Bỏ qua warning nếu mảng hằng số (ví dụ: null stream)
    if len(np.unique(true_risks)) > 1 and len(np.unique(pred_risks)) > 1:
        pearson_corr, _ = pearsonr(true_risks, pred_risks)
        spearman_corr, _ = spearmanr(true_risks, pred_risks)
    else:
        pearson_corr, spearman_corr = 0.0, 0.0

    return {
        "risk_mae": mae,
        "risk_rmse": rmse,
        "risk_pearson": pearson_corr,
        "risk_spearman": spearman_corr
    }

# ==========================================
# B2: Attribution Metrics
# ==========================================
def compute_attribution_metrics(true_attr: Dict[str, float], pred_attr: Dict[str, float], k: int = 3) -> Dict[str, float]:
    """Chấm điểm việc phân bổ nguyên nhân lỗi cho từng feature."""
    features = list(true_attr.keys())
    y_true = np.array([true_attr[f] for f in features])
    y_pred = np.array([pred_attr.get(f, 0.0) for f in features])
    
    # 1. nDCG@k: Đánh giá chất lượng xếp hạng (ranking)
    # sklearn ndcg_score yêu cầu mảng 2D shape (n_samples, n_features)
    has_ground_truth = np.any(y_true != 0)
    ndcg_k = ndcg_score([y_true], [y_pred], k=k) if has_ground_truth else float('nan')
    
    # 2. Top-k Recall: Có bắt trúng các feature gây án trong top K không?
    active_count = int(np.count_nonzero(y_true))
    effective_k = min(k, active_count)
    if effective_k:
        top_k_true = set(np.array(features)[np.argsort(np.abs(y_true))[-effective_k:]])
        top_k_pred = set(np.array(features)[np.argsort(np.abs(y_pred))[-effective_k:]])
        top_k_recall = len(top_k_true & top_k_pred) / effective_k
    else:
        top_k_recall = float('nan')
    
    # 3. Sign Accuracy: Đoán đúng hướng (làm tăng hay giảm loss) không?
    # Chỉ tính trên các feature thực sự bị shift trong ground truth
    active_features_mask = y_true != 0
    if np.any(active_features_mask):
        sign_acc = np.mean(np.sign(y_true[active_features_mask]) == np.sign(y_pred[active_features_mask]))
    else:
        sign_acc = float('nan')
        
    return {
        f"ndcg@{k}": ndcg_k,
        f"top{k}_recall": top_k_recall,
        "sign_accuracy": sign_acc,
        "false_attribution_mass": float(np.sum(np.abs(y_pred))) if not has_ground_truth else 0.0,
    }

# ==========================================
# B3: Sequential Metrics
# ==========================================
def compute_sequential_metrics(alarms: List[bool], shift_point: int) -> Dict[str, float]:
    """Đánh giá chất lượng báo động trên luồng dữ liệu (Stream)."""
    alarms = np.array(alarms)
    n_batches = len(alarms)
    
    # Mọi báo động trước shift_point đều là False Alarm
    pre_shift_alarms = alarms[:shift_point]
    far = np.mean(pre_shift_alarms) if len(pre_shift_alarms) > 0 else 0.0
    
    # Mọi báo động từ shift_point trở đi là True Positive (Power)
    post_shift_alarms = alarms[shift_point:]
    power = np.mean(post_shift_alarms) if len(post_shift_alarms) > 0 else 0.0
    
    # Detection Delay: Số batch trôi qua kể từ khi shift xảy ra cho đến khi còi kêu lần đầu
    if len(post_shift_alarms) > 0 and np.any(post_shift_alarms):
        delay = np.argmax(post_shift_alarms) 
    else:
        delay = float('inf') # Missed detection
        
    return {
        "false_alarm_rate": far,
        "power": power,
        "detection_delay": delay
    }

# ==========================================
# B4: Reliability Metrics
# ==========================================
def compute_reliability_metrics(true_val: float, ci_lower: float, ci_upper: float) -> Dict[str, float]:
    """Đo lường chất lượng của Confidence Intervals / Bounds."""
    is_covered = 1 if (ci_lower <= true_val <= ci_upper) else 0
    width = ci_upper - ci_lower
    
    return {
        "coverage": float(is_covered),
        "interval_width": width
    }
