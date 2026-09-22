import time
import tracemalloc
import pandas as pd
from typing import Dict, Any, Callable, Tuple
from .metrics import (
    compute_risk_metrics, 
    compute_attribution_metrics,
    compute_reliability_metrics
)

class TABMONProtocol:
    """
    B6: Giao thức đánh giá chuẩn (Locked Protocol) cho TABMON-Bench.
    Đảm bảo mọi Monitor bị đánh giá trên cùng một bộ tiêu chuẩn về độ chính xác và tài nguyên.
    """
    def __init__(self, k_features: int = 3):
        self.k_features = k_features

    def profile_monitor(self, monitor_func: Callable, target_X: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        B5: Chạy monitor và đo lường Runtime & RAM (Efficiency metrics).
        """
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            # The monitor only receives features; labels stay with the oracle.
            report = monitor_func(target_X)
            if not isinstance(report, dict):
                raise TypeError("Monitor output must be a dictionary")
        finally:
            end_time = time.perf_counter()
            _, peak_ram = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        
        efficiency_metrics = {
            "runtime_seconds": end_time - start_time,
            "peak_ram_mb": peak_ram / (1024 * 1024)
        }
        
        return report, efficiency_metrics

    def evaluate_batch(self, 
                       monitor_report: Dict[str, Any], 
                       ground_truth_attr: Dict[str, float], 
                       true_delta_loss: float) -> Dict[str, float]:
        """
        Chấm điểm toàn diện một báo cáo từ Monitor trên 1 batch.
        """
        results = {}
        
        # 1. Chấm điểm Risk (Nếu monitor có trả về estimated risk)
        pred_risk = monitor_report.get('estimated_risk_change')
        if pred_risk is not None:
            # Ghi nhận error tuyệt đối trên batch này
            results['risk_abs_error'] = abs(true_delta_loss - pred_risk)
            
            # Đánh giá Reliability (nếu có Confidence Interval)
            ci_lower = monitor_report.get('risk_ci_lower')
            ci_upper = monitor_report.get('risk_ci_upper')
            if ci_lower is not None and ci_upper is not None:
                rel_mets = compute_reliability_metrics(true_delta_loss, ci_lower, ci_upper)
                results.update(rel_mets)
        
        # 2. Chấm điểm Attribution
        pred_attr = monitor_report.get('feature_attribution', {})
        attr_mets = compute_attribution_metrics(ground_truth_attr, pred_attr, k=self.k_features)
        results.update(attr_mets)
        
        return results
