import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import shap
from typing import Dict, Any
from sklearn.inspection import permutation_importance
from .abstract_monitor import AbstractMonitor

class DriftSHAPMonitor(AbstractMonitor):
    def __init__(self, reference_X: pd.DataFrame, reference_y: pd.Series, model: Any):
        super().__init__(reference_X, reference_y, model)
        self._compute_reference_shap()

    def _compute_reference_shap(self):
        """Tính toán Global Feature Importance bằng SHAP trên tập Reference."""
        print("    [DriftSHAP] Computing reference feature importance...")
        # Lấy mẫu 1000 dòng để tính SHAP cho nhanh (CPU friendly)
        sample_X = self.reference_X.sample(n=min(1000, len(self.reference_X)), random_state=42)
        
        # Thử TreeExplainer (nhanh cho CatBoost/XGB), fallback sang KernelExplainer
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sample_X)
        except Exception:
            # Wrap predict_proba để chỉ lấy class 1
            sample_y = self.reference_y.loc[sample_X.index]
            result = permutation_importance(
                self.model,
                sample_X,
                sample_y,
                scoring='neg_log_loss',
                n_repeats=3,
                random_state=42,
                n_jobs=-1,
            )
            self.global_shap = np.abs(result.importances_mean)
            self.shap_dict = dict(zip(self.features, self.global_shap))
            return
            
        # Nếu output là list (multi-class), lấy class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        # Global importance = Mean absolute SHAP
        self.global_shap = np.abs(shap_values).mean(axis=0)
        self.shap_dict = dict(zip(self.features, self.global_shap))

    def analyze_batch(self, target_X: pd.DataFrame, target_y: pd.Series = None) -> Dict[str, Any]:
        """Tính KS-Statistic Drift và nhân với SHAP."""
        attribution = {}
        total_drift = 0.0
        
        for f in self.features:
            ref_val = self.reference_X[f].values
            tgt_val = target_X[f].values
            
            # Tính KS-statistic (Khoảng cách giữa 2 phân phối biên, giá trị từ 0 đến 1)
            if pd.api.types.is_numeric_dtype(self.reference_X[f]):
                stat, _ = ks_2samp(ref_val, tgt_val, nan_policy='omit')
            else:
                ref_freq = pd.Series(ref_val).value_counts(normalize=True, dropna=False)
                tgt_freq = pd.Series(tgt_val).value_counts(normalize=True, dropna=False)
                categories = ref_freq.index.union(tgt_freq.index)
                stat = 0.5 * np.abs(
                    ref_freq.reindex(categories, fill_value=0.0)
                    - tgt_freq.reindex(categories, fill_value=0.0)
                ).sum()
            
            # Feature Attribution = Drift Score * Model Importance (SHAP)
            impact = stat * self.shap_dict.get(f, 0.0)
            attribution[f] = impact
            total_drift += impact
            
        # Chuẩn hóa Attribution về tổng 1.0 để dễ chấm điểm Rank Correlation
        if total_drift > 0:
            normalized_attribution = {k: v / total_drift for k, v in attribution.items()}
        else:
            normalized_attribution = {f: 0.0 for f in self.features}
            
        return {
            'estimated_risk_change': total_drift, # Dùng tổng drift thô như một proxy cho risk change
            'feature_attribution': normalized_attribution
        }   
