import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Callable

class TabularShiftGenerator:
    def __init__(self, test_pool_df: pd.DataFrame, target_col: str, random_seed: int = 42):
        self.df = test_pool_df.copy()
        self.target_col = target_col
        self.rng = np.random.default_rng(random_seed)
        self.features = [c for c in self.df.columns if c != target_col]
        
        # B7: Ánh xạ mức độ nghiêm trọng (Severity Mapping)
        self.severity_map = {
            'low': 0.5,
            'medium': 1.5,
            'high': 3.0
        }

    def _rejection_sampling(self, data: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
        """Hàm dùng chung để lấy mẫu lại (resample) dựa trên trọng số"""
        if data.empty:
            raise ValueError("Cannot sample from an empty dataframe")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(data),) or not np.all(np.isfinite(weights)):
            raise ValueError("Shift weights must be finite and match the data length")
        total = weights.sum()
        if total <= 0:
            raise ValueError("Shift weights must have a positive sum")
        sampled_positions = self.rng.choice(
            len(data), size=len(data), replace=True, p=weights / total
        )
        return data.iloc[sampled_positions].copy()

    def _create_stream(self, 
                       base_df: pd.DataFrame, 
                       shifted_df: pd.DataFrame, 
                       mode: str, 
                       num_batches: int, 
                       batch_size: int) -> List[pd.DataFrame]:
        """
        B3: Tạo luồng dữ liệu (Stream) dạng Abrupt (Đột ngột) hoặc Gradual (Từ từ)
        """
        if num_batches < 1 or batch_size < 1:
            raise ValueError("num_batches and batch_size must both be positive")
        if mode not in {'abrupt', 'gradual', 'static'}:
            raise ValueError(f"Unknown stream mode: {mode}")

        stream = []
        for i in range(num_batches):
            if mode == 'abrupt':
                # Nửa đầu bình thường, nửa sau bóp méo hoàn toàn
                alpha = 0.0 if i < num_batches // 2 else 1.0
            elif mode == 'gradual':
                # Mức độ bóp méo tăng dần tuyến tính từ 0 đến 1
                alpha = i / (num_batches - 1) if num_batches > 1 else 1.0
            else:
                alpha = 1.0 # Static shift (bị lỗi từ đầu)

            # Trộn base_df và shifted_df theo tỷ lệ alpha
            n_shifted = int(batch_size * alpha)
            n_base = batch_size - n_shifted
            
            batch_base = base_df.sample(n=n_base, replace=True, random_state=int(self.rng.integers(1e5))) if n_base > 0 else pd.DataFrame()
            batch_shift = shifted_df.sample(n=n_shifted, replace=True, random_state=int(self.rng.integers(1e5))) if n_shifted > 0 else pd.DataFrame()
            
            batch = pd.concat([batch_base, batch_shift]).sample(frac=1.0, random_state=int(self.rng.integers(1e5)))
            stream.append(batch)
            
        return stream

    # ==========================================
    # CÁC KỊCH BẢN SHIFT (B1, B2, B4, B5, B6)
    # ==========================================

    def shift_b1_single_covariate(self, feature: str, severity: str, mode: str = 'abrupt', 
                                  num_batches: int = 10, batch_size: int = 1000) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B1: Single-feature covariate shift (Bóp méo 1 feature bằng Exponential Tilt)"""
        s_val = self.severity_map[severity]
        if feature not in self.features:
            raise ValueError(f"Unknown shift feature: {feature}")
        if not pd.api.types.is_numeric_dtype(self.df[feature]):
            raise TypeError(f"Covariate tilt requires a numeric feature, got: {feature}")
        x_val = self.df[feature].values
        
        # Chỉ xử lý numerical (nếu categorical, cần mapping riêng)
        norm_x = (x_val - np.mean(x_val)) / (np.std(x_val) + 1e-9)
        weights = np.exp(np.clip(s_val * norm_x, -50.0, 50.0))
        
        shifted_df = self._rejection_sampling(self.df, weights)
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)
        
        # Ground-truth: 100% lỗi do feature này
        gt = {f: 0.0 for f in self.features}
        gt[feature] = 1.0
        return stream, gt

    def shift_b2_correlated_multi(self, features: List[str], severity: str, mode: str = 'abrupt', 
                                  num_batches: int = 10, batch_size: int = 1000) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B2: Correlated multi-feature shift (Bóp méo đồng thời tổ hợp tuyến tính của các feature)"""
        s_val = self.severity_map[severity]
        if not features:
            raise ValueError("At least one feature is required")
        
        # Tính trọng số nghiêng dựa trên tổng chuẩn hóa của các feature
        combined_x = np.zeros(len(self.df))
        for f in features:
            if f not in self.features or not pd.api.types.is_numeric_dtype(self.df[f]):
                raise TypeError(f"Correlated tilt requires numeric feature: {f}")
            x_val = self.df[f].values
            combined_x += (x_val - np.mean(x_val)) / (np.std(x_val) + 1e-9)
            
        weights = np.exp(np.clip(s_val * combined_x / len(features), -50.0, 50.0))
        shifted_df = self._rejection_sampling(self.df, weights)
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)
        
        # Ground-truth: Chia đều trách nhiệm cho các feature bị can thiệp
        gt = {f: 0.0 for f in self.features}
        for f in features:
            gt[f] = 1.0 / len(features)
        return stream, gt

    def shift_b4_support_violation(self, feature: str, severity: str, mode: str = 'abrupt', 
                                   num_batches: int = 10, batch_size: int = 1000) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B4: Support violation (Đưa giá trị ra ngoài vùng dữ liệu đã thấy trong Source)"""
        shifted_df = self.df.copy()
        if feature not in self.features or not pd.api.types.is_numeric_dtype(shifted_df[feature]):
            raise TypeError(f"Support violation requires numeric feature: {feature}")
        
        # Bơm giá trị ngoại lai. Severity càng cao, giá trị càng xa Max của Source
        shift_multiplier = 1.2 if severity == 'low' else (1.5 if severity == 'medium' else 2.0)
        max_val = shifted_df[feature].max()
        std_val = shifted_df[feature].std()
        
        # Đẩy dữ liệu ra khỏi support
        shifted_df[feature] = shifted_df[feature] + shift_multiplier * (max_val + std_val)
        
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)
        
        gt = {f: 0.0 for f in self.features}
        gt[feature] = 1.0
        return stream, gt

    def shift_b5_pipeline_corruption(self, feature: str, severity: str, mode: str = 'abrupt', 
                                     num_batches: int = 10, batch_size: int = 1000) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B5: Data-pipeline corruption (Thiếu dữ liệu / Sensor hỏng)"""
        shifted_df = self.df.copy()
        if feature not in self.features or not pd.api.types.is_numeric_dtype(shifted_df[feature]):
            raise TypeError(f"Pipeline corruption requires numeric feature: {feature}")
        
        # Tỷ lệ corruption: Low 20%, Medium 50%, High 80%
        corrupt_rate = 0.2 if severity == 'low' else (0.5 if severity == 'medium' else 0.8)
        
        corrupt_mask = self.rng.random(len(shifted_df)) < corrupt_rate
        # Gán bằng -999 để mô phỏng lỗi pipeline không bắt được
        shifted_df.loc[corrupt_mask, feature] = -999 
        
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)
        
        gt = {f: 0.0 for f in self.features}
        gt[feature] = 1.0
        return stream, gt

    def shift_b6_concept_shift_negative_control(self, condition_func: Callable, severity: str, mode: str = 'abrupt', 
                                                num_batches: int = 10, batch_size: int = 1000) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """
        B6: Concept-shift negative control (Chỉ lật nhãn Y, giữ nguyên X).
        Đây là cái bẫy dành cho các thuật toán giám sát.
        """
        shifted_df = self.df.copy()
        
        # Flip probability: Low 20%, Medium 50%, High 100% (lật toàn bộ vùng thỏa điều kiện)
        flip_prob = 0.2 if severity == 'low' else (0.5 if severity == 'medium' else 1.0)
        
        mask = condition_func(shifted_df)
        flip_mask = mask & (self.rng.random(len(shifted_df)) < flip_prob)
        
        shifted_df.loc[flip_mask, self.target_col] = 1 - shifted_df.loc[flip_mask, self.target_col]
        
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)
        
        # Ground-truth CỰC KỲ QUAN TRỌNG: 0.0 cho tất cả
        # Vì X không đổi, nếu công cụ nào kết luận "Feature X gây lỗi" -> công cụ đó bị False Alarm
        gt = {f: 0.0 for f in self.features}
        return stream, gt
