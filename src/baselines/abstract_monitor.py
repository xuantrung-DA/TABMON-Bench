from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class AbstractMonitor(ABC):
    def __init__(self, reference_X: pd.DataFrame, reference_y: pd.Series, model: Any):
        """
        Khởi tạo Monitor với dữ liệu quá khứ (Reference/Source) và mô hình đã train.
        """
        self.reference_X = reference_X.copy()
        self.reference_y = reference_y.copy()
        self.model = model
        self.features = list(reference_X.columns)

    @abstractmethod
    def analyze_batch(self, target_X: pd.DataFrame, target_y: pd.Series = None) -> Dict[str, Any]:
        """
        Đánh giá một batch dữ liệu từ luồng (Stream).
        Tham số target_y chỉ được dùng BỞI ORACLE (B6), các label-free monitors KHÔNG ĐƯỢC chạm vào.
        
        Output bắt buộc phải chứa:
        - 'estimated_risk_change': (float) Giá trị ước lượng Delta Loss.
        - 'feature_attribution': (Dict[str, float]) Phần trăm đóng góp của mỗi feature vào lỗi.
        """
        pass