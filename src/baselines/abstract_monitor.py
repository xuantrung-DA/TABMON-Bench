from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class AbstractMonitor(ABC):
    """Base class for monitors that never receive target labels at inference."""

    def __init__(
        self,
        reference_X: pd.DataFrame,
        reference_y: pd.Series,
        model: Any,
    ):
        self.reference_X = reference_X.copy()
        self.reference_y = reference_y.copy()
        self.model = model
        self.features = list(reference_X.columns)

    @abstractmethod
    def analyze_batch(self, target_X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze one unlabeled target batch and return declared outputs."""

        raise NotImplementedError
