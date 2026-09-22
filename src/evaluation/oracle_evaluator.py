from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss


class OracleEvaluator:
    """Offline evaluator and the only component allowed to read target labels.

    Risk is binary log loss. Positive excess risk means the monitored model is
    performing worse on the target batch than on the reference data.
    """

    def __init__(self, model: Any, reference_df: pd.DataFrame, target_col: str):
        if target_col not in reference_df.columns:
            raise ValueError(f"Target column '{target_col}' is missing from reference_df")
        if reference_df.empty:
            raise ValueError("reference_df must not be empty")

        self.model = model
        self.target_col = target_col
        self.feature_columns = [c for c in reference_df.columns if c != target_col]
        self.reference_risk = self._calculate_log_loss(reference_df)

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_col not in df.columns:
            raise ValueError(
                f"Oracle requires hidden target column '{self.target_col}' for offline scoring"
            )
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Batch is missing model features: {sorted(missing)}")
        return df.loc[:, self.feature_columns], df[self.target_col]

    def _calculate_log_loss(self, df: pd.DataFrame) -> float:
        X, y = self._split(df)
        probabilities = np.asarray(self.model.predict_proba(X))[:, 1]
        probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y, probabilities, labels=[0, 1]))

    def calculate_true_excess_risk(self, target_batch: pd.DataFrame) -> float:
        """Return target log loss minus reference log loss."""
        if target_batch.empty:
            raise ValueError("target_batch must not be empty")
        return self._calculate_log_loss(target_batch) - self.reference_risk

    def evaluate_attribution(
        self,
        predicted: Dict[str, float],
        ground_truth: Dict[str, float],
    ) -> Dict[str, float]:
        """Compatibility helper used by the phase-4 smoke test."""
        features = list(ground_truth)
        if not features:
            raise ValueError("ground_truth attribution must not be empty")

        true_values = np.asarray([ground_truth[f] for f in features], dtype=float)
        pred_values = np.asarray([predicted.get(f, 0.0) for f in features], dtype=float)

        if np.allclose(true_values, 0.0):
            return {
                "top1_accuracy": float(np.allclose(pred_values, 0.0)),
                "rank_correlation": float("nan"),
            }

        top1_accuracy = float(np.argmax(np.abs(true_values)) == np.argmax(np.abs(pred_values)))
        if np.unique(true_values).size < 2 or np.unique(pred_values).size < 2:
            rank_correlation = 0.0
        else:
            rank_correlation = float(spearmanr(true_values, pred_values).statistic)

        return {
            "top1_accuracy": top1_accuracy,
            "rank_correlation": rank_correlation,
        }
