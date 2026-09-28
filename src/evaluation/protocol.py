"""Locked evaluation protocol for TABMON-Bench monitors."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Dict, Tuple

import pandas as pd

from .metrics import compute_attribution_metrics, compute_reliability_metrics


class TABMONProtocol:
    """Evaluate monitor reports while keeping target labels outside monitors."""

    def __init__(self, k_features: int = 3):
        self.k_features = k_features

    def profile_monitor(
        self, monitor_func: Callable, target_X: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Run a label-free monitor and measure wall time and Python peak memory."""
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            report = monitor_func(target_X)
            if not isinstance(report, dict):
                raise TypeError("Monitor output must be a dictionary")
        finally:
            end_time = time.perf_counter()
            _, peak_ram = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        efficiency_metrics = {
            "runtime_seconds": end_time - start_time,
            "peak_ram_mb": peak_ram / (1024 * 1024),
        }
        return report, efficiency_metrics

    def evaluate_batch(
        self,
        monitor_report: Dict[str, Any],
        ground_truth_attr: Dict[str, float],
        true_delta_loss: float,
    ) -> Dict[str, float]:
        """Score one report against offline oracle risk and attribution targets."""
        results: Dict[str, float] = {}
        capabilities = monitor_report.get("capabilities", {})

        pred_risk = monitor_report.get("estimated_risk_change")
        supports_risk = bool(
            capabilities.get("risk_estimation", pred_risk is not None)
        )
        if supports_risk and pred_risk is not None:
            results["risk_abs_error"] = abs(true_delta_loss - pred_risk)
            ci_lower = monitor_report.get(
                "estimated_risk_ci_lower", monitor_report.get("risk_ci_lower")
            )
            ci_upper = monitor_report.get(
                "estimated_risk_ci_upper", monitor_report.get("risk_ci_upper")
            )
            if ci_lower is not None and ci_upper is not None:
                results.update(
                    compute_reliability_metrics(true_delta_loss, ci_lower, ci_upper)
                )

        supports_attribution = bool(
            capabilities.get(
                "attribution", "feature_attribution" in monitor_report
            )
        )
        if not supports_attribution:
            return results

        pred_attr = monitor_report.get("feature_attribution", {})
        attribution_results = compute_attribution_metrics(
            ground_truth_attr, pred_attr, k=self.k_features
        )
        # Ranking metrics use normalized attribution, but false explanations
        # must be measured in raw units. Otherwise any non-zero normalized
        # vector has mass one, even when its magnitude is negligible.
        attribution_results["false_attribution_mass_normalized"] = (
            attribution_results["false_attribution_mass"]
        )
        has_attribution_target = any(
            abs(float(value)) > 0.0 for value in ground_truth_attr.values()
        )
        raw_attribution = monitor_report.get("raw_feature_attribution")
        if not has_attribution_target and raw_attribution is not None:
            attribution_results["false_attribution_mass"] = float(
                sum(abs(float(value)) for value in raw_attribution.values())
            )
        results.update(attribution_results)
        return results
