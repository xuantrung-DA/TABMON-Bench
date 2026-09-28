"""Canonical deterministic stream construction shared by runs and caches."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from src.shift_generator import TabularShiftGenerator


class StreamConfiguration(Protocol):
    shift: str
    severity: str
    mode: str
    num_batches: int
    batch_size: int


def generate_stream(
    generator: TabularShiftGenerator,
    scenario: StreamConfiguration,
    dataset_config: dict[str, Any],
) -> tuple[list[pd.DataFrame], dict[str, float]]:
    """Generate one stream using the exact benchmark shift definitions."""
    common = {
        "severity": scenario.severity,
        "mode": scenario.mode,
        "num_batches": scenario.num_batches,
        "batch_size": scenario.batch_size,
    }
    feature = dataset_config["shift_feature"]
    if scenario.shift == "no_shift":
        return generator.shift_b0_no_shift(
            num_batches=scenario.num_batches,
            batch_size=scenario.batch_size,
        )
    if scenario.shift == "covariate":
        return generator.shift_b1_single_covariate(feature=feature, **common)
    if scenario.shift == "correlated":
        return generator.shift_b2_correlated_multi(
            features=dataset_config["corr_features"], **common
        )
    if scenario.shift == "support":
        return generator.shift_b4_support_violation(feature=feature, **common)
    if scenario.shift == "pipeline":
        return generator.shift_b5_pipeline_corruption(feature=feature, **common)
    if scenario.shift == "concept":
        condition = lambda frame: frame[feature] > frame[feature].mean()
        return generator.shift_b6_concept_shift_negative_control(
            condition_func=condition, **common
        )
    raise ValueError(f"Unsupported shift family: {scenario.shift}")


def shift_fraction(mode: str, batch_index: int, num_batches: int) -> float:
    if mode == "abrupt":
        return 0.0 if batch_index < num_batches // 2 else 1.0
    if mode == "gradual":
        return batch_index / (num_batches - 1) if num_batches > 1 else 1.0
    return 1.0
