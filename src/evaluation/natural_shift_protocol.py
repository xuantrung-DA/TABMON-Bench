"""Leakage-resistant partition helpers for natural-shift validation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split


MONITOR_OUTPUT_COLUMNS = (
    "dataset",
    "model",
    "domain_id",
    "domain_type",
    "method",
    "endpoint",
    "target_rows",
    "source_value",
    "estimated_value",
    "estimated_excess_value",
    "domain_classifier_auc",
    "effective_sample_size_fraction",
    "density_ratio_clipping_rate",
    "prediction_entropy_shift",
    "prediction_confidence_shift",
    "novel_support_rate",
    "novel_row_rate",
    "schema_guard_score",
    "schema_guard_alarm",
    "schema_guard_p_value",
)

FORBIDDEN_TARGET_MONITOR_TOKENS = (
    "target_label",
    "true_",
    "oracle",
    "failure",
    "ground_truth",
)


def select_largest_groups(groups: Iterable[object], count: int) -> list[object]:
    """Select sufficiently large domains without consulting outcomes."""
    values, counts = np.unique(np.asarray(list(groups)), return_counts=True)
    if count <= 0 or count >= len(values):
        raise ValueError("count must leave at least one source group")
    order = sorted(range(len(values)), key=lambda i: (-int(counts[i]), str(values[i])))
    return [values[index] for index in order[:count]]


def temporal_source_and_windows(
    n_rows: int, target_fraction: float, window_count: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Keep an early source period and split the late period contiguously."""
    if n_rows <= 0 or not 0.0 < target_fraction < 1.0 or window_count < 2:
        raise ValueError("Invalid temporal split configuration")
    target_start = int(np.floor(n_rows * (1.0 - target_fraction)))
    source = np.arange(target_start, dtype=np.int64)
    late = np.arange(target_start, n_rows, dtype=np.int64)
    windows = [part for part in np.array_split(late, window_count) if len(part)]
    if len(windows) != window_count:
        raise ValueError("Not enough target rows for requested temporal windows")
    return source, windows


def split_source_indices(
    source_positions: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 60/20/20 train/calibration/internal-holdout source splits."""
    positions = np.asarray(source_positions, dtype=np.int64)
    labels = np.asarray(labels)
    if len(positions) != len(labels):
        raise ValueError("Source positions and labels are not aligned")
    train, remainder = train_test_split(
        positions,
        test_size=0.4,
        random_state=random_seed,
        stratify=labels,
    )
    remainder_labels = labels[np.searchsorted(positions, remainder)] if np.all(
        positions[:-1] <= positions[1:]
    ) else None
    if remainder_labels is None:
        mapping = {int(position): value for position, value in zip(positions, labels)}
        remainder_labels = np.asarray([mapping[int(position)] for position in remainder])
    calibration, holdout = train_test_split(
        remainder,
        test_size=0.5,
        random_state=random_seed + 1,
        stratify=remainder_labels,
    )
    return np.sort(train), np.sort(calibration), np.sort(holdout)


def assert_disjoint_complete(
    universe: np.ndarray, partitions: Iterable[np.ndarray]
) -> None:
    universe = np.asarray(universe, dtype=np.int64)
    parts = [np.asarray(part, dtype=np.int64) for part in partitions]
    combined = np.concatenate(parts)
    if len(combined) != len(np.unique(combined)):
        raise ValueError("Natural-shift partitions overlap")
    if set(combined.tolist()) != set(universe.tolist()):
        raise ValueError("Natural-shift partitions do not cover the universe")


def validate_monitor_output_columns(columns: Iterable[str]) -> None:
    columns = tuple(columns)
    if columns != MONITOR_OUTPUT_COLUMNS:
        raise ValueError("Natural monitor output differs from its explicit allowlist")
    leaked = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_TARGET_MONITOR_TOKENS)
    ]
    if leaked:
        raise ValueError(f"Target-oracle fields reached monitor output: {leaked}")

