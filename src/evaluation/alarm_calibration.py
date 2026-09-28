"""Finite-sample upper-tail alarm calibration utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def split_conformal_threshold(
    calibration_scores: Iterable[float], alpha: float
) -> tuple[float, int, int]:
    """Return a strict upper-tail split-conformal order statistic."""

    values = np.sort(np.asarray(list(calibration_scores), dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a non-empty finite vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie between zero and one")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    threshold = float("inf") if rank > len(values) else float(values[rank - 1])
    return threshold, rank, len(values)


def stream_max_threshold(
    batch_scores: np.ndarray,
    num_trajectories: int,
    num_batches: int,
    alpha: float,
) -> tuple[float, np.ndarray, int]:
    """Calibrate a family-wise threshold from null-trajectory maxima."""

    scores = np.asarray(batch_scores, dtype=float)
    expected = int(num_trajectories) * int(num_batches)
    if scores.ndim != 1 or len(scores) != expected:
        raise ValueError(f"Expected {expected} batch scores; found {scores.shape}")
    maxima = scores.reshape(num_trajectories, num_batches).max(axis=1)
    threshold, rank, _ = split_conformal_threshold(maxima, alpha)
    return threshold, maxima, rank
