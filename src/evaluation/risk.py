"""Shared risk definitions used by online experiments and cached rescoring."""

from __future__ import annotations

from typing import Any

import numpy as np


# Frozen in schema v7.  Keeping this constant in one module prevents an oracle
# cache from silently changing the estimand when probabilities become extreme.
BINARY_LOG_LOSS_EPSILON = 1e-7


def binary_log_losses(
    probabilities: np.ndarray,
    classes: np.ndarray | list[Any],
    labels: np.ndarray | list[Any],
    *,
    epsilon: float = BINARY_LOG_LOSS_EPSILON,
) -> np.ndarray:
    """Return per-sample binary log loss under the frozen clipping rule."""

    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability matrix does not match declared classes")
    if len(classes) != 2:
        raise ValueError("The frozen TABMON risk endpoint is binary log loss")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities have unequal length")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie between zero and 0.5")
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    try:
        positions = np.asarray([class_to_index[value] for value in labels.tolist()])
    except KeyError as exc:
        raise ValueError(f"Unknown target class: {exc.args[0]}") from exc
    selected = probabilities[np.arange(len(probabilities)), positions]
    return -np.log(np.clip(selected, epsilon, 1.0 - epsilon))


def multiclass_log_losses(
    probabilities: np.ndarray,
    classes: np.ndarray | list[Any],
    labels: np.ndarray | list[Any],
    *,
    epsilon: float = BINARY_LOG_LOSS_EPSILON,
) -> np.ndarray:
    """Return per-sample log loss for two or more declared classes.

    This endpoint is used only by the Step-12 multiclass extension.  The
    frozen binary endpoint above remains unchanged so earlier schemas are
    bit-for-bit reproducible.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability matrix does not match declared classes")
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities have unequal length")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie between zero and 0.5")
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    try:
        positions = np.asarray([class_to_index[value] for value in labels.tolist()])
    except KeyError as exc:
        raise ValueError(f"Unknown target class: {exc.args[0]}") from exc
    normalized = probabilities / np.clip(
        probabilities.sum(axis=1, keepdims=True), epsilon, None
    )
    selected = normalized[np.arange(len(normalized)), positions]
    return -np.log(np.clip(selected, epsilon, 1.0))
