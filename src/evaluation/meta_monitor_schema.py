"""Strict, auditable predictor schema for the reliability meta-monitor.

The meta-monitor is trained offline against ``monitor_failure``.  Its
predictors, however, must remain available at deployment time without target
labels.  This module implements a default-deny boundary: a column can be used
only when it is explicitly registered below with observable-only provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MetaMonitorFeature:
    name: str
    provenance: str
    uses_target_labels: bool = False
    uses_oracle_risk: bool = False
    uses_failure_labels: bool = False


_FEATURES = (
    MetaMonitorFeature("domain_classifier_auc", "reference_X + target_X"),
    MetaMonitorFeature(
        "effective_sample_size_fraction", "reference_X + target_X"
    ),
    MetaMonitorFeature("density_ratio_clipping_rate", "reference_X + target_X"),
    MetaMonitorFeature(
        "prediction_entropy_shift", "reference_X + target_X + model outputs"
    ),
    MetaMonitorFeature(
        "prediction_confidence_shift",
        "reference_X + target_X + model outputs",
    ),
    MetaMonitorFeature("novel_support_rate", "reference_X + target_X"),
    MetaMonitorFeature("novel_row_rate", "reference_X + target_X"),
    MetaMonitorFeature(
        "attribution_abs_mass", "reference_X + target_X + model outputs"
    ),
    MetaMonitorFeature(
        "attribution_max_abs", "reference_X + target_X + model outputs"
    ),
    MetaMonitorFeature(
        "attribution_l1_instability",
        "current and previous label-free monitor outputs",
    ),
    MetaMonitorFeature(
        "attribution_topk_jaccard",
        "current and previous label-free monitor outputs",
    ),
    MetaMonitorFeature("monitor_score", "label-free monitor output"),
    MetaMonitorFeature("alarm_p_value", "label-free monitor output"),
)

META_MONITOR_FEATURES = tuple(feature.name for feature in _FEATURES)
META_MONITOR_TARGET = "monitor_failure"

# These groups are documented in the generated audit record.  The default-deny
# allowlist is the actual enforcement mechanism, so aliases and future oracle
# columns are rejected even when they are absent from this illustrative list.
FORBIDDEN_COLUMN_CLASSES = {
    "target_labels": [
        "target_label",
        "y_true",
        "income",
        "PINCP",
        "Cover_Type",
        "readmitted_30d",
    ],
    "oracle_risk": [
        "true_excess_risk",
        "true_risk_event",
        "oracle_risk",
        "risk_abs_error",
        "coverage",
    ],
    "failure_labels": [
        "risk_failure",
        "attribution_failure",
        "alarm_failure",
        "monitor_failure",
    ],
    "target_derived_diagnostics": [
        "sign_accuracy",
        "ground_truth_attribution_json",
        "alarm_target_event",
        "early_warning",
    ],
}


def validate_meta_monitor_features(requested: Iterable[str]) -> tuple[str, ...]:
    """Return a validated feature tuple or reject any non-allowlisted column."""
    columns = tuple(requested)
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate meta-monitor features: {duplicates}")
    rejected = sorted(set(columns) - set(META_MONITOR_FEATURES))
    if rejected:
        raise ValueError(
            "Meta-monitor feature allowlist rejected columns: " f"{rejected}"
        )
    registry = {feature.name: feature for feature in _FEATURES}
    unsafe = [
        name
        for name in columns
        if registry[name].uses_target_labels
        or registry[name].uses_oracle_risk
        or registry[name].uses_failure_labels
    ]
    if unsafe:
        raise RuntimeError(f"Unsafe meta-monitor feature provenance: {unsafe}")
    return columns


def available_meta_monitor_features(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select present predictors exclusively through the canonical allowlist."""
    return validate_meta_monitor_features(
        name for name in META_MONITOR_FEATURES if name in frame.columns
    )


def meta_monitor_predictors(
    frame: pd.DataFrame, requested: Iterable[str] | None = None
) -> pd.DataFrame:
    """Build the predictor matrix without exposing the offline failure label."""
    columns = (
        available_meta_monitor_features(frame)
        if requested is None
        else validate_meta_monitor_features(requested)
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing allowlisted meta-monitor features: {missing}")
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def meta_monitor_failure_target(frame: pd.DataFrame) -> pd.Series:
    """Return the offline evaluation target through a separate code path."""
    if META_MONITOR_TARGET not in frame.columns:
        raise ValueError(f"Missing meta-monitor target: {META_MONITOR_TARGET}")
    return frame[META_MONITOR_TARGET].astype(int)


def meta_monitor_allowlist_audit() -> dict[str, object]:
    """Return a serializable proof record for manifests and audit reports."""
    validate_meta_monitor_features(META_MONITOR_FEATURES)
    return {
        "policy": "default-deny explicit allowlist",
        "offline_training_target": META_MONITOR_TARGET,
        "target_is_separate_from_predictors": (
            META_MONITOR_TARGET not in META_MONITOR_FEATURES
        ),
        "features": [asdict(feature) for feature in _FEATURES],
        "forbidden_column_classes": FORBIDDEN_COLUMN_CLASSES,
        "guarantees": {
            "target_labels_are_predictors": False,
            "oracle_risk_is_a_predictor": False,
            "failure_labels_are_predictors": False,
            "target_derived_diagnostics_are_predictors": False,
        },
    }
