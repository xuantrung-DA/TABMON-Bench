from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score


def compute_risk_metrics(
    true_risks: np.ndarray, pred_risks: np.ndarray
) -> Dict[str, float]:
    """Score estimated risk changes against the offline oracle values."""

    mae = mean_absolute_error(true_risks, pred_risks)
    rmse = np.sqrt(mean_squared_error(true_risks, pred_risks))

    # Correlations are undefined for constant arrays (for example, null streams).
    if len(np.unique(true_risks)) > 1 and len(np.unique(pred_risks)) > 1:
        pearson_corr, _ = pearsonr(true_risks, pred_risks)
        spearman_corr, _ = spearmanr(true_risks, pred_risks)
    else:
        pearson_corr, spearman_corr = 0.0, 0.0

    return {
        "risk_mae": mae,
        "risk_rmse": rmse,
        "risk_pearson": pearson_corr,
        "risk_spearman": spearman_corr,
    }


def compute_attribution_metrics(
    true_attr: Dict[str, float], pred_attr: Dict[str, float], k: int = 3
) -> Dict[str, float]:
    """Score a feature ranking against controlled intervention targets."""

    features = list(true_attr.keys())
    if not features:
        raise ValueError("true_attr must contain at least one feature")
    if k < 1:
        raise ValueError("k must be greater than zero")

    y_true = np.array([true_attr[f] for f in features], dtype=float)
    y_pred = np.array([pred_attr.get(f, 0.0) for f in features], dtype=float)
    true_magnitude = np.abs(y_true)
    pred_magnitude = np.abs(y_pred)
    ranking_k = min(k, len(features))

    has_ground_truth = np.any(true_magnitude > 0)
    ndcg_k = (
        ndcg_score([true_magnitude], [pred_magnitude], k=ranking_k)
        if has_ground_truth
        else float("nan")
    )

    active_features = set(np.array(features)[true_magnitude > 0])
    if active_features:
        predicted_top_k = set(
            np.array(features)[np.argsort(pred_magnitude)[-ranking_k:]]
        )
        top_k_recall = len(active_features & predicted_top_k) / len(active_features)
    else:
        top_k_recall = float("nan")

    active_features_mask = y_true != 0
    if np.any(active_features_mask):
        sign_acc = np.mean(
            np.sign(y_true[active_features_mask])
            == np.sign(y_pred[active_features_mask])
        )
    else:
        sign_acc = float("nan")

    return {
        f"ndcg@{k}": ndcg_k,
        f"top{k}_recall": top_k_recall,
        "sign_accuracy": sign_acc,
        "false_attribution_mass": (
            float(np.sum(np.abs(y_pred))) if not has_ground_truth else 0.0
        ),
    }


def compute_sequential_metrics(
    alarms: List[bool], shift_point: int
) -> Dict[str, float]:
    """Score an alarm trajectory against a known shift point."""

    alarms = np.array(alarms)

    pre_shift_alarms = alarms[:shift_point]
    far = np.mean(pre_shift_alarms) if len(pre_shift_alarms) > 0 else 0.0

    post_shift_alarms = alarms[shift_point:]
    power = np.mean(post_shift_alarms) if len(post_shift_alarms) > 0 else 0.0

    if len(post_shift_alarms) > 0 and np.any(post_shift_alarms):
        delay = np.argmax(post_shift_alarms)
    else:
        delay = float("inf")

    return {
        "false_alarm_rate": far,
        "power": power,
        "detection_delay": delay,
    }


def compute_event_based_sequential_metrics(
    alarms: List[bool],
    true_risks: np.ndarray,
    risk_event_threshold: float,
) -> Dict[str, float]:
    """Score alarms against the first oracle risk event.

    Labels are used only here, after monitoring, to establish the offline event
    time. Alarm generation itself remains label-free.
    """
    alarm_array = np.asarray(alarms, dtype=bool)
    risk_array = np.asarray(true_risks, dtype=float)
    event_indices = np.flatnonzero(risk_array > risk_event_threshold)

    if not len(event_indices):
        return {
            "true_risk_event_batch": float("nan"),
            "first_alarm_batch": (
                float(np.flatnonzero(alarm_array)[0])
                if np.any(alarm_array)
                else float("nan")
            ),
            "false_alarm_rate": float(np.mean(alarm_array)),
            "power": float("nan"),
            "detection_delay": float("nan"),
            "missed_alarm": 0.0,
        }

    event_batch = int(event_indices[0])
    pre_event = alarm_array[:event_batch]
    post_event = alarm_array[event_batch:]
    post_alarm_indices = np.flatnonzero(post_event)
    first_alarm_indices = np.flatnonzero(alarm_array)
    missed = not len(post_alarm_indices)
    return {
        "true_risk_event_batch": float(event_batch),
        "first_alarm_batch": (
            float(first_alarm_indices[0])
            if len(first_alarm_indices)
            else float("nan")
        ),
        "false_alarm_rate": (
            float(np.mean(pre_event)) if len(pre_event) else 0.0
        ),
        "power": float(np.mean(post_event)),
        "detection_delay": (
            float(post_alarm_indices[0]) if not missed else float("nan")
        ),
        "missed_alarm": float(missed),
    }


def compute_alarm_event_metrics(
    alarms: List[bool], event_flags: List[bool] | np.ndarray
) -> Dict[str, float]:
    """Score an alarm trajectory against an explicit binary event trajectory.

    This target-agnostic form is used by schema v6 so a risk alarm can be
    evaluated against an oracle risk event while a drift alarm is evaluated
    against the onset of the controlled distribution shift.
    """
    alarm_array = np.asarray(alarms, dtype=bool)
    event_array = np.asarray(event_flags, dtype=bool)
    if alarm_array.ndim != 1 or event_array.ndim != 1:
        raise ValueError("alarms and event_flags must be one-dimensional")
    if len(alarm_array) != len(event_array):
        raise ValueError("alarms and event_flags must have equal length")

    event_indices = np.flatnonzero(event_array)
    first_alarm_indices = np.flatnonzero(alarm_array)
    if not len(event_indices):
        return {
            "event_batch": float("nan"),
            "first_alarm_batch": (
                float(first_alarm_indices[0])
                if len(first_alarm_indices)
                else float("nan")
            ),
            "false_alarm_rate": (
                float(np.mean(alarm_array)) if len(alarm_array) else 0.0
            ),
            "power": float("nan"),
            "detection_delay": float("nan"),
            "missed_alarm": 0.0,
        }

    event_batch = int(event_indices[0])
    pre_event = alarm_array[:event_batch]
    post_event = alarm_array[event_batch:]
    post_alarm_indices = np.flatnonzero(post_event)
    missed = not len(post_alarm_indices)
    return {
        "event_batch": float(event_batch),
        "first_alarm_batch": (
            float(first_alarm_indices[0])
            if len(first_alarm_indices)
            else float("nan")
        ),
        "false_alarm_rate": float(np.mean(pre_event)) if len(pre_event) else 0.0,
        "power": float(np.mean(post_event)),
        "detection_delay": (
            float(post_alarm_indices[0]) if not missed else float("nan")
        ),
        "missed_alarm": float(missed),
    }


def compute_reliability_metrics(
    true_val: float, ci_lower: float, ci_upper: float
) -> Dict[str, float]:
    """Return empirical interval coverage and width for one target value."""

    is_covered = int(ci_lower <= true_val <= ci_upper)
    width = ci_upper - ci_lower

    return {"coverage": float(is_covered), "interval_width": width}
