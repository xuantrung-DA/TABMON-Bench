"""Cross-fitted null-control recalibration of TABMON sequential alarms.

The script reuses schema-v7 monitor scores.  For every held-out seed, alarm
thresholds are fitted exclusively from no-shift batches belonging to the other
seeds.  The primary policy pools models within each dataset and monitor after
normalizing scores by the original reference-calibration scale.  A global
monitor-level policy is retained as a sensitivity analysis.

Oracle event flags are joined only after all recalibrated alarms have been
computed.  Target labels, oracle risks, and failure labels are never calibration
features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache.stream_cache import atomic_json, atomic_parquet
from src.evaluation.alarm_calibration import split_conformal_threshold
from src.evaluation.metrics import compute_alarm_event_metrics


RESULT_SCHEMA_VERSION = 1
PRIMARY_POLICY = "dataset_monitor"
POLICY_GROUPS = {
    "global_monitor": ["monitor"],
    PRIMARY_POLICY: ["dataset", "monitor"],
}
DEFAULT_ALPHAS = (0.01, 0.02, 0.05)
MONITOR_SCORE_COLUMNS = [
    "scenario_id",
    "dataset",
    "model",
    "monitor",
    "shift",
    "severity",
    "mode",
    "seed",
    "batch_index",
    "monitor_score",
    "null_score_median",
    "alarm_threshold",
    "alarm",
]
OFFLINE_EVENT_COLUMNS = ["scenario_id", "batch_index", "alarm_target_event"]
FORBIDDEN_CALIBRATION_COLUMNS = {
    "target_label",
    "y_true",
    "true_excess_risk",
    "true_risk_event",
    "alarm_target_event",
    "risk_failure",
    "attribution_failure",
    "alarm_failure",
    "monitor_failure",
    "ground_truth_attribution_json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS)
    )
    args = parser.parse_args(argv)
    if not args.alphas or any(not 0.0 < value < 0.5 for value in args.alphas):
        parser.error("alphas must lie strictly between 0 and 0.5")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("alphas must be unique")
    return args


def _validate_v7(source: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    required = [
        "batch_metrics.parquet",
        "aggregate_metrics.parquet",
        "benchmark_manifest.json",
        "benchmark_status.json",
    ]
    for filename in required:
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    manifest = json.loads((source / "benchmark_manifest.json").read_text("utf-8"))
    status = json.loads((source / "benchmark_status.json").read_text("utf-8"))
    if int(manifest.get("schema_version", -1)) != 7:
        raise ValueError("Step 7 requires a schema-v7 benchmark output")
    if not status.get("complete"):
        raise ValueError("Step 7 requires a complete schema-v7 run")
    batches = pd.read_parquet(source / "batch_metrics.parquet")
    missing = sorted(
        (set(MONITOR_SCORE_COLUMNS) | set(OFFLINE_EVENT_COLUMNS))
        - set(batches.columns)
    )
    if missing:
        raise ValueError(f"Schema-v7 batch table is missing columns: {missing}")
    if len(batches) != 62_000 or batches["scenario_id"].nunique() != 6_200:
        raise ValueError("Step 7 expects the complete 62,000-row schema-v7 grid")
    return manifest, batches


def reference_normalized_score(frame: pd.DataFrame) -> pd.Series:
    """Map pair-specific raw scores to the original reference-null scale."""

    denominator = pd.to_numeric(
        frame["alarm_threshold"] - frame["null_score_median"], errors="raise"
    )
    if denominator.isna().any() or (denominator <= 0).any():
        raise ValueError("Alarm threshold must exceed the null-score median")
    score = (
        pd.to_numeric(frame["monitor_score"], errors="raise")
        - pd.to_numeric(frame["null_score_median"], errors="raise")
    ) / denominator
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite reference-normalized alarm score")
    return score.astype(float)


def _calibration_frame(batches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monitor_frame = batches[MONITOR_SCORE_COLUMNS].copy()
    if set(monitor_frame.columns) & FORBIDDEN_CALIBRATION_COLUMNS:
        raise RuntimeError("Offline oracle field reached alarm calibration frame")
    monitor_frame["normalized_score"] = reference_normalized_score(monitor_frame)
    events = batches[OFFLINE_EVENT_COLUMNS].copy()
    if events.duplicated(["scenario_id", "batch_index"]).any():
        raise ValueError("Duplicate offline event keys")
    return monitor_frame, events


def _fit_cross_fitted_thresholds(
    monitor_frame: pd.DataFrame,
    alphas: list[float],
) -> pd.DataFrame:
    null = monitor_frame[monitor_frame["shift"] == "no_shift"].copy()
    seeds = sorted(monitor_frame["seed"].unique())
    records: list[dict[str, Any]] = []
    for policy, group_columns in POLICY_GROUPS.items():
        groups = null[group_columns].drop_duplicates().sort_values(group_columns)
        for held_out_seed in seeds:
            training = null[null["seed"] != held_out_seed]
            for group in groups.itertuples(index=False, name=None):
                group_values = dict(zip(group_columns, group))
                mask = pd.Series(True, index=training.index)
                for column, value in group_values.items():
                    mask &= training[column].eq(value)
                calibration_scores = training.loc[mask, "normalized_score"]
                calibration_streams = training.loc[mask, "scenario_id"].nunique()
                calibration_seeds = [
                    int(value)
                    for value in sorted(training.loc[mask, "seed"].unique())
                ]
                if held_out_seed in calibration_seeds:
                    raise RuntimeError("Held-out seed leaked into alarm calibration")
                for alpha in alphas:
                    threshold, rank, count = split_conformal_threshold(
                        calibration_scores, alpha
                    )
                    records.append(
                        {
                            "policy": policy,
                            "held_out_seed": int(held_out_seed),
                            "alpha": float(alpha),
                            **group_values,
                            "normalized_threshold": threshold,
                            "conformal_rank": rank,
                            "calibration_batches": count,
                            "calibration_streams": calibration_streams,
                            "calibration_seed_count": len(calibration_seeds),
                            "calibration_seeds_json": json.dumps(calibration_seeds),
                        }
                    )
    return pd.DataFrame(records)


def _apply_recalibration(
    monitor_frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    alphas: list[float],
) -> pd.DataFrame:
    identity = [
        "scenario_id",
        "dataset",
        "model",
        "monitor",
        "shift",
        "severity",
        "mode",
        "seed",
        "batch_index",
        "normalized_score",
    ]
    outputs = []

    original = monitor_frame[identity + ["alarm"]].copy()
    original["policy"] = "original_v7"
    original["alpha"] = 0.01
    original["normalized_threshold"] = 1.0
    original = original.rename(columns={"alarm": "recalibrated_alarm"})
    outputs.append(original)

    for policy, group_columns in POLICY_GROUPS.items():
        policy_thresholds = thresholds[thresholds["policy"] == policy]
        join_left = ["seed", *group_columns]
        join_right = ["held_out_seed", *group_columns]
        for alpha in alphas:
            selected = policy_thresholds[np.isclose(policy_thresholds["alpha"], alpha)]
            applied = monitor_frame[identity].merge(
                selected[
                    join_right
                    + ["normalized_threshold", "calibration_batches"]
                ],
                left_on=join_left,
                right_on=join_right,
                how="left",
                validate="many_to_one",
            )
            if applied["normalized_threshold"].isna().any():
                raise ValueError(f"Missing threshold for policy {policy}")
            applied["recalibrated_alarm"] = (
                applied["normalized_score"] > applied["normalized_threshold"]
            )
            applied["policy"] = policy
            applied["alpha"] = float(alpha)
            applied = applied.drop(columns=["held_out_seed", "calibration_batches"])
            outputs.append(applied)

    result = pd.concat(outputs, ignore_index=True)
    result["recalibrated_alarm"] = result["recalibrated_alarm"].astype(bool)
    return result


def _score_scenarios(
    alarms: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Alarm generation is complete before this offline join.
    evaluated = alarms.merge(
        events,
        on=["scenario_id", "batch_index"],
        how="inner",
        validate="many_to_one",
    )
    if len(evaluated) != len(alarms):
        raise ValueError("Alarm/event join is incomplete")
    group_columns = [
        "scenario_id",
        "dataset",
        "model",
        "monitor",
        "shift",
        "severity",
        "mode",
        "seed",
        "policy",
        "alpha",
    ]
    records = []
    for key, group in evaluated.groupby(group_columns, sort=True, observed=True):
        group = group.sort_values("batch_index")
        alarm_values = group["recalibrated_alarm"].astype(bool).to_numpy()
        event_values = group["alarm_target_event"].astype(bool).to_numpy()
        metrics = compute_alarm_event_metrics(alarm_values, event_values)
        event_indices = np.flatnonzero(event_values)
        if len(event_indices):
            event_batch = int(event_indices[0])
            pre_event_any = bool(alarm_values[:event_batch].any())
            post_event_any = bool(alarm_values[event_batch:].any())
        else:
            pre_event_any = bool(alarm_values.any())
            post_event_any = False
        records.append(
            {
                **dict(zip(group_columns, key)),
                **metrics,
                "event_exists": bool(len(event_indices)),
                "pre_event_any_alarm": pre_event_any,
                "detected_event": post_event_any,
                "batch_alarm_rate": float(alarm_values.mean()),
                "evaluated_batches": len(group),
            }
        )
    return evaluated, pd.DataFrame(records)


def _null_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    null = scenarios[scenarios["shift"] == "no_shift"]
    return (
        null.groupby(
            ["policy", "alpha", "monitor", "dataset"],
            sort=True,
            observed=True,
        )
        .agg(
            null_streams=("scenario_id", "size"),
            null_batch_alarm_rate=("batch_alarm_rate", "mean"),
            null_stream_any_alarm_rate=("pre_event_any_alarm", "mean"),
        )
        .reset_index()
    )


def _detection_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    shifted = scenarios[scenarios["event_exists"]].copy()
    shifted["conditional_delay"] = shifted["detection_delay"].where(
        shifted["detected_event"]
    )
    return (
        shifted.groupby(
            ["policy", "alpha", "monitor", "mode"],
            sort=True,
            observed=True,
        )
        .agg(
            event_streams=("scenario_id", "size"),
            pre_event_any_alarm_rate=("pre_event_any_alarm", "mean"),
            detection_rate=("detected_event", "mean"),
            mean_post_event_power=("power", "mean"),
            mean_conditional_delay=("conditional_delay", "mean"),
        )
        .reset_index()
    )


def _streamwise_feasibility(
    monitor_frame: pd.DataFrame, alphas: list[float]
) -> pd.DataFrame:
    """Audit whether finite-sample streamwise conformal control is possible.

    Models sharing a dataset/seed consume the same target-feature trajectory,
    so dataset/seed rather than model/scenario is counted as an independent
    null trajectory for this conservative feasibility check.
    """

    null = monitor_frame[monitor_frame["shift"] == "no_shift"]
    records = []
    for policy, group_columns in POLICY_GROUPS.items():
        groups = null[group_columns].drop_duplicates().sort_values(group_columns)
        for held_out_seed in sorted(null["seed"].unique()):
            training = null[null["seed"] != held_out_seed]
            for group in groups.itertuples(index=False, name=None):
                group_values = dict(zip(group_columns, group))
                mask = pd.Series(True, index=training.index)
                for column, value in group_values.items():
                    mask &= training[column].eq(value)
                selected = training.loc[mask]
                effective_trajectories = selected[
                    ["dataset", "seed"]
                ].drop_duplicates()
                for alpha in alphas:
                    minimum = int(math.ceil((1.0 - alpha) / alpha))
                    count = len(effective_trajectories)
                    records.append(
                        {
                            "policy": policy,
                            "held_out_seed": int(held_out_seed),
                            "alpha": float(alpha),
                            **group_values,
                            "effective_independent_null_trajectories": count,
                            "minimum_for_finite_split_conformal_threshold": minimum,
                            "streamwise_control_feasible": count >= minimum,
                        }
                    )
    return pd.DataFrame(records)


def _paired_primary_comparison(scenarios: pd.DataFrame) -> pd.DataFrame:
    subset = scenarios[
        ((scenarios["policy"] == "original_v7") & np.isclose(scenarios["alpha"], 0.01))
        | ((scenarios["policy"] == PRIMARY_POLICY) & np.isclose(scenarios["alpha"], 0.01))
    ].copy()
    metrics = [
        "false_alarm_rate",
        "pre_event_any_alarm",
        "detected_event",
        "power",
        "detection_delay",
        "batch_alarm_rate",
    ]
    keys = [
        "scenario_id",
        "dataset",
        "model",
        "monitor",
        "shift",
        "severity",
        "mode",
        "seed",
    ]
    original = subset[subset["policy"] == "original_v7"][keys + metrics]
    calibrated = subset[subset["policy"] == PRIMARY_POLICY][keys + metrics]
    paired = calibrated.merge(
        original,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_recalibrated", "_original"),
    )
    if len(paired) * 2 != len(subset):
        raise ValueError("Primary recalibration pairing is incomplete")
    for metric in metrics:
        paired[f"{metric}_difference"] = (
            pd.to_numeric(
                paired[f"{metric}_recalibrated"], errors="coerce"
            ).astype(float)
            - pd.to_numeric(
                paired[f"{metric}_original"], errors="coerce"
            ).astype(float)
        )
    return paired


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.v7_results_dir.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise ValueError("output-dir must differ from v7-results-dir")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifest, batches = _validate_v7(source)
    monitor_frame, events = _calibration_frame(batches)
    alphas = sorted(float(value) for value in args.alphas)
    thresholds = _fit_cross_fitted_thresholds(monitor_frame, alphas)
    alarms = _apply_recalibration(monitor_frame, thresholds, alphas)
    evaluated_batches, scenarios = _score_scenarios(alarms, events)
    null_summary = _null_summary(scenarios)
    detection_summary = _detection_summary(scenarios)
    streamwise_feasibility = _streamwise_feasibility(monitor_frame, alphas)
    paired = _paired_primary_comparison(scenarios)

    atomic_parquet(thresholds, output / "calibration_thresholds.parquet")
    atomic_parquet(evaluated_batches, output / "batch_alarm_metrics.parquet")
    atomic_parquet(scenarios, output / "scenario_alarm_metrics.parquet")
    atomic_parquet(paired, output / "paired_primary_comparison.parquet")
    _write_csv_atomic(null_summary, output / "null_calibration_summary.csv")
    _write_csv_atomic(detection_summary, output / "detection_summary.csv")
    _write_csv_atomic(
        streamwise_feasibility, output / "streamwise_feasibility.csv"
    )

    source_rows = len(monitor_frame)
    variants = 1 + len(POLICY_GROUPS) * len(alphas)
    expected_batch_rows = source_rows * variants
    expected_scenario_rows = batches["scenario_id"].nunique() * variants
    if len(evaluated_batches) != expected_batch_rows:
        raise RuntimeError("Step 7 batch grid is incomplete")
    if len(scenarios) != expected_scenario_rows:
        raise RuntimeError("Step 7 scenario grid is incomplete")

    created_at = datetime.now(timezone.utc).isoformat()
    status = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "complete": True,
        "source_batch_rows": source_rows,
        "alarm_variants": variants,
        "batch_alarm_rows": len(evaluated_batches),
        "scenario_alarm_rows": len(scenarios),
        "paired_primary_rows": len(paired),
        "created_at_utc": created_at,
    }
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "purpose": "cross-fitted null-control alarm recalibration",
        "source_schema_version": source_manifest["schema_version"],
        "primary_policy": PRIMARY_POLICY,
        "policies": {
            "original_v7": "saved schema-v7 alarm decisions",
            "global_monitor": "threshold pooled across datasets and models per monitor",
            PRIMARY_POLICY: "threshold pooled across models within dataset and monitor",
        },
        "alphas": alphas,
        "threshold_rule": (
            "strict upper-tail alarm using split-conformal order statistic "
            "ceil((n+1)*(1-alpha))"
        ),
        "cross_fitting": {
            "held_out_unit": "seed",
            "training_seeds_per_fold": 4,
            "held_out_seed_excluded_from_threshold_fit": True,
        },
        "score_normalization": (
            "(monitor_score - reference_null_median) / "
            "(reference_99pct_threshold - reference_null_median)"
        ),
        "calibration_inputs": [
            "monitor score",
            "reference null median",
            "reference alarm threshold",
            "dataset/model/monitor identity",
            "dedicated no-shift control indicator",
        ],
        "forbidden_calibration_inputs": sorted(FORBIDDEN_CALIBRATION_COLUMNS),
        "offline_evaluation": (
            "monitor-declared event flags are joined only after alarms are fixed"
        ),
        "interpretation": {
            "alpha_target": "marginal batch-level false-alarm probability",
            "streamwise_error_controlled": False,
            "streamwise_any_alarm_reported_separately": True,
            "formal_exchangeability_guarantee": False,
            "reason": "batches within a deployment stream may be dependent",
            "streamwise_feasibility_audited": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "created_at_utc": created_at,
    }
    atomic_json(status, output / "step7_status.json")
    atomic_json(manifest, output / "step7_manifest.json")
    print("=== TABMON STEP 7 ALARM RECALIBRATION ===")
    print(f"Source batch rows: {source_rows}")
    print(f"Alarm variants: {variants}")
    print(f"Batch alarm rows: {len(evaluated_batches)}")
    print(f"Scenario alarm rows: {len(scenarios)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
