"""Validate and summarize a TABMON-Bench schema-v7 reliability run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.meta_monitor_schema import (
    META_MONITOR_FEATURES,
    available_meta_monitor_features,
    meta_monitor_allowlist_audit,
    meta_monitor_failure_target,
    meta_monitor_predictors,
)


REQUIRED_BATCH_COLUMNS = {
    "scenario_id",
    "batch_index",
    "alarm",
    "alarm_p_value",
    "monitor_score",
    "estimated_risk_ci_lower",
    "estimated_risk_ci_upper",
    "supports_risk_estimation",
    "supports_attribution",
    "supports_alarm",
    "alarm_target",
    "alarm_direction",
    "true_risk_event",
    "distribution_shift_event",
    "alarm_target_event",
    "early_warning",
    "risk_failure",
    "attribution_failure",
    "alarm_failure",
    "monitor_failure",
    "attribution_abs_mass",
    "attribution_max_abs",
    "attribution_topk_jaccard",
    "domain_classifier_auc",
    "effective_sample_size_fraction",
    "density_ratio_clipping_rate",
    "prediction_entropy_shift",
    "prediction_confidence_shift",
    "novel_support_rate",
}

DIAGNOSTIC_COLUMNS = list(META_MONITOR_FEATURES)


def _usable_meta_monitor_features(frame: pd.DataFrame) -> list[str]:
    usable = []
    for column in available_meta_monitor_features(frame):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() and values.std() > 0:
            usable.append(column)
    return usable


def _prepare_meta_monitor_split(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str]]:
    """Create X/y with a strict separation between observables and oracle label."""
    usable = _usable_meta_monitor_features(train)
    if not usable:
        raise ValueError("No usable allowlisted meta-monitor features")
    train_X = meta_monitor_predictors(train, usable)
    test_X = meta_monitor_predictors(test, usable)
    medians = train_X.median().fillna(0.0)
    train_X = train_X.fillna(medians)
    test_X = test_X.fillna(medians)
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_X),
        scaler.transform(test_X),
        meta_monitor_failure_target(train),
        meta_monitor_failure_target(test),
        usable,
    )


def cross_seed_failure_auc(batches: pd.DataFrame) -> pd.DataFrame:
    """Evaluate diagnostic failure prediction with every seed held out once."""
    records = []
    for monitor, monitor_frame in batches.groupby("monitor"):
        for held_out_seed in sorted(monitor_frame["seed"].unique()):
            train = monitor_frame[monitor_frame["seed"] != held_out_seed].copy()
            test = monitor_frame[monitor_frame["seed"] == held_out_seed].copy()
            y_train = meta_monitor_failure_target(train)
            y_test = meta_monitor_failure_target(test)
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            try:
                train_scaled, test_scaled, y_train, y_test, usable = (
                    _prepare_meta_monitor_split(train, test)
                )
            except ValueError:
                continue
            classifier = LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=17_021
            )
            classifier.fit(train_scaled, y_train)
            probability = classifier.predict_proba(test_scaled)[:, 1]
            records.append(
                {
                    "monitor": monitor,
                    "held_out_seed": int(held_out_seed),
                    "roc_auc": float(roc_auc_score(y_test, probability)),
                    "average_precision": float(
                        average_precision_score(y_test, probability)
                    ),
                    "failure_prevalence": float(y_test.mean()),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_features": len(usable),
                    "features": json.dumps(usable),
                }
            )
    return pd.DataFrame(records)


def grouped_failure_auc(
    batches: pd.DataFrame, group_columns: tuple[str, ...] = ("dataset", "model", "shift")
) -> pd.DataFrame:
    """Evaluate failure prediction on unseen datasets, models, and shifts."""
    records = []
    for group_column in group_columns:
        for monitor, monitor_frame in batches.groupby("monitor"):
            for held_out_group in sorted(monitor_frame[group_column].unique()):
                train = monitor_frame[
                    monitor_frame[group_column] != held_out_group
                ].copy()
                test = monitor_frame[
                    monitor_frame[group_column] == held_out_group
                ].copy()
                y_train = meta_monitor_failure_target(train)
                y_test = meta_monitor_failure_target(test)
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                try:
                    train_scaled, test_scaled, y_train, y_test, usable = (
                        _prepare_meta_monitor_split(train, test)
                    )
                except ValueError:
                    continue
                classifier = LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=17_021,
                )
                classifier.fit(train_scaled, y_train)
                probability = classifier.predict_proba(test_scaled)[:, 1]
                records.append(
                    {
                        "held_out_dimension": group_column,
                        "monitor": monitor,
                        "held_out_group": held_out_group,
                        "roc_auc": float(roc_auc_score(y_test, probability)),
                        "average_precision": float(
                            average_precision_score(y_test, probability)
                        ),
                        "failure_prevalence": float(y_test.mean()),
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_features": len(usable),
                        "features": json.dumps(usable),
                    }
                )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_path = args.results_dir / "benchmark_status.json"
    aggregate_path = args.results_dir / "aggregate_metrics.parquet"
    batch_path = args.results_dir / "batch_metrics.parquet"
    for path in (status_path, aggregate_path, batch_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    aggregate = pd.read_parquet(aggregate_path)
    batches = pd.read_parquet(batch_path)
    missing = REQUIRED_BATCH_COLUMNS - set(batches.columns)
    if missing:
        raise ValueError(f"Missing schema-v7 columns: {sorted(missing)}")
    allowlist_audit = meta_monitor_allowlist_audit()
    (args.results_dir / "meta_monitor_feature_audit.json").write_text(
        json.dumps(allowlist_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    alarm_rows = batches["supports_alarm"].astype(bool)
    if batches.loc[alarm_rows, "alarm"].isna().any():
        raise ValueError("Alarm-capable monitor contains null alarms")
    if batches.loc[alarm_rows, "alarm_p_value"].isna().any():
        raise ValueError("Alarm-capable monitor contains null conformal p-values")
    declared_targets = set(batches.loc[alarm_rows, "alarm_target"].dropna().unique())
    valid_targets = {"risk_event", "distribution_shift"}
    if not declared_targets or not declared_targets <= valid_targets:
        raise ValueError(f"Invalid alarm targets: {sorted(declared_targets)}")
    declared_directions = set(
        batches.loc[alarm_rows, "alarm_direction"].dropna().unique()
    )
    if declared_directions != {"increase"}:
        raise ValueError(f"Invalid alarm directions: {sorted(declared_directions)}")
    expected_target_event = np.where(
        batches["alarm_target"].eq("risk_event"),
        batches["true_risk_event"].astype(bool),
        batches["distribution_shift_event"].astype(bool),
    )
    if not np.array_equal(
        batches.loc[alarm_rows, "alarm_target_event"].astype(bool).to_numpy(),
        np.asarray(expected_target_event, dtype=bool)[alarm_rows.to_numpy()],
    ):
        raise ValueError("alarm_target_event does not match the declared target")
    expected_alarm_failure = (
        batches.loc[alarm_rows, "alarm"].astype(bool).to_numpy()
        != batches.loc[alarm_rows, "alarm_target_event"].astype(bool).to_numpy()
    )
    if not np.array_equal(
        batches.loc[alarm_rows, "alarm_failure"].astype(bool).to_numpy(),
        expected_alarm_failure,
    ):
        raise ValueError("alarm_failure is not scored against alarm_target_event")
    event_batches = (
        batches.loc[batches["true_risk_event"].astype(bool)]
        .groupby("scenario_id")["batch_index"]
        .min()
    )
    first_event = batches["scenario_id"].map(event_batches)
    expected_early_warning = (
        batches["alarm"].astype(bool)
        & batches["distribution_shift_event"].astype(bool)
        & first_event.notna()
        & pd.to_numeric(batches["batch_index"], errors="coerce").lt(first_event)
    )
    if not np.array_equal(
        batches["early_warning"].astype(bool).to_numpy(),
        expected_early_warning.to_numpy(),
    ):
        raise ValueError("early_warning labels are inconsistent")
    risk_rows = batches["supports_risk_estimation"].astype(bool)
    interval_columns = ["estimated_risk_ci_lower", "estimated_risk_ci_upper"]
    if batches.loc[risk_rows, interval_columns].isna().any().any():
        raise ValueError("Risk-capable monitor contains null risk intervals")
    if batches.loc[risk_rows, "risk_failure"].isna().any():
        raise ValueError("Risk-capable monitor contains null risk-failure labels")
    if batches.loc[~risk_rows, "estimated_risk_change"].notna().any():
        raise ValueError("Proxy-only monitor was incorrectly scored as a risk estimator")
    if batches.loc[~risk_rows, "risk_failure"].notna().any():
        raise ValueError("Proxy-only monitor has risk-failure labels")
    attribution_rows = batches["supports_attribution"].astype(bool)
    if batches.loc[~attribution_rows, "attribution_failure"].notna().any():
        raise ValueError("Non-attribution monitor has attribution-failure labels")

    summary_columns = [
        "risk_mae",
        "risk_failure_mean",
        "attribution_failure_mean",
        "alarm_failure_mean",
        "monitor_failure_mean",
        "alarm_mean",
        "coverage_mean",
        "false_alarm_rate",
        "power",
        "detection_delay",
        "missed_alarm",
        "early_warning_mean",
        "early_warning_count",
        "risk_alarm_false_alarm_rate",
        "risk_alarm_power",
        "shift_alarm_false_alarm_rate",
        "shift_alarm_power",
        "attribution_abs_mass_mean",
        "domain_classifier_auc_mean",
        "effective_sample_size_fraction_mean",
        "novel_support_rate_mean",
    ]
    available = [column for column in summary_columns if column in aggregate]
    summary = (
        aggregate.groupby(
            [
                "dataset", "model", "monitor", "alarm_target",
                "alarm_direction", "shift", "mode",
            ],
            dropna=False,
        )[available]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(args.results_dir / "reliability_summary.csv", index=False)
    diagnostic_auc = cross_seed_failure_auc(batches)
    diagnostic_auc.to_csv(
        args.results_dir / "reliability_diagnostic_auc.csv", index=False
    )
    grouped_auc = grouped_failure_auc(batches)
    grouped_auc.to_csv(
        args.results_dir / "reliability_diagnostic_grouped_auc.csv", index=False
    )

    print("=== RELIABILITY RUN AUDIT ===")
    print(
        f"complete={status.get('complete')}  "
        f"scenarios={status.get('completed_scenarios')}/"
        f"{status.get('requested_scenarios')}  "
        f"errors={status.get('errors_this_run')}"
    )
    print(f"aggregate_rows={len(aggregate)}  batch_rows={len(batches)}")
    null_summary = summary[summary["shift"] == "no_shift"]
    if len(null_summary):
        print("\nNULL CONTROL")
        display_columns = [
            "model", "monitor", "risk_mae", "risk_failure_mean",
            "false_alarm_rate", "coverage_mean", "attribution_abs_mass_mean",
            "domain_classifier_auc_mean",
        ]
        print(
            null_summary[[c for c in display_columns if c in null_summary]]
            .round(4)
            .to_string(index=False)
        )
    if len(diagnostic_auc):
        print("\nLEAVE-ONE-SEED-OUT FAILURE PREDICTION")
        print(
            diagnostic_auc.groupby("monitor")["roc_auc"]
            .agg(["mean", "min", "max"])
            .round(4)
            .to_string()
        )
    if len(grouped_auc):
        print("\nGROUPED FAILURE PREDICTION")
        print(
            grouped_auc.groupby(["held_out_dimension", "monitor"])["roc_auc"]
            .agg(["mean", "min", "max"])
            .round(4)
            .to_string()
        )
    print(f"\nSaved: {args.results_dir / 'reliability_summary.csv'}")
    print(f"Saved: {args.results_dir / 'reliability_diagnostic_auc.csv'}")
    print(
        f"Saved: {args.results_dir / 'reliability_diagnostic_grouped_auc.csv'}"
    )
    return 0 if status.get("complete") and status.get("errors_this_run") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
