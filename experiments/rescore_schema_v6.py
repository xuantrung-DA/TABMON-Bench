"""Rescore a completed schema-v5 run with schema-v6 alarm semantics.

The operation is label-offline and does not rerun models or monitors. It writes
a new results directory, preserving the original schema-v5 artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_alarm_event_metrics


SCHEMA_VERSION = 6
ALARM_TARGET_BY_MONITOR = {
    "confidence": "risk_event",
    "drift_shap": "distribution_shift",
}


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _monitor_failure(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    applicable = pd.Series(False, index=frame.index, dtype=bool)
    for column in ("risk_failure", "attribution_failure", "alarm_failure"):
        values = frame[column]
        valid = values.notna()
        applicable |= valid
        result |= valid & values.fillna(False).astype(bool)
    return result.where(applicable, other=pd.NA)


def rescore_batches(
    batches: pd.DataFrame, risk_event_threshold: float
) -> pd.DataFrame:
    required = {
        "scenario_id",
        "batch_index",
        "monitor",
        "shift",
        "shift_fraction",
        "alarm",
        "supports_alarm",
        "risk_failure",
        "attribution_failure",
        "true_excess_risk",
    }
    missing = required - set(batches.columns)
    if missing:
        raise ValueError(f"Missing schema-v5 batch columns: {sorted(missing)}")

    rescored = batches.copy()
    if "true_risk_event" not in rescored:
        rescored["true_risk_event"] = (
            pd.to_numeric(rescored["true_excess_risk"], errors="coerce")
            > risk_event_threshold
        )
    rescored["distribution_shift_event"] = (
        rescored["shift"].ne("no_shift")
        & pd.to_numeric(rescored["shift_fraction"], errors="coerce").gt(0.0)
    )

    targets = rescored["monitor"].map(ALARM_TARGET_BY_MONITOR)
    unknown = sorted(rescored.loc[targets.isna(), "monitor"].unique())
    if unknown:
        raise ValueError(
            "No schema-v6 alarm target is registered for monitors: "
            f"{unknown}"
        )
    rescored["alarm_target"] = targets
    rescored["alarm_target_event"] = np.where(
        targets.eq("risk_event"),
        rescored["true_risk_event"].astype(bool),
        rescored["distribution_shift_event"].astype(bool),
    )
    alarm_rows = rescored["supports_alarm"].astype(bool)
    rescored["alarm_failure"] = pd.Series(pd.NA, index=rescored.index)
    rescored.loc[alarm_rows, "alarm_failure"] = (
        rescored.loc[alarm_rows, "alarm"].astype(bool).to_numpy()
        != rescored.loc[alarm_rows, "alarm_target_event"].astype(bool).to_numpy()
    )
    event_batches = (
        rescored.loc[rescored["true_risk_event"].astype(bool)]
        .groupby("scenario_id")["batch_index"]
        .min()
    )
    first_event = rescored["scenario_id"].map(event_batches)
    rescored["early_warning"] = (
        rescored["alarm"].astype(bool)
        & rescored["distribution_shift_event"].astype(bool)
        & first_event.notna()
        & pd.to_numeric(rescored["batch_index"], errors="coerce").lt(first_event)
    )
    rescored["monitor_failure"] = _monitor_failure(rescored)
    return rescored


def _update_aggregate_row(
    row: pd.Series, batches: pd.DataFrame
) -> pd.Series:
    row = row.copy()
    alarm_target = str(batches["alarm_target"].iloc[0])
    row["alarm_target"] = alarm_target

    mean_columns = [
        "alarm_failure",
        "monitor_failure",
        "alarm",
        "true_risk_event",
        "distribution_shift_event",
        "alarm_target_event",
        "early_warning",
    ]
    for column in mean_columns:
        row[f"{column}_mean"] = float(
            pd.to_numeric(batches[column], errors="coerce").mean()
        )

    pre_mask = pd.to_numeric(batches["shift_fraction"], errors="coerce").eq(0.0)
    post_mask = ~pre_mask
    for phase, mask in (("pre", pre_mask), ("post", post_mask)):
        for column in mean_columns:
            values = pd.to_numeric(batches.loc[mask, column], errors="coerce")
            row[f"{phase}_{column}_mean"] = (
                float(values.mean()) if values.notna().any() else np.nan
            )

    alarms = batches["alarm"].astype(bool).tolist()
    target_metrics = compute_alarm_event_metrics(
        alarms, batches["alarm_target_event"].astype(bool).tolist()
    )
    row["alarm_target_event_batch"] = target_metrics["event_batch"]
    for name in (
        "first_alarm_batch",
        "false_alarm_rate",
        "power",
        "detection_delay",
        "missed_alarm",
    ):
        row[name] = target_metrics[name]

    for prefix, column in (
        ("risk_alarm", "true_risk_event"),
        ("shift_alarm", "distribution_shift_event"),
    ):
        metrics = compute_alarm_event_metrics(
            alarms, batches[column].astype(bool).tolist()
        )
        for name, value in metrics.items():
            row[f"{prefix}_{name}"] = value
    row["true_risk_event_batch"] = row["risk_alarm_event_batch"]
    row["early_warning_count"] = int(batches["early_warning"].sum())
    return row


def rescore_aggregate(
    aggregate: pd.DataFrame, batches: pd.DataFrame
) -> pd.DataFrame:
    if "scenario_id" not in aggregate:
        raise ValueError("aggregate_metrics.parquet has no scenario_id")
    grouped = {key: frame for key, frame in batches.groupby("scenario_id")}
    missing = sorted(set(aggregate["scenario_id"]) - set(grouped))
    if missing:
        raise ValueError(f"Missing batch trajectories for {len(missing)} scenarios")
    rows = [
        _update_aggregate_row(row, grouped[row["scenario_id"]])
        for _, row in aggregate.iterrows()
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore schema-v5 results using schema-v6 alarm targets."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.results_dir.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise ValueError("output-dir must differ from results-dir")
    for filename in (
        "batch_metrics.parquet",
        "aggregate_metrics.parquet",
        "benchmark_manifest.json",
        "benchmark_status.json",
    ):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (source / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    if int(manifest.get("schema_version", -1)) != 5:
        raise ValueError("Input manifest must be schema version 5")
    threshold = float(manifest["configuration"]["risk_event_threshold"])
    batches = rescore_batches(
        pd.read_parquet(source / "batch_metrics.parquet"), threshold
    )
    aggregate = rescore_aggregate(
        pd.read_parquet(source / "aggregate_metrics.parquet"), batches
    )

    _write_table_atomic(batches, output / "batch_metrics.parquet")
    _write_table_atomic(aggregate, output / "aggregate_metrics.parquet")
    _write_table_atomic(aggregate, output / "aggregate_metrics.csv")

    rescored_at = datetime.now(timezone.utc).isoformat()
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["rescored_from_schema_version"] = 5
    manifest["rescored_at_utc"] = rescored_at
    manifest["source_results_dir"] = str(source)
    manifest.setdefault("methodology", {}).update(
        {
            "alarm_target_aware_scoring": True,
            "alarm_targets": ALARM_TARGET_BY_MONITOR,
            "early_warning_definition": (
                "alarm after controlled shift onset but before oracle risk event"
            ),
        }
    )
    _write_json_atomic(manifest, output / "benchmark_manifest.json")

    status = json.loads(
        (source / "benchmark_status.json").read_text(encoding="utf-8")
    )
    status.update(
        {
            "schema_version": SCHEMA_VERSION,
            "rescored_from_schema_version": 5,
            "rescored_at_utc": rescored_at,
        }
    )
    _write_json_atomic(status, output / "benchmark_status.json")

    print("=== TABMON-BENCH SCHEMA-V6 RESCORE ===")
    print(f"Source scenarios: {len(aggregate)}")
    print(f"Source batches: {len(batches)}")
    print(f"Saved at: {output}")
    print("Models and monitors were not rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
