"""Upgrade a completed schema-v6 run to one-sided schema-v7 risk alarms.

Only the 20 Confidence null calibrators are reconstructed. Saved target-batch
risk estimates and every DriftSHAP result are reused, so shift scenarios and
base models are not rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.b0_confidence import ConfidenceShiftMonitor
from src.evaluation.metrics import compute_alarm_event_metrics


SCHEMA_VERSION = 7


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


def _recalibrate_confidence_alarms(
    batches: pd.DataFrame,
    manifest: dict[str, Any],
    data_dir: Path,
    base_models_dir: Path,
) -> pd.DataFrame:
    rescored = batches.copy()
    confidence_rows = rescored["monitor"].eq("confidence")
    rescored.loc[confidence_rows, "alarm_direction"] = "increase"
    rescored.loc[~confidence_rows, "alarm_direction"] = "increase"

    configuration = manifest["configuration"]
    batch_size = int(configuration["batch_size"])
    null_batches = int(configuration["null_calibration_batches"])
    alarm_quantile = float(configuration["alarm_quantile"])
    alpha = 1.0 - alarm_quantile

    pairs = (
        rescored.loc[confidence_rows, ["dataset", "model"]]
        .drop_duplicates()
        .sort_values(["dataset", "model"])
    )
    for pair_index, pair in enumerate(pairs.itertuples(index=False), start=1):
        dataset, model_name = pair.dataset, pair.model
        target = manifest["dataset_config"][dataset]["target"]
        reference = pd.read_parquet(data_dir / dataset / "calibration.parquet")
        model = joblib.load(
            base_models_dir / "models" / dataset / f"{model_name}_calibrated.pkl"
        )
        monitor = ConfidenceShiftMonitor(
            reference.drop(columns=[target]),
            reference[target],
            model,
            batch_size=batch_size,
            null_calibration_batches=null_batches,
            alarm_quantile=alarm_quantile,
        )

        mask = (
            confidence_rows
            & rescored["dataset"].eq(dataset)
            & rescored["model"].eq(model_name)
        )
        scores = pd.to_numeric(
            rescored.loc[mask, "estimated_risk_change"], errors="raise"
        ).to_numpy(dtype=float)
        p_values = (
            1
            + np.sum(
                monitor.null_scores.reshape(1, -1) >= scores.reshape(-1, 1),
                axis=1,
            )
        ) / (len(monitor.null_scores) + 1)

        previous_median = pd.to_numeric(
            rescored.loc[mask, "null_score_median"], errors="coerce"
        ).dropna()
        if len(previous_median) and not np.allclose(
            previous_median.to_numpy(), monitor.null_score_median, atol=1e-12
        ):
            raise RuntimeError(
                f"Null calibration mismatch for {dataset}/{model_name}"
            )

        rescored.loc[mask, "alarm_p_value"] = p_values
        rescored.loc[mask, "alarm"] = p_values <= alpha
        rescored.loc[mask, "alarm_threshold"] = monitor.alarm_threshold
        rescored.loc[mask, "null_score_median"] = monitor.null_score_median
        rescored.loc[mask, "alarm_direction"] = "increase"
        print(
            f"[{pair_index}/{len(pairs)}] calibrated {dataset}/{model_name}",
            flush=True,
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


def _update_aggregate_row(row: pd.Series, batches: pd.DataFrame) -> pd.Series:
    row = row.copy()
    row["alarm_direction"] = str(batches["alarm_direction"].iloc[0])
    mean_columns = [
        "alarm",
        "alarm_p_value",
        "alarm_failure",
        "monitor_failure",
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


def _rescore_aggregate(
    aggregate: pd.DataFrame, batches: pd.DataFrame
) -> pd.DataFrame:
    grouped = {key: frame for key, frame in batches.groupby("scenario_id")}
    missing = sorted(set(aggregate["scenario_id"]) - set(grouped))
    if missing:
        raise ValueError(f"Missing batch trajectories for {len(missing)} scenarios")
    return pd.DataFrame(
        [
            _update_aggregate_row(row, grouped[row["scenario_id"]])
            for _, row in aggregate.iterrows()
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore schema-v6 Confidence alarms using an upper-tail test."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.results_dir.resolve()
    output = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    base_models_dir = args.base_models_dir.resolve()
    if source == output:
        raise ValueError("output-dir must differ from results-dir")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
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
    if int(manifest.get("schema_version", -1)) != 6:
        raise ValueError("Input manifest must be schema version 6")
    batches = _recalibrate_confidence_alarms(
        pd.read_parquet(source / "batch_metrics.parquet"),
        manifest,
        data_dir,
        base_models_dir,
    )
    aggregate = _rescore_aggregate(
        pd.read_parquet(source / "aggregate_metrics.parquet"), batches
    )
    _write_table_atomic(batches, output / "batch_metrics.parquet")
    _write_table_atomic(aggregate, output / "aggregate_metrics.parquet")
    _write_table_atomic(aggregate, output / "aggregate_metrics.csv")

    rescored_at = datetime.now(timezone.utc).isoformat()
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["rescored_from_schema_version"] = 6
    manifest["rescored_at_utc"] = rescored_at
    manifest["source_results_dir"] = str(source)
    manifest.setdefault("methodology", {}).update(
        {
            "alarm_direction_aware_scoring": True,
            "alarm_directions": {
                "confidence": "increase",
                "drift_shap": "increase",
            },
            "confidence_alarm_test": (
                "upper-tail conformal test for an increase in estimated risk"
            ),
            "scenario_ids_preserved_from_schema_version": 6,
        }
    )
    _write_json_atomic(manifest, output / "benchmark_manifest.json")

    status = json.loads(
        (source / "benchmark_status.json").read_text(encoding="utf-8")
    )
    status.update(
        {
            "schema_version": SCHEMA_VERSION,
            "rescored_from_schema_version": 6,
            "rescored_at_utc": rescored_at,
        }
    )
    _write_json_atomic(status, output / "benchmark_status.json")
    print("=== TABMON-BENCH SCHEMA-V7 RESCORE ===")
    print(f"Scenarios: {len(aggregate)}")
    print(f"Batches: {len(batches)}")
    print(f"Saved at: {output}")
    print("Shift streams and target-batch monitor scores were not rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
