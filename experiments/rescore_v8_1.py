"""Rescore schema-v8 Confidence oracle metrics with the frozen v7 risk rule.

The operation is post-hoc and immutable: the parent v8 directory is read-only,
monitor estimates and alarm decisions are preserved, and only label-aware
offline Confidence evaluation fields are recomputed from the repaired cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_step8_full_v8 import (
    CONFIDENCE_METHOD,
    RISK_EVENT_THRESHOLD,
    RISK_FAILURE_TOLERANCE,
    _scenario_summaries,
)
from src.baselines.cached_confidence import probability_matrix
from src.cache.stream_cache import (
    BATCH_INDEX,
    KEY_COLUMNS,
    TARGET_LABEL,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
)
from src.evaluation.risk import BINARY_LOG_LOSS_EPSILON, binary_log_losses


RESCORE_VERSION = "8.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_and_validate(v8_dir: Path, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    status = json.loads((v8_dir / "v8_status.json").read_text("utf-8"))
    manifest = json.loads((v8_dir / "v8_manifest.json").read_text("utf-8"))
    cache_status = json.loads((cache_dir / "cache_status.json").read_text("utf-8"))
    if status.get("complete") is not True or status.get("schema_version") != 8:
        raise ValueError("Rescoring requires a complete schema-v8 parent")
    if cache_status.get("complete") is not True:
        raise ValueError("Rescoring requires a complete stream cache")
    if cache_status.get("cache_schema_version") != 1:
        raise ValueError("Unexpected stream-cache schema")
    if cache_status.get("cache_revision", 0) < 1:
        raise ValueError("Rescoring requires repaired cache revision 1 or newer")
    batches = pd.read_parquet(v8_dir / "batch_metrics.parquet")
    if len(batches) != 217_000 or batches["predictor_stream_id"].nunique() != 3100:
        raise ValueError("Parent v8 batch grid is incomplete")
    return batches, manifest


def _reference_risks(
    observable: ObservableCacheReader,
    predictor_index: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in predictor_index.drop_duplicates(["dataset", "model"]).itertuples(
        index=False
    ):
        predictions = observable.load_reference_probabilities(row.dataset, row.model)
        labels = observable.load_source_calibration_labels(row.dataset)
        classes = np.asarray(json.loads(row.classes_json))
        losses = binary_log_losses(
            probability_matrix(predictions), classes, labels.to_numpy()
        )
        result[(row.dataset, row.model)] = float(losses.mean())
    return result


def _corrected_oracle_grid(
    cache_dir: Path,
    confidence_grid: pd.DataFrame,
) -> pd.DataFrame:
    observable = ObservableCacheReader(cache_dir)
    oracle = OracleCacheReader(cache_dir)
    predictor_index = pd.read_parquet(
        cache_dir / "control" / "predictor_stream_index.parquet"
    )
    selected = confidence_grid[
        ["predictor_stream_id", "stream_id", "dataset", "model"]
    ].drop_duplicates()
    predictor_index = selected.merge(
        predictor_index[
            ["predictor_stream_id", "stream_id", "dataset", "model", "classes_json"]
        ],
        on=["predictor_stream_id", "stream_id", "dataset", "model"],
        how="left",
        validate="one_to_one",
    )
    if predictor_index["classes_json"].isna().any() or len(predictor_index) != 3100:
        raise ValueError("Could not align the v8 grid with cached predictors")
    reference_risk = _reference_risks(observable, predictor_index)

    records: list[pd.DataFrame] = []
    grouped = predictor_index.sort_values(["dataset", "stream_id", "model"]).groupby(
        ["dataset", "stream_id"], sort=True
    )
    for position, ((dataset, stream_id), stream_rows) in enumerate(grouped, start=1):
        targets = oracle.load_target_labels(stream_id)
        for row in stream_rows.itertuples(index=False):
            predictions = observable.load_target_probabilities(row.predictor_stream_id)
            paired = predictions.merge(
                targets[KEY_COLUMNS + [TARGET_LABEL]],
                on=KEY_COLUMNS,
                how="inner",
                validate="one_to_one",
            )
            classes = np.asarray(json.loads(row.classes_json))
            losses = binary_log_losses(
                probability_matrix(paired), classes, paired[TARGET_LABEL].to_numpy()
            )
            working = paired[[BATCH_INDEX]].copy()
            working["_corrected_sample_log_loss"] = losses
            corrected = (
                working.groupby(BATCH_INDEX, sort=True)["_corrected_sample_log_loss"]
                .mean()
                .sub(reference_risk[(dataset, row.model)])
                .rename("corrected_true_excess_risk")
                .reset_index()
            )
            corrected.insert(0, "predictor_stream_id", row.predictor_stream_id)
            records.append(corrected)
        if position == 1 or position % 50 == 0 or position == len(grouped):
            print(f"[{position}/{len(grouped)}] rescored {dataset}/{stream_id}", flush=True)
    result = pd.concat(records, ignore_index=True)
    if len(result) != 31_000:
        raise RuntimeError(f"Expected 31,000 corrected risk records; found {len(result)}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    v8_dir = args.v8_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == v8_dir:
        raise ValueError("v8.1 must not overwrite its parent v8 directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    batches, parent_manifest = _load_and_validate(v8_dir, cache_dir)
    original = batches.copy(deep=True)
    confidence_mask = batches["method"].eq(CONFIDENCE_METHOD)
    corrected = _corrected_oracle_grid(cache_dir, batches.loc[confidence_mask])
    lookup = corrected.set_index(["predictor_stream_id", BATCH_INDEX])[
        "corrected_true_excess_risk"
    ]
    confidence_index = pd.MultiIndex.from_frame(
        batches.loc[confidence_mask, ["predictor_stream_id", BATCH_INDEX]]
    )
    corrected_values = lookup.reindex(confidence_index).to_numpy(dtype=float)
    if not np.isfinite(corrected_values).all():
        raise RuntimeError("Corrected oracle grid contains missing values")

    old_values = batches.loc[confidence_mask, "true_value"].to_numpy(dtype=float)
    estimates = batches.loc[confidence_mask, "estimated_value"].to_numpy(dtype=float)
    batches.loc[confidence_mask, "true_value"] = corrected_values
    batches.loc[confidence_mask, "signed_error"] = estimates - corrected_values
    batches.loc[confidence_mask, "absolute_error"] = np.abs(
        estimates - corrected_values
    )
    batches.loc[confidence_mask, "risk_failure"] = (
        np.abs(estimates - corrected_values) > RISK_FAILURE_TOLERANCE
    ).astype(float)
    batches.loc[confidence_mask, "alarm_target_event"] = (
        corrected_values > RISK_EVENT_THRESHOLD
    )

    # Fail closed: monitoring outputs and every non-Confidence row are immutable.
    immutable_columns = [
        "estimated_value",
        "monitor_score",
        "alarm",
        "alarm_threshold",
        "alarm_p_value",
        "predicted_attribution_json",
    ]
    for column in immutable_columns:
        if not original[column].equals(batches[column]):
            raise RuntimeError(f"Rescore modified monitor output column: {column}")
    if not original.loc[~confidence_mask].equals(batches.loc[~confidence_mask]):
        raise RuntimeError("Rescore modified a non-Confidence record")

    scenarios = _scenario_summaries(batches)
    atomic_parquet(batches, output_dir / "batch_metrics.parquet")
    atomic_parquet(scenarios, output_dir / "scenario_metrics.parquet")
    for name in (
        "alarm_calibration_thresholds.parquet",
        "alarm_calibration_maxima.parquet",
    ):
        shutil.copy2(v8_dir / name, output_dir / name)

    diagnostic_frame = batches.loc[confidence_mask, [
        "shift", "dataset", "model", "true_value", "alarm_target_event"
    ]].copy()
    diagnostic_frame["old_true_excess_risk"] = old_values
    diagnostic_frame["absolute_correction"] = np.abs(corrected_values - old_values)
    diagnostics = (
        diagnostic_frame.groupby("shift", as_index=False)
        .agg(
            batches=("true_value", "size"),
            corrected_true_excess_risk=("true_value", "mean"),
            old_true_excess_risk=("old_true_excess_risk", "mean"),
            mean_absolute_correction=("absolute_correction", "mean"),
            corrections_over_0_01=("absolute_correction", lambda x: int((x > 0.01).sum())),
            corrections_over_0_10=("absolute_correction", lambda x: int((x > 0.10).sum())),
        )
    )
    diagnostics.to_csv(output_dir / "rescore_diagnostics.csv", index=False)

    parent_hashes = {
        name: _sha256(v8_dir / name)
        for name in ("batch_metrics.parquet", "scenario_metrics.parquet", "v8_manifest.json")
    }
    manifest = {
        "schema_version": RESCORE_VERSION,
        "operation": "oracle-only post-hoc rescore",
        "parent_schema_version": parent_manifest["schema_version"],
        "parent_file_sha256": parent_hashes,
        "risk_endpoint": "binary log loss",
        "probability_clip_epsilon": BINARY_LOG_LOSS_EPSILON,
        "risk_event_threshold": RISK_EVENT_THRESHOLD,
        "risk_failure_tolerance": RISK_FAILURE_TOLERANCE,
        "monitor_outputs_recomputed": False,
        "alarm_decisions_recomputed": False,
        "offline_fields_recomputed": [
            "true_value",
            "signed_error",
            "absolute_error",
            "risk_failure",
            "alarm_target_event",
            "scenario event/power/delay summaries",
        ],
        "non_confidence_rows_changed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest, output_dir / "v8_1_manifest.json")
    atomic_json(
        {
            "schema_version": RESCORE_VERSION,
            "complete": True,
            "predictor_streams": int(batches["predictor_stream_id"].nunique()),
            "method_stream_evaluations": len(scenarios),
            "batch_method_records": len(batches),
            "corrected_confidence_batch_records": int(confidence_mask.sum()),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "v8_1_status.json",
    )
    print("=== TABMON V8.1 ORACLE RESCORE ===")
    print(f"Corrected Confidence batches: {int(confidence_mask.sum())}")
    print(f"Mean absolute correction: {np.mean(np.abs(corrected_values - old_values)):.6f}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
