"""Cached SHD inference sensitivity for alpha and harmful-change epsilon.

The published SHD source error regressor and quantile selector are fitted once
per dataset/model cell.  The sensitivity grid changes only the terms that
mathematically depend on ``alpha`` and ``epsilon``: source Hoeffding bounds,
the target PM-EB confidence sequence, and the final change margin.  Target
labels are loaded solely by the offline oracle to define evaluation events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_step12_q1_extension import _load_index, _stable_seed
from src.baselines.shd import SHDMonitor, pm_eb_lower_bound
from src.cache.stream_cache import (
    BATCH_INDEX,
    TARGET_LABEL,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
)
from src.evaluation.metrics import compute_alarm_event_metrics


SCHEMA_VERSION = 1
DEFAULT_ALPHAS = (0.01, 0.05, 0.10)
DEFAULT_EPSILONS = (0.00, 0.02, 0.05)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS
    )
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if any(not 0.0 < value < 1.0 for value in args.alphas):
        parser.error("Every alpha must lie in (0, 1)")
    if any(value < 0.0 for value in args.epsilons):
        parser.error("Every epsilon must be nonnegative")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("Alpha values must be unique")
    if len(set(args.epsilons)) != len(args.epsilons):
        parser.error("Epsilon values must be unique")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    return args


def _monitor_path(root: Path, dataset: str, model: str) -> Path:
    token = hashlib.sha256(f"{dataset}|{model}".encode()).hexdigest()[:16]
    return root / f"{dataset}__{model}__{token}.joblib"


def _fit_or_load_monitors(
    observable: ObservableCacheReader,
    index: pd.DataFrame,
    monitor_dir: Path,
) -> dict[tuple[str, str], SHDMonitor]:
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitors: dict[tuple[str, str], SHDMonitor] = {}
    cells = index.drop_duplicates(["dataset", "model"])
    for row in cells.itertuples(index=False):
        key = (str(row.dataset), str(row.model))
        path = _monitor_path(monitor_dir, *key)
        if path.is_file():
            monitor = joblib.load(path)
        else:
            monitor = SHDMonitor.fit(
                observable.load_reference_features(row.dataset),
                observable.load_reference_probabilities(row.dataset, row.model),
                observable.load_source_calibration_labels(row.dataset),
                json.loads(row.classes_json),
                random_seed=_stable_seed("shd", *key),
                alpha=0.01,
                epsilon=0.02,
            )
            temporary = path.with_suffix(path.suffix + ".tmp")
            joblib.dump(monitor, temporary)
            os.replace(temporary, path)
        monitors[key] = monitor
        print(
            f"SHD source fit {row.dataset}/{row.model}: "
            f"feasible={monitor.calibration.selector_feasible}",
            flush=True,
        )
    return monitors


def _oracle_error_flags(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    true_error_threshold: float,
) -> np.ndarray:
    merged = predictions.merge(
        labels,
        on=["_tabmon_batch_index", "_tabmon_row_index"],
        validate="one_to_one",
    ).sort_values(["_tabmon_batch_index", "_tabmon_row_index"])
    # ``predicted_class`` is already produced from the complete declared class
    # vector in the frozen cache, so it remains authoritative even when a
    # finite stream does not contain predictions from every class.
    sample_error = (
        merged["predicted_class"].to_numpy() != merged[TARGET_LABEL].to_numpy()
    ).astype(float)
    return sample_error > true_error_threshold


def _evaluate_stream(
    row: Any,
    observable: ObservableCacheReader,
    oracle: OracleCacheReader,
    base_monitor: SHDMonitor,
    alphas: list[float],
    epsilons: list[float],
) -> pd.DataFrame:
    common = {
        "predictor_stream_id": row.predictor_stream_id,
        "stream_id": row.stream_id,
        "dataset": row.dataset,
        "model": row.model,
        "shift": row.shift,
        "severity": row.severity,
        "mode": row.mode,
        "seed": int(row.seed),
        "selector_feasible": bool(base_monitor.calibration.selector_feasible),
    }
    records: list[dict[str, Any]] = []
    if not base_monitor.calibration.selector_feasible:
        for alpha in alphas:
            for epsilon in epsilons:
                records.append(
                    {
                        **common,
                        "alpha": alpha,
                        "epsilon": epsilon,
                        "evaluated": False,
                        "event_batch": np.nan,
                        "first_alarm_batch": np.nan,
                        "false_alarm_rate": np.nan,
                        "power": np.nan,
                        "detection_delay": np.nan,
                        "missed_alarm": np.nan,
                        "stream_any_event": np.nan,
                        "stream_any_alarm": np.nan,
                        "max_oracle_assumption_4_1_gap": np.nan,
                    }
                )
        return pd.DataFrame(records)

    target_X = observable.load_target_features(row.stream_id).reset_index(drop=True)
    predictions = observable.load_target_probabilities(row.predictor_stream_id)
    predictions = predictions.sort_values(
        ["_tabmon_batch_index", "_tabmon_row_index"]
    ).reset_index(drop=True)
    if len(target_X) != len(predictions):
        raise ValueError("Target features and probabilities are not aligned")
    batch_index = predictions[BATCH_INDEX].to_numpy(dtype=int)
    predicted_error = base_monitor.predicted_error(target_X)
    selected = predicted_error > base_monitor.calibration.predicted_error_threshold
    selected_float = selected.astype(float)
    running_selected = np.cumsum(selected_float) / np.arange(1, len(selected) + 1)
    true_high = _oracle_error_flags(
        predictions,
        oracle.load_target_labels(row.stream_id),
        base_monitor.calibration.true_error_threshold,
    )
    selected_high = (selected & true_high).astype(float)
    false_discovery = (selected & ~true_high).astype(float)
    running_selected_high = np.cumsum(selected_high) / np.arange(
        1, len(selected_high) + 1
    )
    assumption_gap = np.cumsum(false_discovery) / np.arange(
        1, len(false_discovery) + 1
    ) - base_monitor.calibration.source_false_discovery_joint_rate

    for alpha in alphas:
        target_lower = pm_eb_lower_bound(selected_float, alpha / 4.0)
        for epsilon in epsilons:
            monitor = base_monitor.with_inference_parameters(
                alpha=alpha, epsilon=epsilon
            )
            calibration = monitor.calibration
            high_error_lower = np.maximum(
                0.0,
                target_lower - calibration.source_false_discovery_joint_upper,
            )
            alarm = np.maximum.accumulate(
                high_error_lower
                > calibration.source_selected_high_error_upper + epsilon
            )
            event = (
                running_selected_high
                > calibration.source_selected_high_error_rate + epsilon
            )
            sample = pd.DataFrame(
                {BATCH_INDEX: batch_index, "alarm": alarm, "event": event}
            )
            batch = sample.groupby(BATCH_INDEX, sort=True).agg(
                alarm=("alarm", "last"), event=("event", "last")
            )
            metrics = compute_alarm_event_metrics(
                batch["alarm"].astype(bool).to_numpy(),
                batch["event"].astype(bool).to_numpy(),
            )
            records.append(
                {
                    **common,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "evaluated": True,
                    **metrics,
                    "stream_any_event": bool(batch["event"].any()),
                    "stream_any_alarm": bool(batch["alarm"].any()),
                    "max_oracle_assumption_4_1_gap": float(
                        np.max(assumption_gap)
                    ),
                    "final_running_selected_rate": float(running_selected[-1]),
                    "final_target_selected_lower": float(target_lower[-1]),
                }
            )
    return pd.DataFrame(records)


def _write_status(
    output: Path,
    requested: int,
    completed: int,
    configurations: int,
    smoke: bool,
    stopped: bool,
) -> None:
    atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "complete": completed == requested and not stopped,
            "smoke_test": smoke,
            "requested_predictor_streams": requested,
            "completed_predictor_streams": completed,
            "sensitivity_configurations": configurations,
            "stopped_for_time": stopped,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output / "shd_sensitivity_status.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = _load_index(cache, args.expected_seeds)
    if args.datasets:
        index = index[index["dataset"].isin(args.datasets)]
    if args.models:
        index = index[index["model"].isin(args.models)]
    if index.empty:
        raise ValueError("SHD sensitivity subset is empty")
    if args.smoke_test:
        first = index.iloc[0]
        index = index[
            index["dataset"].eq(first.dataset)
            & index["model"].eq(first.model)
            & index["seed"].eq(first.seed)
            & index["shift"].isin(["no_shift", "pipeline"])
        ].head(2)
    observable = ObservableCacheReader(cache)
    oracle = OracleCacheReader(cache)
    monitors = _fit_or_load_monitors(
        observable, index, output / "monitor_checkpoints"
    )
    calibration = []
    for (dataset, model), base in monitors.items():
        for alpha in args.alphas:
            for epsilon in args.epsilons:
                configured = base.with_inference_parameters(
                    alpha=alpha, epsilon=epsilon
                )
                calibration.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        **configured.calibration_record(),
                    }
                )
    atomic_parquet(
        pd.DataFrame(calibration), output / "shd_sensitivity_calibration.parquet"
    )

    checkpoints = output / "scenario_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stopped = False
    for position, row in enumerate(index.itertuples(index=False), start=1):
        path = checkpoints / f"{row.predictor_stream_id}.parquet"
        if not path.is_file():
            frame = _evaluate_stream(
                row,
                observable,
                oracle,
                monitors[(row.dataset, row.model)],
                list(args.alphas),
                list(args.epsilons),
            )
            atomic_parquet(frame, path)
        if position == 1 or position % 50 == 0 or position == len(index):
            completed = sum(1 for _ in checkpoints.glob("*.parquet"))
            _write_status(
                output,
                len(index),
                completed,
                len(args.alphas) * len(args.epsilons),
                args.smoke_test,
                False,
            )
            print(f"[{position}/{len(index)}] streams={completed}", flush=True)
        if args.max_hours is not None and (time.monotonic() - started) / 3600 >= args.max_hours:
            stopped = True
            break

    paths = sorted(checkpoints.glob("*.parquet"))
    scenarios = pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True
    )
    atomic_parquet(scenarios, output / "shd_sensitivity_metrics.parquet")
    completed = len(paths)
    _write_status(
        output,
        len(index),
        completed,
        len(args.alphas) * len(args.epsilons),
        args.smoke_test,
        stopped,
    )
    atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "method": "shd_quantile_phi2_pm_eb",
            "selector": "fixed published FDP<0.20 quantile selector",
            "alphas": list(args.alphas),
            "epsilons": list(args.epsilons),
            "fit_reuse": (
                "one source error-regression and selector fit per dataset/model; "
                "only inference boundaries vary"
            ),
            "oracle_use": "evaluation events and diagnostics only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output / "shd_sensitivity_manifest.json",
    )
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
