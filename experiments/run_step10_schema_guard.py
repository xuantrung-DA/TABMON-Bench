"""Step 10: evaluate a label-free schema guard on frozen TABMON streams.

The observable guard pass is completed and persisted before this runner opens
the offline oracle or the v8.1 confidence results.  The guard is an abstention
signal, not a corrected estimate of predictive risk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_phase1_v7 import cluster_bootstrap_ratio
from src.baselines.schema_guard import SchemaGuardProfile
from src.cache.stream_cache import (
    BATCH_INDEX,
    ObservableCacheReader,
    OracleCacheReader,
    atomic_json,
    atomic_parquet,
)
from src.evaluation.alarm_calibration import stream_max_threshold
from src.evaluation.metrics import compute_alarm_event_metrics


OBSERVABLE_OUTPUT_COLUMNS = (
    "stream_id",
    "dataset",
    BATCH_INDEX,
    "guard_score",
    "guard_alarm",
    "guard_threshold",
    "guard_p_value",
    "top_feature",
    "top_component",
    "feature_scores_json",
    "component_scores_json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--v8-1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--null-trajectories", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260926)
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie between zero and one")
    if args.null_trajectories < 100:
        parser.error("Use at least 100 independent null trajectories")
    if args.bootstrap_replicates < 1000:
        parser.error("Use at least 1,000 bootstrap replicates")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _load_inputs(
    cache_root: Path, v8_root: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cache_status = json.loads((cache_root / "cache_status.json").read_text("utf-8"))
    cache_manifest = json.loads(
        (cache_root / "cache_manifest.json").read_text("utf-8")
    )
    v8_status = json.loads((v8_root / "v8_1_status.json").read_text("utf-8"))
    if cache_status.get("complete") is not True or cache_status.get("cache_revision", 0) < 1:
        raise ValueError("Step 10 requires the complete repaired stream cache")
    if v8_status.get("complete") is not True or v8_status.get("schema_version") != "8.1":
        raise ValueError("Step 10 requires complete schema-v8.1 results")
    streams = pd.read_parquet(cache_root / "control" / "stream_index.parquet")
    predictors = pd.read_parquet(
        cache_root / "control" / "predictor_stream_index.parquet"
    )
    if len(streams) != 775 or streams["stream_id"].nunique() != 775:
        raise ValueError("Stream cache grid is incomplete")
    if len(predictors) != 3100 or predictors["predictor_stream_id"].nunique() != 3100:
        raise ValueError("Predictor-stream cache grid is incomplete")
    return cache_manifest, streams, predictors


def _calibrate_guard(
    observable: ObservableCacheReader,
    streams: pd.DataFrame,
    alpha: float,
    null_trajectories: int,
) -> tuple[dict[str, SchemaGuardProfile], pd.DataFrame, pd.DataFrame]:
    profiles: dict[str, SchemaGuardProfile] = {}
    threshold_rows: list[dict[str, Any]] = []
    maximum_rows: list[dict[str, Any]] = []
    for dataset in sorted(streams["dataset"].unique()):
        row = streams.loc[streams["dataset"].eq(dataset)].iloc[0]
        num_batches = int(row["num_batches"])
        batch_size = int(row["batch_size"])
        reference_X = observable.load_reference_features(dataset)
        profile = SchemaGuardProfile.fit(reference_X)
        profiles[dataset] = profile
        rng = np.random.default_rng(_stable_seed("step10-null", dataset))
        positions = rng.integers(
            0,
            len(reference_X),
            size=(null_trajectories * num_batches, batch_size),
            dtype=np.int32,
        )
        scores = profile.score_position_batches(reference_X, positions)
        threshold, maxima, rank = stream_max_threshold(
            scores, null_trajectories, num_batches, alpha
        )
        threshold_rows.append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "threshold": threshold,
                "conformal_rank": rank,
                "null_trajectories": null_trajectories,
                "num_batches": num_batches,
                "batch_size": batch_size,
                "reference_rows": len(reference_X),
            }
        )
        maximum_rows.extend(
            {
                "dataset": dataset,
                "trajectory_index": index,
                "maximum_score": float(value),
            }
            for index, value in enumerate(maxima)
        )
        print(
            f"[calibration] {dataset}: threshold={threshold:.6g}, rank={rank}",
            flush=True,
        )
    return profiles, pd.DataFrame(threshold_rows), pd.DataFrame(maximum_rows)


def _observable_guard_pass(
    observable: ObservableCacheReader,
    streams: pd.DataFrame,
    profiles: dict[str, SchemaGuardProfile],
    thresholds: pd.DataFrame,
    maxima: pd.DataFrame,
) -> pd.DataFrame:
    threshold_map = thresholds.set_index("dataset")["threshold"].to_dict()
    maximum_map = {
        dataset: group["maximum_score"].to_numpy(dtype=float)
        for dataset, group in maxima.groupby("dataset", sort=False)
    }
    records: list[dict[str, Any]] = []
    for number, row in enumerate(streams.itertuples(index=False), start=1):
        target_X = observable.load_target_features(row.stream_id)
        threshold = float(threshold_map[row.dataset])
        null_maxima = maximum_map[row.dataset]
        for batch_index in range(int(row.num_batches)):
            start = batch_index * int(row.batch_size)
            stop = start + int(row.batch_size)
            report = profiles[row.dataset].score_batch(target_X.iloc[start:stop])
            p_value = float(
                (1 + np.count_nonzero(null_maxima >= report.score))
                / (len(null_maxima) + 1)
            )
            records.append(
                {
                    "stream_id": row.stream_id,
                    "dataset": row.dataset,
                    BATCH_INDEX: batch_index,
                    "guard_score": report.score,
                    "guard_alarm": bool(report.score > threshold),
                    "guard_threshold": threshold,
                    "guard_p_value": p_value,
                    "top_feature": report.top_feature,
                    "top_component": report.top_component,
                    "feature_scores_json": report.feature_scores_json,
                    "component_scores_json": report.component_scores_json,
                }
            )
        if number % 100 == 0 or number == len(streams):
            print(f"[observable guard] {number}/{len(streams)} streams", flush=True)
    result = pd.DataFrame(records, columns=OBSERVABLE_OUTPUT_COLUMNS)
    if len(result) != 7750:
        raise RuntimeError(f"Expected 7,750 guard batches; found {len(result)}")
    return result


def _parse_active_features(payload: str) -> set[str]:
    values = json.loads(payload)
    return {str(name) for name, value in values.items() if float(value) > 0.0}


def _offline_guard_evaluation(
    guard: pd.DataFrame,
    streams: pd.DataFrame,
    oracle: OracleCacheReader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptors = streams[
        ["stream_id", "dataset", "shift", "severity", "mode", "seed"]
    ]
    evaluated = guard.merge(descriptors, on=["stream_id", "dataset"], validate="many_to_one")
    intervention_parts = []
    for row in streams.itertuples(index=False):
        part = oracle.load_intervention_targets(row.stream_id).copy()
        part["stream_id"] = row.stream_id
        intervention_parts.append(part)
    interventions = pd.concat(intervention_parts, ignore_index=True)
    evaluated = evaluated.merge(
        interventions,
        on=["stream_id", BATCH_INDEX],
        validate="one_to_one",
    )
    evaluated["controlled_event"] = (
        evaluated["shift"].ne("no_shift") & evaluated["shift_fraction"].gt(0.0)
    )
    active_features = evaluated["ground_truth_attribution_json"].map(
        _parse_active_features
    )
    evaluated["attribution_target_active"] = active_features.map(bool)
    evaluated["top1_intervention_recall"] = [
        float(top in active) if active else np.nan
        for top, active in zip(evaluated["top_feature"], active_features)
    ]

    scenario_rows: list[dict[str, Any]] = []
    for stream_id, group in evaluated.groupby("stream_id", sort=False):
        group = group.sort_values(BATCH_INDEX)
        first = group.iloc[0]
        alarm_metrics = compute_alarm_event_metrics(
            group["guard_alarm"].tolist(), group["controlled_event"].tolist()
        )
        scenario_rows.append(
            {
                "stream_id": stream_id,
                "dataset": first["dataset"],
                "shift": first["shift"],
                "severity": first["severity"],
                "mode": first["mode"],
                "seed": int(first["seed"]),
                "mean_guard_score": float(group["guard_score"].mean()),
                "max_guard_score": float(group["guard_score"].max()),
                "any_guard_alarm": bool(group["guard_alarm"].any()),
                "active_batch_alarm_rate": float(
                    group.loc[group["controlled_event"], "guard_alarm"].mean()
                )
                if group["controlled_event"].any()
                else np.nan,
                "top1_intervention_recall": float(
                    group["top1_intervention_recall"].mean()
                )
                if group["attribution_target_active"].any()
                else np.nan,
                **alarm_metrics,
            }
        )
    return evaluated, pd.DataFrame(scenario_rows)


def _guarded_confidence_evaluation(
    evaluated_guard: pd.DataFrame,
    predictors: pd.DataFrame,
    v8_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batches = pd.read_parquet(v8_root / "batch_metrics.parquet")
    confidence = batches.loc[
        batches["method"].eq("confidence_log_loss"),
        [
            "predictor_stream_id",
            "stream_id",
            "dataset",
            "model",
            "shift",
            "severity",
            "mode",
            "seed",
            BATCH_INDEX,
            "shift_fraction",
            "true_value",
            "estimated_value",
        ],
    ].copy()
    if len(confidence) != 31_000:
        raise ValueError("Schema-v8.1 Confidence grid is incomplete")
    guard_columns = [
        "stream_id",
        BATCH_INDEX,
        "guard_score",
        "guard_alarm",
        "guard_p_value",
        "top_feature",
        "top_component",
    ]
    joined = confidence.merge(
        evaluated_guard[guard_columns],
        on=["stream_id", BATCH_INDEX],
        validate="many_to_one",
    )
    joined["harmful"] = joined["true_value"].gt(0.05)
    joined["sign_reversal"] = joined["harmful"] & joined["estimated_value"].lt(0.0)
    joined["guard_caught_reversal"] = joined["sign_reversal"] & joined["guard_alarm"]
    joined["unsafe_unflagged_reversal"] = joined["sign_reversal"] & ~joined["guard_alarm"]
    joined["unflagged_harmful"] = joined["harmful"] & ~joined["guard_alarm"]

    pipeline = joined[
        joined["shift"].eq("pipeline") & joined["shift_fraction"].gt(0.0)
    ].copy()
    keys = [
        "predictor_stream_id",
        "stream_id",
        "dataset",
        "model",
        "severity",
        "mode",
        "seed",
    ]
    counts = (
        pipeline.groupby(keys, dropna=False)
        .agg(
            active_batches=(BATCH_INDEX, "size"),
            harmful_batches=("harmful", "sum"),
            reversed_batches=("sign_reversal", "sum"),
            caught_reversals=("guard_caught_reversal", "sum"),
            unsafe_unflagged=("unsafe_unflagged_reversal", "sum"),
            unflagged_harmful=("unflagged_harmful", "sum"),
            guard_alarms=("guard_alarm", "sum"),
        )
        .reset_index()
    )
    return joined, counts


def _ratio_rows(
    counts: pd.DataFrame,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    metrics = {
        "raw_sign_reversal_rate": ("reversed_batches", "harmful_batches"),
        "reversal_capture_rate": ("caught_reversals", "reversed_batches"),
        "residual_unsafe_rate": ("unsafe_unflagged", "harmful_batches"),
        "selective_sign_reversal_rate": ("unsafe_unflagged", "unflagged_harmful"),
        "guard_abstention_rate": ("guard_alarms", "active_batches"),
    }
    scopes: list[tuple[str, str, pd.DataFrame, tuple[str, ...]]] = [
        ("overall", "all", counts, ("dataset", "seed"))
    ]
    for column in ("model", "dataset", "severity", "mode"):
        for value, group in counts.groupby(column, sort=True):
            clusters = ("seed",) if column == "dataset" else ("dataset", "seed")
            scopes.append((column, str(value), group, clusters))
    rows = []
    for scope, level, frame, clusters in scopes:
        for metric, (numerator, denominator) in metrics.items():
            rng = np.random.default_rng(_stable_seed(random_seed, scope, level, metric))
            estimate, lower, upper = cluster_bootstrap_ratio(
                frame,
                numerator,
                denominator,
                clusters,
                replicates,
                rng,
            )
            rows.append(
                {
                    "scope": scope,
                    "level": level,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "numerator_total": int(frame[numerator].sum()),
                    "denominator_total": int(frame[denominator].sum()),
                    "scenario_count": len(frame),
                    "cluster_columns": "+".join(clusters),
                }
            )
    return pd.DataFrame(rows)


def _guard_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shift, group in scenarios.groupby("shift", sort=True):
        rows.append(
            {
                "shift": shift,
                "scenario_count": len(group),
                "any_alarm_rate": float(group["any_guard_alarm"].mean()),
                "mean_active_batch_alarm_rate": float(
                    group["active_batch_alarm_rate"].mean()
                ),
                "mean_top1_intervention_recall": float(
                    group["top1_intervention_recall"].mean()
                ),
                "missed_event_rate": float(group["missed_alarm"].mean()),
                "mean_detection_delay": float(group["detection_delay"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = args.cache_dir.resolve()
    v8_root = args.v8_1_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_manifest, streams, predictors = _load_inputs(cache_root, v8_root)
    observable = ObservableCacheReader(cache_root)

    profiles, thresholds, maxima = _calibrate_guard(
        observable, streams, args.alpha, args.null_trajectories
    )
    guard = _observable_guard_pass(
        observable, streams, profiles, thresholds, maxima
    )
    # This file is the immutable monitor-facing result.  It is persisted before
    # the first OracleCacheReader is instantiated or any v8.1 result is opened.
    atomic_parquet(thresholds, output / "schema_guard_calibration_thresholds.parquet")
    atomic_parquet(maxima, output / "schema_guard_calibration_maxima.parquet")
    atomic_parquet(guard, output / "guard_batch_scores.parquet")
    observable_hash = _sha256(output / "guard_batch_scores.parquet")

    oracle = OracleCacheReader(cache_root)
    evaluated_guard, guard_scenarios = _offline_guard_evaluation(
        guard, streams, oracle
    )
    guarded_confidence, pipeline_counts = _guarded_confidence_evaluation(
        evaluated_guard, predictors, v8_root
    )
    guard_summary = _guard_summary(guard_scenarios)
    confidence_summary = _ratio_rows(
        pipeline_counts, args.bootstrap_replicates, args.random_seed
    )

    atomic_parquet(evaluated_guard, output / "guard_batch_offline_evaluation.parquet")
    atomic_parquet(guard_scenarios, output / "guard_scenario_metrics.parquet")
    atomic_parquet(
        guarded_confidence, output / "guarded_confidence_batch_metrics.parquet"
    )
    atomic_parquet(
        pipeline_counts, output / "pipeline_guard_scenario_counts.parquet"
    )
    guard_summary.to_csv(output / "schema_guard_summary.csv", index=False)
    confidence_summary.to_csv(
        output / "guarded_confidence_summary.csv", index=False
    )

    input_files = {
        "cache_manifest.json": cache_root / "cache_manifest.json",
        "cache_status.json": cache_root / "cache_status.json",
        "stream_index.parquet": cache_root / "control" / "stream_index.parquet",
        "predictor_stream_index.parquet": cache_root
        / "control"
        / "predictor_stream_index.parquet",
        "v8_1_manifest.json": v8_root / "v8_1_manifest.json",
        "v8_1_status.json": v8_root / "v8_1_status.json",
        "batch_metrics.parquet": v8_root / "batch_metrics.parquet",
    }
    manifest = {
        "step": 10,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "label-free schema guard and guarded-confidence evaluation",
        "guard_role": "abstention/escalation signal; not a corrected risk estimate",
        "alpha": args.alpha,
        "null_trajectories_per_dataset": args.null_trajectories,
        "bootstrap_replicates": args.bootstrap_replicates,
        "risk_event_threshold": 0.05,
        "observable_guard_sha256_before_oracle_access": observable_hash,
        "observable_output_columns": list(OBSERVABLE_OUTPUT_COLUMNS),
        "observable_inputs": ["source calibration features", "target batch features"],
        "prohibited_guard_inputs": [
            "target labels",
            "oracle risk",
            "failure labels",
            "shift identity",
            "shift severity",
            "generator shift fraction",
            "intervention ground truth",
        ],
        "input_sha256": {name: _sha256(path) for name, path in input_files.items()},
        "cache_revision": cache_manifest.get("cache_revision"),
    }
    status = {
        "step": 10,
        "complete": True,
        "guard_streams": int(guard["stream_id"].nunique()),
        "guard_batch_records": len(guard),
        "guard_scenario_records": len(guard_scenarios),
        "guarded_confidence_batch_records": len(guarded_confidence),
        "pipeline_predictor_streams": len(pipeline_counts),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest, output / "step10_manifest.json")
    atomic_json(status, output / "step10_status.json")

    overall = confidence_summary[
        confidence_summary["scope"].eq("overall")
    ][["metric", "estimate", "ci_lower", "ci_upper"]]
    print("\n=== STEP 10 COMPLETE ===")
    print(status)
    print("\nPipeline guard headline:")
    print(overall.to_string(index=False))
    print("\nGuard by shift:")
    print(guard_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

