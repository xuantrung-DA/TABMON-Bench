"""Fail-closed integrity and leakage audit for TABMON Step 10 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_OBSERVABLE_TOKENS = (
    "target",
    "true",
    "oracle",
    "risk",
    "failure",
    "shift",
    "severity",
    "mode",
    "seed",
    "ground_truth",
    "intervention",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step10-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--v8-1-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    root = args.step10_dir.resolve()
    cache = args.cache_dir.resolve()
    v8 = args.v8_1_dir.resolve()
    status = json.loads((root / "step10_status.json").read_text("utf-8"))
    manifest = json.loads((root / "step10_manifest.json").read_text("utf-8"))
    _require(status.get("complete") is True and status.get("step") == 10, "Incomplete Step 10")
    _require(manifest.get("step") == 10, "Wrong Step 10 manifest")

    input_paths = {
        "cache_manifest.json": cache / "cache_manifest.json",
        "cache_status.json": cache / "cache_status.json",
        "stream_index.parquet": cache / "control" / "stream_index.parquet",
        "predictor_stream_index.parquet": cache / "control" / "predictor_stream_index.parquet",
        "v8_1_manifest.json": v8 / "v8_1_manifest.json",
        "v8_1_status.json": v8 / "v8_1_status.json",
        "batch_metrics.parquet": v8 / "batch_metrics.parquet",
    }
    for name, path in input_paths.items():
        _require(
            _sha256(path) == manifest["input_sha256"][name],
            f"Frozen input changed: {name}",
        )

    thresholds = pd.read_parquet(root / "schema_guard_calibration_thresholds.parquet")
    maxima = pd.read_parquet(root / "schema_guard_calibration_maxima.parquet")
    guard = pd.read_parquet(root / "guard_batch_scores.parquet")
    offline = pd.read_parquet(root / "guard_batch_offline_evaluation.parquet")
    scenarios = pd.read_parquet(root / "guard_scenario_metrics.parquet")
    joined = pd.read_parquet(root / "guarded_confidence_batch_metrics.parquet")
    counts = pd.read_parquet(root / "pipeline_guard_scenario_counts.parquet")
    summaries = pd.read_csv(root / "guarded_confidence_summary.csv")

    _require(len(thresholds) == 5, "Expected one threshold per dataset")
    _require(len(maxima) == 1000, "Expected 200 maxima for each of five datasets")
    _require(len(guard) == 7750 and guard["stream_id"].nunique() == 775, "Guard grid incomplete")
    _require(len(offline) == 7750, "Offline guard grid incomplete")
    _require(len(scenarios) == 775, "Guard scenario grid incomplete")
    _require(len(joined) == 31_000, "Guarded Confidence grid incomplete")
    _require(len(counts) == 600, "Pipeline predictor-stream grid incomplete")

    expected_columns = manifest["observable_output_columns"]
    _require(list(guard.columns) == expected_columns, "Observable schema differs from allowlist")
    lowered = [column.lower() for column in guard.columns]
    leaked = [
        column
        for column in lowered
        if any(token in column for token in FORBIDDEN_OBSERVABLE_TOKENS)
    ]
    _require(not leaked, f"Forbidden offline field in guard output: {leaked}")
    _require(
        _sha256(root / "guard_batch_scores.parquet")
        == manifest["observable_guard_sha256_before_oracle_access"],
        "Observable guard output changed after oracle access",
    )
    _require(
        guard["guard_alarm"].eq(guard["guard_score"].gt(guard["guard_threshold"])).all(),
        "Alarm decisions do not match strict calibrated threshold",
    )
    _require(guard["guard_p_value"].between(0, 1).all(), "Invalid guard p-values")
    _require(
        np.isfinite(guard[["guard_score", "guard_threshold", "guard_p_value"]]).all().all(),
        "Non-finite observable guard score",
    )

    for row in counts.itertuples(index=False):
        _require(row.caught_reversals + row.unsafe_unflagged == row.reversed_batches, "Reversal accounting failed")
        _require(row.guard_alarms <= row.active_batches, "Invalid abstention count")
        _require(row.reversed_batches <= row.harmful_batches, "Invalid harmful count")
        _require(row.unsafe_unflagged <= row.unflagged_harmful, "Invalid selective count")

    overall = summaries[summaries["scope"].eq("overall")].set_index("metric")
    expected_ratios = {
        "raw_sign_reversal_rate": counts["reversed_batches"].sum() / counts["harmful_batches"].sum(),
        "reversal_capture_rate": counts["caught_reversals"].sum() / counts["reversed_batches"].sum(),
        "residual_unsafe_rate": counts["unsafe_unflagged"].sum() / counts["harmful_batches"].sum(),
        "selective_sign_reversal_rate": counts["unsafe_unflagged"].sum() / counts["unflagged_harmful"].sum(),
        "guard_abstention_rate": counts["guard_alarms"].sum() / counts["active_batches"].sum(),
    }
    for metric, expected in expected_ratios.items():
        actual = float(overall.loc[metric, "estimate"])
        _require(abs(actual - expected) < 1e-12, f"Summary mismatch: {metric}")
    _require(
        manifest.get("guard_role")
        == "abstention/escalation signal; not a corrected risk estimate",
        "Guard role is overstated",
    )

    result = {
        "audit_passed": True,
        "input_hashes_verified": True,
        "observable_allowlist_verified": True,
        "observable_hash_frozen_before_oracle": True,
        "guard_streams": int(guard["stream_id"].nunique()),
        "guard_batches": len(guard),
        "guarded_confidence_batches": len(joined),
        "pipeline_predictor_streams": len(counts),
        "ratio_summaries_recomputed": True,
        "abstention_only_claim_verified": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

