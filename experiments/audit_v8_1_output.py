"""Fail-closed audit for the schema-v8.1 oracle-only rescore."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


CONFIDENCE_METHOD = "confidence_log_loss"
ALLOWED_CONFIDENCE_CHANGES = {
    "true_value",
    "signed_error",
    "absolute_error",
    "risk_failure",
    "alarm_target_event",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--parent-v8-dir", type=Path, required=True)
    parser.add_argument("--v7-dir", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _same(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    try:
        assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise AssertionError(f"Unexpected v8.1 change in {label}: {exc}") from exc


def main() -> int:
    args = parse_args()
    root = args.results_dir.resolve()
    parent_root = args.parent_v8_dir.resolve()
    status = json.loads((root / "v8_1_status.json").read_text("utf-8"))
    manifest = json.loads((root / "v8_1_manifest.json").read_text("utf-8"))
    if status.get("complete") is not True or status.get("schema_version") != "8.1":
        raise AssertionError("v8.1 status is not complete")
    if manifest.get("operation") != "oracle-only post-hoc rescore":
        raise AssertionError("Unexpected v8.1 operation")
    for name, expected in manifest["parent_file_sha256"].items():
        if _sha256(parent_root / name) != expected:
            raise AssertionError(f"Parent v8 file changed after rescore: {name}")

    corrected = pd.read_parquet(root / "batch_metrics.parquet")
    parent = pd.read_parquet(parent_root / "batch_metrics.parquet")
    scenarios = pd.read_parquet(root / "scenario_metrics.parquet")
    parent_scenarios = pd.read_parquet(parent_root / "scenario_metrics.parquet")
    if len(corrected) != 217_000 or len(scenarios) != 21_700:
        raise AssertionError("v8.1 output grid is incomplete")
    keys = ["predictor_stream_id", "_tabmon_batch_index", "method"]
    _same(parent[keys], corrected[keys], "row identity/order")

    confidence = corrected["method"].eq(CONFIDENCE_METHOD)
    _same(
        parent.loc[~confidence],
        corrected.loc[~confidence],
        "non-Confidence batch records",
    )
    immutable = [
        column for column in corrected.columns if column not in ALLOWED_CONFIDENCE_CHANGES
    ]
    _same(
        parent.loc[confidence, immutable],
        corrected.loc[confidence, immutable],
        "Confidence monitor outputs",
    )
    scenario_confidence = scenarios["method"].eq(CONFIDENCE_METHOD)
    _same(
        parent_scenarios.loc[~scenario_confidence],
        scenarios.loc[~scenario_confidence],
        "non-Confidence scenario summaries",
    )

    c = corrected.loc[confidence]
    signed = c["estimated_value"].to_numpy(float) - c["true_value"].to_numpy(float)
    if not np.allclose(c["signed_error"], signed, atol=1e-12, rtol=0):
        raise AssertionError("Corrected signed errors are inconsistent")
    if not np.allclose(c["absolute_error"], np.abs(signed), atol=1e-12, rtol=0):
        raise AssertionError("Corrected absolute errors are inconsistent")
    if not np.array_equal(
        c["alarm_target_event"].astype(bool).to_numpy(),
        c["true_value"].gt(0.05).to_numpy(),
    ):
        raise AssertionError("Corrected risk-event flags are inconsistent")

    v7_max_abs_difference = None
    if args.v7_dir is not None:
        v7 = pd.read_parquet(args.v7_dir.resolve() / "batch_metrics.parquet")
        v7 = v7[v7["monitor"].eq("confidence")][
            [
                "dataset", "model", "shift", "severity", "mode", "seed",
                "batch_index", "true_excess_risk",
            ]
        ].rename(columns={"batch_index": "_tabmon_batch_index"})
        current = c[
            [
                "dataset", "model", "shift", "severity", "mode", "seed",
                "_tabmon_batch_index", "true_value",
            ]
        ]
        join_keys = [
            "dataset", "model", "shift", "severity", "mode", "seed",
            "_tabmon_batch_index",
        ]
        paired = current.merge(v7, on=join_keys, validate="one_to_one")
        v7_max_abs_difference = float(
            np.max(np.abs(paired["true_value"] - paired["true_excess_risk"]))
        )
        if v7_max_abs_difference > 2e-6:
            raise AssertionError(
                "Corrected risk is inconsistent with frozen v7: "
                f"max difference {v7_max_abs_difference}"
            )

    old_c = parent.loc[confidence]
    event_changes = int(
        np.sum(
            old_c["alarm_target_event"].astype(bool).to_numpy()
            != c["alarm_target_event"].astype(bool).to_numpy()
        )
    )
    summary = {
        "audit_passed": True,
        "batch_method_records": len(corrected),
        "method_stream_evaluations": len(scenarios),
        "corrected_confidence_batches": int(confidence.sum()),
        "risk_event_flags_changed": event_changes,
        "v7_max_abs_risk_difference": v7_max_abs_difference,
        "monitor_outputs_unchanged": True,
        "non_confidence_rows_unchanged": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
