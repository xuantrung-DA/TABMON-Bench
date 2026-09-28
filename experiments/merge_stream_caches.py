"""Merge disjoint seed partitions of TABMON stream cache v1.1 safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache.stream_cache import atomic_json, atomic_parquet


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    status = json.loads((root / "cache_status.json").read_text("utf-8"))
    manifest = json.loads((root / "cache_manifest.json").read_text("utf-8"))
    streams = pd.read_parquet(root / "control" / "stream_index.parquet")
    predictors = pd.read_parquet(
        root / "control" / "predictor_stream_index.parquet"
    )
    if status.get("complete") is not True:
        raise ValueError(f"Incomplete cache cannot be merged: {root}")
    if int(status.get("cache_schema_version", -1)) != 1:
        raise ValueError(f"Unsupported cache schema: {root}")
    return status, manifest, streams, predictors


def _json_equivalent(left: Path, right: Path) -> bool:
    first = json.loads(left.read_text("utf-8"))
    second = json.loads(right.read_text("utf-8"))
    if first == second:
        return True
    if set(first) != set(second):
        return False
    for key in first:
        left_value, right_value = first[key], second[key]
        if key == "reference_log_loss":
            if not np.isclose(left_value, right_value, rtol=1e-8, atol=1e-10):
                return False
        elif left_value != right_value:
            return False
    return True


def _parquet_equivalent(left: Path, right: Path) -> bool:
    first = pd.read_parquet(left)
    second = pd.read_parquet(right)
    try:
        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=False,
            check_categorical=False,
            check_exact=False,
            rtol=1e-7,
            atol=1e-9,
        )
    except AssertionError:
        return False
    return True


def _shared_file_equivalent(left: Path, right: Path) -> bool:
    if _digest(left) == _digest(right):
        return True
    if left.suffix == ".json" and right.suffix == ".json":
        return _json_equivalent(left, right)
    if left.suffix == ".parquet" and right.suffix == ".parquet":
        return _parquet_equivalent(left, right)
    return False


def _copy_tree_checked(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "control" in path.relative_to(source).parts:
            continue
        if path.name in {"cache_status.json", "cache_manifest.json"}:
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not _shared_file_equivalent(path, destination):
                raise ValueError(f"Conflicting cache file: {relative}")
            # Preserve the reference artifact from the frozen base cache.
        else:
            shutil.copy2(path, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--extension-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = [args.base_cache.resolve(), args.extension_cache.resolve()]
    output = args.output_dir.resolve()
    if output in roots:
        parser.error("output-dir must differ from both input caches")

    loaded = [_load(root) for root in roots]
    stream_frames = [item[2] for item in loaded]
    predictor_frames = [item[3] for item in loaded]
    first_keys = set(stream_frames[0]["stream_id"])
    second_keys = set(stream_frames[1]["stream_id"])
    if first_keys & second_keys:
        raise ValueError("Input cache partitions contain overlapping stream IDs")
    for source in roots:
        _copy_tree_checked(source, output)

    streams = pd.concat(stream_frames, ignore_index=True).sort_values(
        ["dataset", "seed", "shift", "severity", "mode"]
    )
    predictors = pd.concat(predictor_frames, ignore_index=True).sort_values(
        ["dataset", "model", "stream_id"]
    )
    if streams["stream_id"].duplicated().any():
        raise RuntimeError("Merged stream IDs are not unique")
    if predictors["predictor_stream_id"].duplicated().any():
        raise RuntimeError("Merged predictor stream IDs are not unique")
    atomic_parquet(streams, output / "control" / "stream_index.parquet")
    atomic_parquet(
        predictors, output / "control" / "predictor_stream_index.parquet"
    )

    seeds = sorted(int(value) for value in streams["seed"].unique())
    status = {
        "cache_schema_version": 1,
        "cache_revision": 1,
        "complete": True,
        "requested_streams": len(streams),
        "completed_streams": len(streams),
        "completed_predictor_streams": len(predictors),
        "stopped_for_time": False,
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest = dict(loaded[0][1])
    manifest["cache_revision"] = 1
    manifest["purpose"] = "ten-seed reusable streams and model probabilities"
    manifest["configuration"] = dict(manifest["configuration"])
    manifest["configuration"]["seeds"] = seeds
    manifest["merge"] = {
        "source_roots": [root.name for root in roots],
        "disjoint_stream_ids": True,
        "merged_streams": len(streams),
        "merged_predictor_streams": len(predictors),
    }
    atomic_json(status, output / "cache_status.json")
    atomic_json(manifest, output / "cache_manifest.json")
    print("=== TABMON TEN-SEED CACHE ===")
    print("Seeds:", seeds)
    print("Streams:", len(streams))
    print("Predictor streams:", len(predictors))
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
