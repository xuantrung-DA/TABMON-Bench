"""Audit a packaged TABMON stream cache without extracting it."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache.stream_cache import FORBIDDEN_OBSERVABLE_COLUMNS


def _root_prefix(names: list[str]) -> str:
    matches = [name for name in names if name.endswith("cache_status.json")]
    if len(matches) != 1:
        raise ValueError(f"Expected one cache_status.json; found {matches}")
    return matches[0][: -len("cache_status.json")]


def audit_archive(path: Path, compare_source: Path | None = None) -> dict[str, Any]:
    with zipfile.ZipFile(path.resolve()) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"CRC failure: {corrupt_member}")
        names = archive.namelist()
        root = _root_prefix(names)
        status = json.loads(archive.read(root + "cache_status.json"))
        if not status.get("complete"):
            raise ValueError("Cache status is not complete")

        predictor_index = pd.read_parquet(
            io.BytesIO(
                archive.read(root + "control/predictor_stream_index.parquet")
            )
        )
        mismatch_rows = 0
        prediction_rows = 0
        maximum_probability_sum_error = 0.0
        forbidden_hits: dict[str, list[str]] = {}

        for row in predictor_index.itertuples(index=False):
            member = root + str(row.prediction_path).replace("\\", "/")
            frame = pd.read_parquet(io.BytesIO(archive.read(member)))
            forbidden = sorted(set(frame.columns) & FORBIDDEN_OBSERVABLE_COLUMNS)
            if forbidden:
                forbidden_hits[member] = forbidden
            probability_columns = sorted(
                [c for c in frame if c.startswith("probability_class_")],
                key=lambda c: int(c.rsplit("_", 1)[1]),
            )
            probabilities = frame[probability_columns].to_numpy(dtype=np.float32)
            classes = np.asarray(json.loads(row.classes_json))
            expected = classes[np.argmax(probabilities, axis=1)]
            mismatch_rows += int(np.count_nonzero(frame["predicted_class"] != expected))
            prediction_rows += len(frame)
            maximum_probability_sum_error = max(
                maximum_probability_sum_error,
                float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
            )

        if forbidden_hits:
            raise ValueError(f"Oracle columns found in predictions: {forbidden_hits}")
        if mismatch_rows:
            raise ValueError(
                f"Found {mismatch_rows} predicted-class/float32-argmax mismatches"
            )

        repair_member = root + "repair_report.json"
        repair = json.loads(archive.read(repair_member)) if repair_member in names else None
        result = {
            "archive": path.name,
            "cache_schema_version": status.get("cache_schema_version"),
            "cache_revision": status.get("cache_revision", 0),
            "complete": bool(status.get("complete")),
            "archive_members": len(names),
            "predictor_streams": len(predictor_index),
            "prediction_rows": prediction_rows,
            "predicted_class_argmax_mismatches": mismatch_rows,
            "maximum_probability_sum_error": maximum_probability_sum_error,
            "observable_prediction_leakage_hits": len(forbidden_hits),
            "repair_report_present": repair is not None,
            "prediction_rows_repaired": (
                repair.get("prediction_rows_repaired") if repair else 0
            ),
        }
        if compare_source is not None:
            if repair is None:
                raise ValueError("Source comparison requires a repair report")
            with zipfile.ZipFile(compare_source.resolve()) as source:
                source_root = _root_prefix(source.namelist())
                allowed_changes = {
                    root + "cache_status.json",
                    root + "cache_manifest.json",
                    root + "repair_report.json",
                    *[item["prediction_member"] for item in repair["repairs"]],
                }
                target_infos = {item.filename: item for item in archive.infolist()}
                unexpected = []
                unchanged = 0
                for source_info in source.infolist():
                    target_name = root + source_info.filename[len(source_root) :]
                    target_info = target_infos.get(target_name)
                    if target_info is None:
                        unexpected.append(source_info.filename + " (missing)")
                    elif target_name not in allowed_changes and (
                        source_info.CRC != target_info.CRC
                        or source_info.file_size != target_info.file_size
                    ):
                        unexpected.append(source_info.filename)
                    elif target_name not in allowed_changes:
                        unchanged += 1

                for item in repair["repairs"]:
                    target_name = item["prediction_member"]
                    source_name = source_root + target_name[len(root) :]
                    before = pd.read_parquet(io.BytesIO(source.read(source_name)))
                    after = pd.read_parquet(io.BytesIO(archive.read(target_name)))
                    columns = [c for c in before if c != "predicted_class"]
                    pd.testing.assert_frame_equal(before[columns], after[columns])

                if unexpected:
                    raise ValueError(f"Unexpected changed members: {unexpected[:5]}")
                result["source_archive"] = compare_source.name
                result["unchanged_source_members"] = unchanged
                result["unexpected_changed_members"] = len(unexpected)
                result["repaired_file_nonclass_columns_unchanged"] = True
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--compare-source", type=Path)
    args = parser.parse_args()
    print("=== TABMON STREAM CACHE ARCHIVE AUDIT ===")
    print(
        json.dumps(
            audit_archive(args.input_zip, compare_source=args.compare_source),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
