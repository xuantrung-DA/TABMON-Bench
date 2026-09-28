"""Repair float32 argmax ties in a completed TABMON stream-cache archive.

No model inference or shift generation is performed.  The command derives
``predicted_class`` from the exact stored float32 probability columns, rewrites
only prediction Parquet members that differ, and copies all other members
byte-for-byte into a new archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPAIR_REVISION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_prefix(names: list[str]) -> str:
    matches = [name for name in names if name.endswith("cache_status.json")]
    if len(matches) != 1:
        raise ValueError(f"Expected one cache_status.json; found {matches}")
    return matches[0][: -len("cache_status.json")]


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    return buffer.getvalue()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def repair_archive(input_zip: Path, output_zip: Path) -> dict[str, Any]:
    input_zip = input_zip.resolve()
    output_zip = output_zip.resolve()
    if input_zip == output_zip:
        raise ValueError("Output ZIP must differ from input ZIP")
    if output_zip.exists():
        raise FileExistsError(output_zip)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    repaired_at = datetime.now(timezone.utc).isoformat()

    try:
        with zipfile.ZipFile(input_zip) as source:
            names = source.namelist()
            if source.testzip() is not None:
                raise ValueError("Input archive failed CRC validation")
            root = _root_prefix(names)
            status = json.loads(source.read(root + "cache_status.json"))
            manifest = json.loads(source.read(root + "cache_manifest.json"))
            if int(status.get("cache_schema_version", -1)) != 1:
                raise ValueError("Repair requires cache schema version 1")
            if not status.get("complete"):
                raise ValueError("Repair requires a complete cache")

            predictor_index = pd.read_parquet(
                io.BytesIO(
                    source.read(root + "control/predictor_stream_index.parquet")
                )
            )
            classes_by_member = {
                root + str(row.prediction_path).replace("\\", "/"): np.asarray(
                    json.loads(row.classes_json)
                )
                for row in predictor_index.itertuples(index=False)
            }
            prediction_members = set(classes_by_member)
            missing = sorted(prediction_members - set(names))
            if missing:
                raise ValueError(f"Missing prediction members: {missing[:3]}")

            repaired_files = 0
            repaired_rows = 0
            repairs: list[dict[str, Any]] = []
            with zipfile.ZipFile(
                temporary,
                "x",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as target:
                for info in source.infolist():
                    name = info.filename
                    if name in {
                        root + "cache_status.json",
                        root + "cache_manifest.json",
                        root + "repair_report.json",
                    }:
                        continue
                    payload = source.read(name)
                    if name in prediction_members:
                        frame = pd.read_parquet(io.BytesIO(payload))
                        probability_columns = sorted(
                            [
                                column
                                for column in frame
                                if column.startswith("probability_class_")
                            ],
                            key=lambda column: int(column.rsplit("_", 1)[1]),
                        )
                        if not probability_columns:
                            raise ValueError(f"No probabilities in {name}")
                        probabilities = frame[probability_columns].to_numpy(
                            dtype=np.float32
                        )
                        classes = classes_by_member[name]
                        expected = classes[np.argmax(probabilities, axis=1)]
                        current = frame["predicted_class"].to_numpy()
                        mismatch = np.flatnonzero(current != expected)
                        if len(mismatch):
                            frame["predicted_class"] = expected
                            payload = _parquet_bytes(frame)
                            repaired_files += 1
                            repaired_rows += len(mismatch)
                            repairs.append(
                                {
                                    "prediction_member": name,
                                    "repaired_rows": int(len(mismatch)),
                                    "row_positions": mismatch.astype(int).tolist(),
                                }
                            )
                    target.writestr(name, payload)

                report = {
                    "repair_revision": REPAIR_REVISION,
                    "repair_type": "float32_probability_argmax_consistency",
                    "repaired_at_utc": repaired_at,
                    "source_archive": input_zip.name,
                    "source_sha256": sha256_file(input_zip),
                    "prediction_files_scanned": len(prediction_members),
                    "prediction_files_repaired": repaired_files,
                    "prediction_rows_repaired": repaired_rows,
                    "model_inference_rerun": False,
                    "shift_generation_rerun": False,
                    "probabilities_modified": False,
                    "oracle_values_modified": False,
                    "repairs": repairs,
                }
                manifest["cache_revision"] = REPAIR_REVISION
                manifest["repair"] = {
                    "type": report["repair_type"],
                    "repaired_at_utc": repaired_at,
                    "prediction_rows_repaired": repaired_rows,
                    "probabilities_modified": False,
                    "oracle_values_modified": False,
                }
                status["cache_revision"] = REPAIR_REVISION
                status["repaired_at_utc"] = repaired_at
                status["prediction_rows_repaired"] = repaired_rows
                target.writestr(root + "cache_manifest.json", _json_bytes(manifest))
                target.writestr(root + "cache_status.json", _json_bytes(status))
                target.writestr(root + "repair_report.json", _json_bytes(report))
        os.replace(temporary, output_zip)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    if zipfile.ZipFile(output_zip).testzip() is not None:
        raise RuntimeError("Repaired archive failed CRC validation")
    report["output_archive"] = output_zip.name
    report["output_sha256"] = sha256_file(output_zip)
    report["output_size_bytes"] = output_zip.stat().st_size
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = repair_archive(args.input_zip, args.output_zip)
    print("=== TABMON STREAM CACHE REPAIR ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
