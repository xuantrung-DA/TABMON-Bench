"""Freeze TABMON-Bench schema-v7 output and the matching manuscript baseline.

The command never edits manuscript sources.  It records SHA-256 digests for
the original Kaggle archive and every archive member, creates a separate
snapshot of the compilable manuscript directory, and emits an auditable
meta-monitor feature allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.meta_monitor_schema import meta_monitor_allowlist_audit


EXCLUDED_MANUSCRIPT_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _find_member(names: list[str], filename: str) -> str:
    matches = [name for name in names if Path(name).name == filename]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} in archive; found {matches}")
    return matches[0]


def inspect_results_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        members = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as stream:
                member_hash = _sha256_stream(stream)
            members.append(
                {
                    "path": info.filename,
                    "sha256": member_hash,
                    "size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )

        manifest_name = _find_member(names, "benchmark_manifest.json")
        status_name = _find_member(names, "benchmark_status.json")
        batch_name = _find_member(names, "batch_metrics.parquet")
        aggregate_name = _find_member(names, "aggregate_metrics.parquet")
        benchmark_manifest = json.loads(archive.read(manifest_name))
        benchmark_status = json.loads(archive.read(status_name))
        batches = pd.read_parquet(io.BytesIO(archive.read(batch_name)))
        aggregate = pd.read_parquet(io.BytesIO(archive.read(aggregate_name)))

    if int(benchmark_manifest.get("schema_version", -1)) != 7:
        raise ValueError("Expected a schema-v7 benchmark archive")
    if not benchmark_status.get("complete"):
        raise ValueError("Refusing to freeze an incomplete benchmark")
    if int(benchmark_status.get("errors_this_run", -1)) != 0:
        raise ValueError("Refusing to freeze a benchmark with run errors")
    if len(aggregate) != 6_200 or len(batches) != 62_000:
        raise ValueError(
            "Unexpected v7 table dimensions: "
            f"aggregate={len(aggregate)}, batches={len(batches)}"
        )
    if aggregate["scenario_id"].nunique() != 6_200:
        raise ValueError("Aggregate table does not contain 6,200 unique evaluations")
    if batches["scenario_id"].nunique() != 6_200:
        raise ValueError("Batch table does not contain 6,200 unique evaluations")

    return {
        "artifact_role": "immutable schema-v7 Kaggle output baseline",
        "archive_name": path.name,
        "archive_size_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "schema_version": 7,
        "complete": True,
        "errors_this_run": 0,
        "monitor_scenario_evaluations": len(aggregate),
        "monitor_batch_records": len(batches),
        "predictor_stream_scenarios": len(aggregate) // 2,
        "members": members,
        "embedded_benchmark_manifest": benchmark_manifest,
        "embedded_benchmark_status": benchmark_status,
    }


def _is_manuscript_source(path: Path) -> bool:
    lower_name = path.name.lower()
    return not any(lower_name.endswith(suffix) for suffix in EXCLUDED_MANUSCRIPT_SUFFIXES)


def freeze_manuscript(manuscript_dir: Path, archive_path: Path) -> dict[str, Any]:
    required = [manuscript_dir / "main.tex", manuscript_dir / "references.bib"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if archive_path.exists():
        raise FileExistsError(
            f"Baseline already exists and will not be overwritten: {archive_path}"
        )

    files = sorted(
        path
        for path in manuscript_dir.rglob("*")
        if path.is_file() and _is_manuscript_source(path)
    )
    file_records = [
        {
            "path": path.relative_to(manuscript_dir).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    with zipfile.ZipFile(
        archive_path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(manuscript_dir).as_posix())

    return {
        "artifact_role": "immutable schema-v7 manuscript source baseline",
        "source_directory_name": manuscript_dir.name,
        "archive_name": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "source_files": file_records,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_zip = args.results_zip.resolve()
    manuscript_dir = args.manuscript_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not results_zip.is_file():
        raise FileNotFoundError(results_zip)
    if not manuscript_dir.is_dir():
        raise NotADirectoryError(manuscript_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manuscript_archive = output_dir / "manuscript_schema_v7_baseline.zip"
    protected_outputs = [
        output_dir / "output_v7_checksum_manifest.json",
        output_dir / "manuscript_v7_baseline_manifest.json",
        output_dir / "meta_monitor_feature_allowlist.json",
        output_dir / "schema_v7_freeze_manifest.json",
        manuscript_archive,
    ]
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Freeze outputs already exist and are immutable: " + ", ".join(existing)
        )

    output_manifest = inspect_results_archive(results_zip)
    manuscript_manifest = freeze_manuscript(manuscript_dir, manuscript_archive)
    allowlist_manifest = meta_monitor_allowlist_audit()
    created_at = datetime.now(timezone.utc).isoformat()
    output_manifest["frozen_at_utc"] = created_at
    manuscript_manifest["frozen_at_utc"] = created_at
    allowlist_manifest["frozen_at_utc"] = created_at

    output_path = output_dir / "output_v7_checksum_manifest.json"
    manuscript_path = output_dir / "manuscript_v7_baseline_manifest.json"
    allowlist_path = output_dir / "meta_monitor_feature_allowlist.json"
    _write_json(output_path, output_manifest)
    _write_json(manuscript_path, manuscript_manifest)
    _write_json(allowlist_path, allowlist_manifest)
    master = {
        "freeze_id": "tabmon-schema-v7",
        "frozen_at_utc": created_at,
        "immutable_outputs": {
            path.name: sha256_file(path)
            for path in (
                results_zip,
                manuscript_archive,
                output_path,
                manuscript_path,
                allowlist_path,
            )
        },
    }
    _write_json(output_dir / "schema_v7_freeze_manifest.json", master)

    print("=== TABMON SCHEMA-V7 FREEZE ===")
    print(f"Results SHA-256: {output_manifest['archive_sha256']}")
    print(f"Manuscript SHA-256: {manuscript_manifest['archive_sha256']}")
    print(f"Saved at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
