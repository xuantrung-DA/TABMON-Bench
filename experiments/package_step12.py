"""Create a Kaggle-safe Step-12 source ZIP with POSIX member names."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


INCLUDE_DIRECTORIES = ("configs", "experiments", "src", "tests")
INCLUDE_FILES = (
    "README.md",
    "requirements.txt",
    "KAGGLE_STEP12_Q1_EXTENSION.md",
    "KAGGLE_STEP12_FIDELITY_RERUN.md",
    "KAGGLE_FINAL_THREE_STAGES.md",
    "METHOD_FIDELITY_SHD_XPE.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--package-name", default="tabmon-step12-q1-v7-final")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_zip.resolve()
    members: list[tuple[Path, str]] = []
    for directory in INCLUDE_DIRECTORIES:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(root).as_posix()
                members.append((path, f"Empirical_Benchmark/{relative}"))
    for filename in INCLUDE_FILES:
        path = root / filename
        if path.is_file():
            members.append((path, f"Empirical_Benchmark/{filename}"))
    processed = root / "data" / "processed"
    required_datasets = {
        "acs_income",
        "adult",
        "bank_marketing",
        "covertype",
        "covertype_multiclass",
        "diabetes_hospitals",
    }
    present_datasets = {path.name for path in processed.iterdir() if path.is_dir()}
    if not required_datasets <= present_datasets:
        raise FileNotFoundError(
            f"Missing processed datasets: {sorted(required_datasets - present_datasets)}"
        )
    for dataset in sorted(required_datasets):
        for path in sorted((processed / dataset).glob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                members.append((path, f"Empirical_Benchmark/{relative}"))
    if not members:
        raise RuntimeError("No Step-12 package members found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in members:
            if "\\" in arcname:
                raise ValueError(f"Forbidden ZIP member separator: {arcname}")
            archive.write(path, arcname)
        package_manifest = {
            "package": args.package_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "member_count": len(members),
            "contains_all_processed_data": True,
            "contains_raw_data": False,
            "contains_manuscript": False,
        }
        archive.writestr(
            "Empirical_Benchmark/step12_package_manifest.json",
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        )
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        if any("\\" in name for name in archive.namelist()):
            raise RuntimeError("ZIP contains a forbidden backslash member")
    print("Created:", output)
    print(f"Size: {output.stat().st_size / 1024**2:.2f} MiB")
    print("SHA256:", _sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
