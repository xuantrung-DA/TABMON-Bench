"""Prepare the original seven-class Covertype task for Step 12."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_data import _read_covertype, optimize_dtypes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "covertype_multiclass",
    )
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame, source = _read_covertype()
    # XGBoost requires contiguous zero-based class indices.
    frame["Cover_Type"] = frame["Cover_Type"].astype(np.int8) - 1
    classes = sorted(frame["Cover_Type"].unique().tolist())
    if classes != list(range(7)):
        raise ValueError(f"Expected Covertype classes 0..6, got {classes}")
    frame = optimize_dtypes(frame)
    train_calibration, test = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame["Cover_Type"],
        random_state=42,
    )
    train, calibration = train_test_split(
        train_calibration,
        test_size=2.0 / 7.0,
        stratify=train_calibration["Cover_Type"],
        random_state=42,
    )
    splits = {
        "train": train.reset_index(drop=True),
        "calibration": calibration.reset_index(drop=True),
        "test_pool": test.reset_index(drop=True),
    }
    for name, split in splits.items():
        split.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    manifest = {
        "dataset": "covertype_multiclass",
        "source": source,
        "target": "Cover_Type",
        "classes": classes,
        "task": "seven_class_classification",
        "split_seed": 42,
        "split_strategy": "target_stratified_random",
        "rows": {name: len(split) for name, split in splits.items()},
        "class_counts": {
            name: split["Cover_Type"].value_counts().sort_index().to_dict()
            for name, split in splits.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
