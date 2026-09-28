"""Prepare local raw datasets for TABMON-Bench training.

Preparation is local-first: no remote API is called. Raw files are validated,
converted to a common binary target, split reproducibly, and saved as Parquet.
"""

import argparse
import gzip
import json
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from folktables import ACSIncome
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.50
CALIBRATION_FRACTION = 0.20
TEST_FRACTION = 0.30

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

COVERTYPE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    *[f"Wilderness_Area{i}" for i in range(1, 5)],
    *[f"Soil_Type{i}" for i in range(1, 41)],
    "Cover_Type",
]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast columns without mistaking pandas string dtypes for floats."""
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if is_bool_dtype(series.dtype):
            df[col] = series.astype(np.int8)
        elif is_integer_dtype(series.dtype):
            df[col] = pd.to_numeric(series, downcast="integer")
        elif is_float_dtype(series.dtype):
            df[col] = pd.to_numeric(series, downcast="float")
        elif not is_numeric_dtype(series.dtype):
            df[col] = series.astype("category")
    return df


def _validate_dataset(df: pd.DataFrame, target_col: str, dataset_name: str) -> None:
    if df.empty:
        raise ValueError(f"{dataset_name}: dataset is empty")
    if target_col not in df.columns:
        raise ValueError(f"{dataset_name}: target column '{target_col}' is missing")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{dataset_name}: duplicate columns: {duplicates}")

    target_values = set(pd.Series(df[target_col]).dropna().unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(
            f"{dataset_name}: target must contain both binary classes, got {target_values}"
        )
    if df[target_col].isna().any():
        raise ValueError(f"{dataset_name}: target contains missing values")


def generate_splits_and_save(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    source: str,
    groups: pd.Series | None = None,
) -> None:
    """Create stratified train/calibration/test splits and save Parquet files."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = optimize_dtypes(df)
    _validate_dataset(df, target_col, dataset_name)

    print(f"\n--- Preparing {dataset_name} ---")
    print(f"[*] Source: {source}")
    print(f"[*] Rows: {len(df):,}; columns: {len(df.columns)}")
    missing_features = int(df.drop(columns=[target_col]).isna().sum().sum())
    print(f"[*] Missing feature values: {missing_features:,}")
    print(f"[*] Positive rate: {float(df[target_col].mean()):.4f}")

    calibration_within_remainder = CALIBRATION_FRACTION / (
        TRAIN_FRACTION + CALIBRATION_FRACTION
    )
    split_strategy = "stratified_random"
    split_group_counts = None
    if groups is None:
        train_calib, test = train_test_split(
            df,
            test_size=TEST_FRACTION,
            stratify=df[target_col],
            random_state=RANDOM_SEED,
        )
        train, calibration = train_test_split(
            train_calib,
            test_size=calibration_within_remainder,
            stratify=train_calib[target_col],
            random_state=RANDOM_SEED,
        )
    else:
        groups = pd.Series(groups).reset_index(drop=True)
        if len(groups) != len(df):
            raise ValueError(f"{dataset_name}: groups must match dataframe length")
        df = df.reset_index(drop=True)
        outer = GroupShuffleSplit(
            n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED
        )
        train_calib_idx, test_idx = next(outer.split(df, df[target_col], groups))
        train_calib = df.iloc[train_calib_idx]
        test = df.iloc[test_idx]
        train_calib_groups = groups.iloc[train_calib_idx]
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=calibration_within_remainder,
            random_state=RANDOM_SEED,
        )
        train_idx, calibration_idx = next(
            inner.split(train_calib, train_calib[target_col], train_calib_groups)
        )
        train = train_calib.iloc[train_idx]
        calibration = train_calib.iloc[calibration_idx]
        absolute_train_idx = train_calib_idx[train_idx]
        absolute_calibration_idx = train_calib_idx[calibration_idx]
        train_groups = set(groups.iloc[absolute_train_idx])
        calibration_groups = set(groups.iloc[absolute_calibration_idx])
        test_groups = set(groups.iloc[test_idx])
        if (
            train_groups & calibration_groups
            or train_groups & test_groups
            or calibration_groups & test_groups
        ):
            raise RuntimeError(f"{dataset_name}: group leakage detected between splits")
        split_strategy = "patient_group_disjoint"
        split_group_counts = {
            "train": len(train_groups),
            "calibration": len(calibration_groups),
            "test_pool": len(test_groups),
        }

    output_dir = PROC_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train,
        "calibration": calibration,
        "test_pool": test,
    }
    for split_name, split_df in splits.items():
        output_path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        print(f"[+] {split_name:11s}: {len(split_df):>8,} rows -> {output_path}")

    manifest = {
        "dataset": dataset_name,
        "source": source,
        "target": target_col,
        "random_seed": RANDOM_SEED,
        "split_strategy": split_strategy,
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "calibration": CALIBRATION_FRACTION,
            "test_pool": TEST_FRACTION,
        },
        "rows": {name: len(split_df) for name, split_df in splits.items()},
        "positive_rate": float(df[target_col].mean()),
        "columns": list(df.columns),
    }
    if split_group_counts is not None:
        manifest["group_counts"] = split_group_counts
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def load_adult() -> None:
    adult_dir = RAW_DIR / "adult"
    train_path = adult_dir / "adult.data"
    test_path = adult_dir / "adult.test"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Adult raw files are missing: {missing}")

    read_options = {
        "names": ADULT_COLUMNS,
        "sep": ",",
        "skipinitialspace": True,
        "na_values": ["?"],
    }
    train = pd.read_csv(train_path, **read_options)
    test = pd.read_csv(test_path, skiprows=1, **read_options)
    df = pd.concat([train, test], ignore_index=True)
    normalized_target = df["income"].astype("string").str.strip().str.rstrip(".")
    df["income"] = normalized_target.map({"<=50K": 0, ">50K": 1})

    generate_splits_and_save(
        df,
        target_col="income",
        dataset_name="adult",
        source=f"{train_path}; {test_path}",
    )


def load_bank_marketing() -> None:
    source_path = RAW_DIR / "bank_marketing" / "bank-full.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Bank Marketing raw file is missing: {source_path}")

    df = pd.read_csv(source_path, sep=";")
    df["y"] = df["y"].map({"no": 0, "yes": 1})
    generate_splits_and_save(
        df,
        target_col="y",
        dataset_name="bank_marketing",
        source=str(source_path),
    )


def _read_covertype() -> tuple[pd.DataFrame, str]:
    covertype_dir = RAW_DIR / "covertype"
    direct_candidates = [
        covertype_dir / "covtype.data",
        covertype_dir / "covtype.data.gz",
    ]
    for source_path in direct_candidates:
        if source_path.is_file():
            return (
                pd.read_csv(source_path, header=None, names=COVERTYPE_COLUMNS),
                str(source_path),
            )

    zip_candidates = [covertype_dir / "covertype.zip", covertype_dir / "covtype.zip"]
    for source_path in zip_candidates:
        if not source_path.is_file():
            continue
        with zipfile.ZipFile(source_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(("covtype.data", "covtype.data.gz"))
            ]
            if not members:
                raise ValueError(f"No covtype.data file found inside {source_path}")
            member = members[0]
            with archive.open(member) as raw_handle:
                if member.lower().endswith(".gz"):
                    with gzip.GzipFile(fileobj=raw_handle) as handle:
                        df = pd.read_csv(handle, header=None, names=COVERTYPE_COLUMNS)
                else:
                    df = pd.read_csv(raw_handle, header=None, names=COVERTYPE_COLUMNS)
        return df, f"{source_path}!{member}"

    raise FileNotFoundError(
        "Covertype is not downloaded. Expected data/raw/covertype/"
        "covtype.data, covtype.data.gz, covertype.zip, or covtype.zip"
    )


def load_covertype() -> None:
    df, source = _read_covertype()
    df["Cover_Type"] = (df["Cover_Type"] == 2).astype(np.int8)
    generate_splits_and_save(
        df,
        target_col="Cover_Type",
        dataset_name="covertype",
        source=source,
    )


def load_folktables_acs() -> None:
    candidates = [
        RAW_DIR / "acs_income" / "psam_p06.csv",
        RAW_DIR / "2018" / "1-Year" / "psam_p06.csv",
    ]
    source_path = next((path for path in candidates if path.is_file()), None)
    if source_path is None:
        raise FileNotFoundError(
            "ACS California file is missing. Expected data/raw/acs_income/"
            "psam_p06.csv or data/raw/2018/1-Year/psam_p06.csv"
        )

    required_columns = list(dict.fromkeys([*ACSIncome.features, "PINCP", "PWGTP"]))
    acs_data = pd.read_csv(source_path, usecols=required_columns, low_memory=False)
    features, labels, _ = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df["PINCP"] = labels.astype(np.int8)

    generate_splits_and_save(
        df,
        target_col="PINCP",
        dataset_name="acs_income",
        source=str(source_path),
    )


def load_diabetes_hospitals() -> None:
    source_path = RAW_DIR / "diabetes" / "diabetic_data.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Diabetes Hospitals raw file is missing: {source_path}")

    df = pd.read_csv(source_path, na_values=["?"], low_memory=False)
    groups = df["patient_nbr"].copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(np.int8)
    df = df.drop(columns=["readmitted", "encounter_id", "patient_nbr", "weight"])

    # These integer values are identifiers for nominal categories, not quantities.
    for column in (
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ):
        df[column] = df[column].astype("string")

    constant_columns = [
        column
        for column in df.columns
        if column != "readmitted_30d" and df[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        df = df.drop(columns=constant_columns)

    generate_splits_and_save(
        df,
        target_col="readmitted_30d",
        dataset_name="diabetes_hospitals",
        source=str(source_path),
        groups=groups,
    )


DATASET_LOADERS: dict[str, Callable[[], None]] = {
    "adult": load_adult,
    "bank_marketing": load_bank_marketing,
    "acs_income": load_folktables_acs,
    "covertype": load_covertype,
    "diabetes_hospitals": load_diabetes_hospitals,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TABMON-Bench datasets")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(DATASET_LOADERS),
        help="Datasets to prepare; defaults to every available local dataset",
    )
    args = parser.parse_args()
    requested = args.datasets or list(DATASET_LOADERS)

    print("=== TABMON-Bench: preparing local datasets ===")
    prepared = []
    missing = []
    for dataset_name in requested:
        try:
            DATASET_LOADERS[dataset_name]()
            prepared.append(dataset_name)
        except FileNotFoundError as exc:
            print(f"[SKIP] {dataset_name}: {exc}")
            missing.append(dataset_name)

    if not prepared:
        print("[ERROR] No dataset was prepared.")
        return 1
    print(f"\n[DONE] Prepared: {', '.join(prepared)}")
    if missing:
        print(f"[INFO] Missing local data: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
