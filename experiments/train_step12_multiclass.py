"""Train and calibrate the four frozen model families on 7-class Covertype."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _models() -> dict[str, object]:
    return {
        "lr": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
        ),
        "xgb": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            objective="multi:softprob",
            num_class=7,
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(_models()), default=list(_models()))
    args = parser.parse_args(argv)
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    model_dir = output / "models" / "covertype_multiclass"
    model_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(data / "train.parquet")
    calibration = pd.read_parquet(data / "calibration.parquet")
    test = pd.read_parquet(data / "test_pool.parquet")
    target = "Cover_Type"
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_cal, y_cal = calibration.drop(columns=[target]), calibration[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X_train if column not in numeric]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )
    records = []
    for name in args.models:
        path = model_dir / f"{name}_calibrated.pkl"
        if path.is_file():
            calibrated = joblib.load(path)
        else:
            print(f"Training multiclass {name.upper()}...", flush=True)
            pipeline = Pipeline(
                [("preprocessor", clone(preprocessor)), ("estimator", _models()[name])]
            )
            pipeline.fit(X_train, y_train)
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(pipeline), method="isotonic"
            ).fit(X_cal, y_cal)
            temporary = path.with_suffix(path.suffix + ".tmp")
            joblib.dump(calibrated, temporary)
            os.replace(temporary, path)
        probabilities = calibrated.predict_proba(X_test)
        prediction = calibrated.classes_[np.argmax(probabilities, axis=1)]
        records.append(
            {
                "dataset": "covertype_multiclass",
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "macro_f1": f1_score(y_test, prediction, average="macro"),
                "log_loss": log_loss(y_test, probabilities, labels=calibrated.classes_),
            }
        )
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "base_model_metrics.csv", index=False)
    manifest = {
        "dataset": "covertype_multiclass",
        "models": args.models,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
