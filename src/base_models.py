import os

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class BaseModelsTrainer:
    def __init__(self, dataset_name: str, data_dir: str, results_dir: str):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, dataset_name)

        self.model_dir = os.path.join(results_dir, "models", dataset_name)
        self.artifacts_dir = os.path.join(results_dir, "artifacts", dataset_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.target_cols = {
            "adult": "income",
            "bank_marketing": "y",
            "covertype": "Cover_Type",
            "acs_income": "PINCP",
            "diabetes_hospitals": "readmitted_30d",
        }
        self.target_col = self.target_cols[dataset_name]

    def load_data(self):
        """Load the frozen train, calibration, and test-pool splits."""

        self.train_df = pd.read_parquet(os.path.join(self.data_dir, "train.parquet"))
        self.calib_df = pd.read_parquet(
            os.path.join(self.data_dir, "calibration.parquet")
        )
        self.test_df = pd.read_parquet(
            os.path.join(self.data_dir, "test_pool.parquet")
        )

        self.X_train = self.train_df.drop(columns=[self.target_col])
        self.y_train = self.train_df[self.target_col]
        self.X_calib = self.calib_df.drop(columns=[self.target_col])
        self.y_calib = self.calib_df[self.target_col]
        self.X_test = self.test_df.drop(columns=[self.target_col])
        self.y_test = self.test_df[self.target_col]

    def get_models(self):
        """Construct the four frozen predictor families."""

        return {
            "lr": LogisticRegression(max_iter=1000, random_state=42),
            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                n_jobs=-1,
                random_state=42,
            ),
            "xgb": XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                random_state=42,
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                early_stopping=True,
                random_state=42,
            ),
        }

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_features = self.X_train.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_features = [
            column
            for column in self.X_train.columns
            if column not in numeric_features
        ]

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ],
            sparse_threshold=1.0,
        )

    @staticmethod
    def _calibrate_prefit(model, X_calib, y_calib):
        """Support both old and new scikit-learn prefit calibration APIs."""
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(
                FrozenEstimator(model), method="isotonic"
            )
        except ImportError:
            calibrated = CalibratedClassifierCV(
                estimator=model, method="isotonic", cv="prefit"
            )
        calibrated.fit(X_calib, y_calib)
        return calibrated

    def train_and_calibrate(self, model_names=None):
        self.load_data()
        models = self.get_models()
        if model_names is not None:
            unknown = set(model_names) - set(models)
            if unknown:
                raise ValueError(f"Unknown model names: {sorted(unknown)}")
            models = {name: models[name] for name in model_names}
        metrics_list = []
        preprocessor = self._build_preprocessor()

        for model_name, model in models.items():
            print(f"  -> Training {model_name.upper()}...")

            model_pipeline = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("estimator", model),
                ]
            )
            model_pipeline.fit(self.X_train, self.y_train)

            calibrated_model = self._calibrate_prefit(
                model_pipeline, self.X_calib, self.y_calib
            )
            joblib.dump(
                calibrated_model,
                os.path.join(self.model_dir, f"{model_name}_calibrated.pkl"),
            )

            probs = calibrated_model.predict_proba(self.X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            eps = 1e-7
            safe_probs = np.clip(probs, eps, 1 - eps)
            logits = logit(safe_probs)

            artifacts_df = pd.DataFrame(
                {
                    "true_label": self.y_test.values,
                    "prediction": preds,
                    "probability": probs,
                    "logit": logits,
                }
            )
            artifacts_df.to_parquet(
                os.path.join(
                    self.artifacts_dir, f"{model_name}_artifacts.parquet"
                ),
                index=False,
            )

            acc = accuracy_score(self.y_test, preds)
            auc = roc_auc_score(self.y_test, probs)
            nll = log_loss(self.y_test, probs)
            brier = brier_score_loss(self.y_test, probs)

            metrics_list.append(
                {
                    "dataset": self.dataset_name,
                    "model": model_name.upper(),
                    "accuracy": acc,
                    "auc": auc,
                    "log_loss": nll,
                    "brier_score": brier,
                }
            )

        return pd.DataFrame(metrics_list)
