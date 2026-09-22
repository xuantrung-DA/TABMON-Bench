import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# B1, B2, B3, B4: Các base models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# B6: Probability Calibration
from sklearn.calibration import CalibratedClassifierCV

# B5: Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

class BaseModelsTrainer:
    def __init__(self, dataset_name: str, data_dir: str, results_dir: str):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, dataset_name)
        
        # Tạo thư mục lưu trữ B5
        self.model_dir = os.path.join(results_dir, 'models', dataset_name)
        self.artifacts_dir = os.path.join(results_dir, 'artifacts', dataset_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Map tên cột target tương ứng với từng dataset
        self.target_cols = {
            'adult': 'income',
            'bank_marketing': 'y',
            'covertype': 'Cover_Type',
            'acs_income': 'PINCP'
        }
        self.target_col = self.target_cols[dataset_name]

    def load_data(self):
        """Tải các split đã tạo từ Phase 1"""
        self.train_df = pd.read_parquet(os.path.join(self.data_dir, 'train.parquet'))
        self.calib_df = pd.read_parquet(os.path.join(self.data_dir, 'calibration.parquet'))
        self.test_df = pd.read_parquet(os.path.join(self.data_dir, 'test_pool.parquet'))
        
        self.X_train, self.y_train = self.train_df.drop(columns=[self.target_col]), self.train_df[self.target_col]
        self.X_calib, self.y_calib = self.calib_df.drop(columns=[self.target_col]), self.calib_df[self.target_col]
        self.X_test, self.y_test = self.test_df.drop(columns=[self.target_col]), self.test_df[self.target_col]

    def get_models(self):
        """Khởi tạo 4 thuật toán theo yêu cầu"""
        return {
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
            # Thiết lập tree_method='hist' tối ưu cho CPU. Nếu muốn test GPU RTX 4050, đổi thành device='cuda'
            'xgb': XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, tree_method='hist', random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
        }

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in self.X_train.columns if c not in numeric_features]

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ])
        return ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, numeric_features),
                ('categorical', categorical_pipeline, categorical_features),
            ],
            sparse_threshold=1.0,
        )

    @staticmethod
    def _calibrate_prefit(model, X_calib, y_calib):
        """Support both old and new scikit-learn prefit calibration APIs."""
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
        except ImportError:
            calibrated = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit')
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
            
            # Huấn luyện mô hình gốc trên tập Train
            model_pipeline = Pipeline([
                ('preprocessor', clone(preprocessor)),
                ('estimator', model),
            ])
            model_pipeline.fit(self.X_train, self.y_train)
            
            # B6: Thực hiện Probability Calibration trên tập Calib bằng Isotonic Regression
            # Sử dụng cv='prefit' để báo cho sklearn biết mô hình gốc đã được train
            calibrated_model = self._calibrate_prefit(model_pipeline, self.X_calib, self.y_calib)
            
            # Lưu mô hình đã hiệu chuẩn
            joblib.dump(calibrated_model, os.path.join(self.model_dir, f'{model_name}_calibrated.pkl'))
            
            # Dự đoán trên tập Test Pool
            probs = calibrated_model.predict_proba(self.X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            # Tính Logits (bảo vệ khỏi cảnh báo chia cho 0 khi prob = 0 hoặc 1)
            eps = 1e-7
            safe_probs = np.clip(probs, eps, 1 - eps)
            logits = logit(safe_probs)
            
            # B5: Lưu Prediction, Probability và Logits thành dạng Parquet để tiết kiệm SSD
            artifacts_df = pd.DataFrame({
                'true_label': self.y_test.values,
                'prediction': preds,
                'probability': probs,
                'logit': logits
            })
            artifacts_df.to_parquet(os.path.join(self.artifacts_dir, f'{model_name}_artifacts.parquet'), index=False)
            
            # B5: Tính toán Metrics chuyên sâu cho monitoring
            acc = accuracy_score(self.y_test, preds)
            auc = roc_auc_score(self.y_test, probs)
            nll = log_loss(self.y_test, probs)
            brier = brier_score_loss(self.y_test, probs) # Brier score rất quan trọng để đo độ chuẩn của Calibration
            
            metrics_list.append({
                'dataset': self.dataset_name,
                'model': model_name.upper(),
                'accuracy': acc,
                'auc': auc,
                'log_loss': nll,
                'brier_score': brier
            })
            
        return pd.DataFrame(metrics_list)
