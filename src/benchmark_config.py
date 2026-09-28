"""Shared dataset/model/shift configuration for TABMON-Bench."""

DATASETS = [
    "adult",
    "bank_marketing",
    "acs_income",
    "covertype",
    "diabetes_hospitals",
]
MODELS = ["lr", "rf", "xgb", "mlp"]
MONITORS = ["drift_shap", "confidence"]
SHIFT_FAMILIES = [
    "no_shift",
    "covariate",
    "correlated",
    "support",
    "pipeline",
    "concept",
]
SEVERITIES = ["low", "medium", "high"]
MODES = ["abrupt", "gradual", "static"]

DATASET_CONFIG = {
    "adult": {
        "target": "income",
        "shift_feature": "age",
        "corr_features": ["age", "education-num"],
    },
    "bank_marketing": {
        "target": "y",
        "shift_feature": "balance",
        "corr_features": ["balance", "duration"],
    },
    "acs_income": {
        "target": "PINCP",
        "shift_feature": "AGEP",
        "corr_features": ["AGEP", "WKHP"],
    },
    "covertype": {
        "target": "Cover_Type",
        "shift_feature": "Elevation",
        "corr_features": ["Elevation", "Slope"],
    },
    "diabetes_hospitals": {
        "target": "readmitted_30d",
        "shift_feature": "time_in_hospital",
        "corr_features": ["num_medications", "num_lab_procedures"],
    },
}
