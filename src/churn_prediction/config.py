"""Central configuration for Telco Churn Prediction System."""

from pathlib import Path
from typing import Dict, List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "churn_pipeline_metadata.json"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Target specification
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TARGET_MAPPING: Dict[str, int] = {"No": 0, "Yes": 1}
INVERSE_TARGET_MAPPING: Dict[int, str] = {0: "Not Likely to Churn", 1: "Likely to Churn"}

# Raw Feature Columns
NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Engineered feature names
ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "service_count",
]

ENGINEERED_CATEGORICAL_FEATURES: List[str] = [
    "tenure_group",
]

# Service features evaluated for service_count
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

