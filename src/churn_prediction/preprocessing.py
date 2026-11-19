"""Leakage-safe Scikit-Learn preprocessing pipelines and column transformers."""

from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_prediction.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
)
from src.churn_prediction.features import TelcoFeatureEngineer
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)

# Complete list of features after feature engineering
ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def build_column_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes, encodes, and optionally scales.
    
    Args:
        scale_numeric: If True, applies StandardScaler to numeric features (for Logistic Regression).
    
    Returns:
        Configured ColumnTransformer fitted only inside pipelines.
    """
    # Numeric pipeline
    num_steps: List[Tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, ALL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(
    classifier: object,
    scale_numeric: bool = False,
    drop_id: bool = True,
) -> Pipeline:
    """Build end-to-end pipeline combining feature engineering, preprocessing, and classifier.
    
    Args:
        classifier: Scikit-learn compatible estimator (e.g., LogisticRegression, RandomForestClassifier, XGBClassifier).
        scale_numeric: Whether to scale numeric features (True for LogisticRegression, False for trees).
        drop_id: Whether to drop customerID column.
        
    Returns:
        Unified Scikit-learn Pipeline.
    """
    preprocessor = build_column_preprocessor(scale_numeric=scale_numeric)
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", TelcoFeatureEngineer(drop_id=drop_id)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline

