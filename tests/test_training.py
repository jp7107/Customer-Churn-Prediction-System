"""Tests for model training components, cross-validation, and metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.churn_prediction.config import TARGET_COLUMN
from src.churn_prediction.evaluation import compute_metrics, find_optimal_threshold
from src.churn_prediction.preprocessing import build_full_pipeline
from src.churn_prediction.training import prepare_data_split


@pytest.fixture
def synthetic_training_df():
    """Create synthetic dataframe for pipeline fitting tests."""
    np.random.seed(42)
    n = 60
    data = {
        "customerID": [f"ID_{i}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(0, 72, n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
        "MultipleLines": np.random.choice(["Yes", "No"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No"], n),
        "OnlineBackup": np.random.choice(["Yes", "No"], n),
        "DeviceProtection": np.random.choice(["Yes", "No"], n),
        "TechSupport": np.random.choice(["Yes", "No"], n),
        "StreamingTV": np.random.choice(["Yes", "No"], n),
        "StreamingMovies": np.random.choice(["Yes", "No"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges": np.random.uniform(20.0, 100.0, n),
        "TotalCharges": np.random.uniform(100.0, 5000.0, n).astype(str),
        TARGET_COLUMN: np.random.choice(["No", "Yes"], n, p=[0.7, 0.3]),
    }
    return pd.DataFrame(data)


def test_prepare_data_split_stratification(synthetic_training_df):
    """Test that stratified split maintains label proportion."""
    X_train, X_test, y_train, y_test = prepare_data_split(synthetic_training_df)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    # Check that proportions are close
    assert abs(train_ratio - test_ratio) < 0.15
    assert len(X_train) + len(X_test) == len(synthetic_training_df)


def test_pipeline_fit_predict_proba(synthetic_training_df):
    """Test fitting a full pipeline on fixture and predicting probabilities."""
    X_train, X_test, y_train, y_test = prepare_data_split(synthetic_training_df)
    pipeline = build_full_pipeline(
        classifier=LogisticRegression(max_iter=200, random_state=42),
        scale_numeric=True,
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    
    assert proba.shape == (len(X_test),)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_compute_metrics():
    """Test metric computation with known values."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    metrics = compute_metrics(y_true, y_proba, threshold=0.50)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_find_optimal_threshold():
    """Test that optimal threshold finding returns sensible threshold."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_proba = np.array([0.2, 0.3, 0.4, 0.55, 0.65, 0.7, 0.45, 0.8])
    thresh, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
    assert 0.1 <= thresh <= 0.9

