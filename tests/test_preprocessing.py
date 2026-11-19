"""Tests for feature engineering, TotalCharges cleaning, and ColumnTransformer."""

import numpy as np
import pandas as pd
import pytest
from src.churn_prediction.config import ID_COLUMN, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.churn_prediction.features import (
    clean_total_charges,
    compute_service_count,
    compute_tenure_group,
    TelcoFeatureEngineer,
)
from src.churn_prediction.preprocessing import build_column_preprocessor


@pytest.fixture
def sample_feature_df():
    """Create sample records for transformation testing."""
    return pd.DataFrame({
        "customerID": ["CUST_01", "CUST_02"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [0, 36],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["No", "Fiber optic"],
        "OnlineSecurity": ["No internet service", "Yes"],
        "OnlineBackup": ["No internet service", "No"],
        "DeviceProtection": ["No internet service", "Yes"],
        "TechSupport": ["No internet service", "No"],
        "StreamingTV": ["No internet service", "Yes"],
        "StreamingMovies": ["No internet service", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        "MonthlyCharges": [20.0, 95.0],
        "TotalCharges": [" ", "3420.0"],
        "Churn": ["No", "No"],
    })


def test_clean_total_charges_zero_tenure(sample_feature_df):
    """Test that blank TotalCharges for tenure=0 becomes 0.0."""
    cleaned = clean_total_charges(sample_feature_df)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 3420.0
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_compute_service_count(sample_feature_df):
    """Test that service count only counts active services."""
    services = compute_service_count(sample_feature_df)
    # Record 0 has only PhoneService = 'Yes' (1 service)
    assert services.iloc[0] == 1
    # Record 1 has Phone(1) + MultipleLines(1) + Fiber(1) + Sec(1) + Dev(1) + TV(1) + Movies(1) = 7 services
    assert services.iloc[1] == 7


def test_compute_tenure_group():
    """Test tenure cohort binning."""
    tenures = pd.Series([0, 6, 12, 13, 24, 25, 48, 50, 72])
    groups = compute_tenure_group(tenures)
    assert groups.iloc[0] == "0-12 Month"
    assert groups.iloc[2] == "0-12 Month"
    assert groups.iloc[3] == "13-24 Month"
    assert groups.iloc[6] == "25-48 Month"
    assert groups.iloc[8] == "49-72 Month"


def test_feature_engineer_drops_id_and_target(sample_feature_df):
    """Test that customerID and Churn are dropped from model predictors."""
    transformer = TelcoFeatureEngineer(drop_id=True)
    transformed = transformer.transform(sample_feature_df)
    assert ID_COLUMN not in transformed.columns
    assert TARGET_COLUMN not in transformed.columns
    assert "service_count" in transformed.columns
    assert "tenure_group" in transformed.columns


def test_column_preprocessor_handles_unseen_category(sample_feature_df):
    """Test that one-hot encoder ignores unseen categories during transform."""
    engineer = TelcoFeatureEngineer(drop_id=True)
    df_eng = engineer.transform(sample_feature_df)
    
    preprocessor = build_column_preprocessor(scale_numeric=True)
    preprocessor.fit(df_eng)
    
    # Introduce an unseen category
    df_unseen = df_eng.copy()
    df_unseen.loc[0, "PaymentMethod"] = "Crypto Coin"
    
    # Transform should not raise an error due to handle_unknown='ignore'
    arr = preprocessor.transform(df_unseen)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 2

