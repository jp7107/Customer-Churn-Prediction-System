"""Tests for model loading, single and batch inference, and artifact reload consistency."""

import json
import pytest
from src.churn_prediction.config import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH
from src.churn_prediction.inference import ChurnPredictor


@pytest.fixture
def predictor():
    """Retrieve predictor singleton."""
    return ChurnPredictor.get_instance(
        model_path=DEFAULT_MODEL_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
    )


@pytest.fixture
def sample_customer():
    """Provide a realistic customer dictionary payload."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_predictor_initialization(predictor):
    """Verify that predictor loads model and metadata."""
    assert predictor.pipeline is not None
    assert isinstance(predictor.decision_threshold, float)
    assert 0.0 < predictor.decision_threshold < 1.0


def test_predict_single(predictor, sample_customer):
    """Test single record inference returns required keys and valid types."""
    res = predictor.predict_single(sample_customer)
    assert "prediction" in res
    assert res["prediction"] in ["Likely to Churn", "Not Likely to Churn"]
    assert "probability" in res
    assert 0.0 <= res["probability"] <= 1.0
    assert "churn_probability" in res
    assert "decision_threshold" in res
    assert isinstance(res["is_churn"], bool)


def test_predict_batch(predictor, sample_customer):
    """Test batch prediction handles multiple records accurately."""
    batch = [sample_customer, dict(sample_customer, tenure=70, Contract="Two year")]
    results = predictor.predict_records(batch)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["probability"] <= 1.0


def test_custom_threshold_override(predictor, sample_customer):
    """Test that specifying custom threshold alters classification appropriately."""
    res_high = predictor.predict_single(sample_customer, custom_threshold=0.99)
    # At 0.99 threshold, customer should almost certainly be classified as Not Likely to Churn
    assert res_high["prediction"] == "Not Likely to Churn"

    res_low = predictor.predict_single(sample_customer, custom_threshold=0.01)
    # At 0.01 threshold, any non-zero probability classifies as Likely to Churn
    assert res_low["prediction"] == "Likely to Churn"


def test_model_reload_reproducibility(sample_customer):
    """Test that reloading the saved artifact produces identical predictions."""
    pred1 = ChurnPredictor(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)
    out1 = pred1.predict_single(sample_customer)
    
    pred2 = ChurnPredictor(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)
    out2 = pred2.predict_single(sample_customer)
    
    assert out1["probability"] == out2["probability"]
    assert out1["prediction"] == out2["prediction"]
