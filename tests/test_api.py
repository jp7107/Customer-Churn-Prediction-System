"""Tests for FastAPI endpoints, request validation, and error handling."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client with lifespan context enabled."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Provide a valid customer payload."""
    return {
        "customerID": "TEST-1234",
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


def test_health_endpoint(client):
    """Test /health route returns healthy status and loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] is not None


def test_model_info_endpoint(client):
    """Test /model-info route returns metadata and threshold."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "selected_threshold" in data
    assert "evaluation_metrics" in data
    assert "target_definition" in data


def test_predict_endpoint_success(client, valid_payload):
    """Test /predict route returns prediction and probability."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Likely to Churn", "Not Likely to Churn"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["churn_probability"] == data["probability"]
    assert data["customerID"] == valid_payload["customerID"]


def test_predict_batch_endpoint_success(client, valid_payload):
    """Test /predict/batch returns multiple predictions."""
    batch_payload = {"customers": [valid_payload, dict(valid_payload, tenure=45)]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2


def test_predict_endpoint_validation_error(client, valid_payload):
    """Test that invalid numeric input triggers 422 Unprocessable Entity."""
    bad_payload = dict(valid_payload, tenure=-5)  # Negative tenure violates ge=0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "errors" in data


def test_predict_endpoint_unseen_category(client, valid_payload):
    """Test that unseen categorical values are handled gracefully without 500 error."""
    unseen_payload = dict(valid_payload, PaymentMethod="Unseen New Cryptopay")
    response = client.post("/predict", json=unseen_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Likely to Churn", "Not Likely to Churn"]

