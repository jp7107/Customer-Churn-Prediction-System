"""Pydantic schemas for Telco Churn API requests and responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerFeatures(BaseModel):
    """Schema representing customer behavioral and subscription attributes."""
    
    customerID: Optional[str] = Field(
        default=None,
        description="Optional tracking identifier (excluded from model features)",
        examples=["7590-VHVEG"],
    )
    gender: str = Field(..., description="Gender: 'Male' or 'Female'", examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen indicator: 1 or 0", examples=[0])
    Partner: str = Field(..., description="Whether customer has partner: 'Yes' or 'No'", examples=["Yes"])
    Dependents: str = Field(..., description="Whether customer has dependents: 'Yes' or 'No'", examples=["No"])
    tenure: int = Field(..., ge=0, le=120, description="Tenure in months", examples=[1])
    PhoneService: str = Field(..., description="Phone service: 'Yes' or 'No'", examples=["No"])
    MultipleLines: str = Field(..., description="Multiple lines: 'Yes', 'No', or 'No phone service'", examples=["No phone service"])
    InternetService: str = Field(..., description="Internet service type: 'DSL', 'Fiber optic', or 'No'", examples=["DSL"])
    OnlineSecurity: str = Field(..., description="Online security: 'Yes', 'No', or 'No internet service'", examples=["No"])
    OnlineBackup: str = Field(..., description="Online backup: 'Yes', 'No', or 'No internet service'", examples=["Yes"])
    DeviceProtection: str = Field(..., description="Device protection: 'Yes', 'No', or 'No internet service'", examples=["No"])
    TechSupport: str = Field(..., description="Tech support: 'Yes', 'No', or 'No internet service'", examples=["No"])
    StreamingTV: str = Field(..., description="Streaming TV: 'Yes', 'No', or 'No internet service'", examples=["No"])
    StreamingMovies: str = Field(..., description="Streaming movies: 'Yes', 'No', or 'No internet service'", examples=["No"])
    Contract: str = Field(..., description="Contract term: 'Month-to-month', 'One year', or 'Two year'", examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., description="Paperless billing: 'Yes' or 'No'", examples=["Yes"])
    PaymentMethod: str = Field(
        ...,
        description="Payment method (e.g. 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring charge", examples=[29.85])
    TotalCharges: Union[float, str] = Field(..., description="Total accumulated charges or numeric string", examples=[29.85])

    @field_validator("gender", "Partner", "Dependents", "PhoneService", "Contract", "PaperlessBilling")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("Field cannot be empty or whitespace.")
        return s

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
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
        },
    )


class PredictionResponse(BaseModel):
    """Schema for churn prediction response."""
    prediction: str = Field(..., description="Human-readable prediction ('Likely to Churn' or 'Not Likely to Churn')", examples=["Likely to Churn"])
    probability: float = Field(..., ge=0.0, le=1.0, description="Model estimated probability of churn (Churn = Yes)", examples=[0.72])
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Explicit churn probability alias", examples=[0.72])
    decision_threshold: float = Field(..., description="Threshold applied to convert probability to binary prediction", examples=[0.50])
    is_churn: bool = Field(..., description="Boolean churn flag", examples=[True])
    customerID: Optional[str] = Field(default=None, description="Echoed trace ID if provided", examples=["7590-VHVEG"])


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500, description="List of customer records (1 to 500)")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    total_records: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Schema for health status response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: str
    model_config = ConfigDict(protected_namespaces=())


class ModelInfoResponse(BaseModel):
    """Schema for model metadata response."""
    model_name: str
    algorithm: str
    selected_threshold: float
    default_threshold: float
    target_definition: str
    evaluation_metrics: Dict[str, Any]
    model_config = ConfigDict(protected_namespaces=())
