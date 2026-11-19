"""Shared deterministic feature engineering and cleaning transformers.

This module provides Scikit-learn compatible transformers ensuring that
feature transformations are identical between training and API inference.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.churn_prediction.config import (
    ID_COLUMN,
    SERVICE_COLUMNS,
    TARGET_COLUMN,
)
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Safely convert TotalCharges to numeric, setting zero only if tenure is 0.
    
    If TotalCharges is blank or whitespace and tenure == 0, TotalCharges is set to 0.0.
    Any other non-convertible values become NaN so downstream imputation handles them.
    """
    df_out = df.copy()
    if "TotalCharges" in df_out.columns:
        # Check if already numeric
        if not pd.api.types.is_numeric_dtype(df_out["TotalCharges"]):
            charges_str = df_out["TotalCharges"].astype(str).str.strip()
            # If tenure is 0 and TotalCharges is blank, set to 0.0
            if "tenure" in df_out.columns:
                zero_tenure_mask = (charges_str == "") & (df_out["tenure"] == 0)
                charges_str = charges_str.mask(zero_tenure_mask, "0.0")
            df_out["TotalCharges"] = pd.to_numeric(charges_str, errors="coerce")
    return df_out


def compute_service_count(df: pd.DataFrame) -> pd.Series:
    """Count subscribed services strictly where value is 'Yes' or active InternetService.
    
    'No internet service' and 'No phone service' are treated as non-subscribed (0).
    """
    count = pd.Series(0, index=df.index, dtype=int)
    
    # Check PhoneService
    if "PhoneService" in df.columns:
        count += (df["PhoneService"] == "Yes").astype(int)
    # Check MultipleLines
    if "MultipleLines" in df.columns:
        count += (df["MultipleLines"] == "Yes").astype(int)
    # Check InternetService
    if "InternetService" in df.columns:
        count += df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    # Check add-on services
    addon_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for service in addon_services:
        if service in df.columns:
            count += (df[service] == "Yes").astype(int)
            
    return count


def compute_tenure_group(tenure_series: pd.Series) -> pd.Series:
    """Discretize tenure in months into business-interpretable tenure cohorts.
    
    Cohorts:
    - '0-12 Month'
    - '13-24 Month'
    - '25-48 Month'
    - '49-72 Month'
    """
    bins = [-1, 12, 24, 48, 100]
    labels = ["0-12 Month", "13-24 Month", "25-48 Month", "49-72 Month"]
    return pd.cut(tenure_series, bins=bins, labels=labels).astype(str)


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn Transformer for Telco Churn feature engineering.
    
    Performs:
    1. Safe TotalCharges numeric conversion with tenure==0 logic.
    2. Service count calculation (strictly active services).
    3. Tenure group discretization.
    4. Drops customerID and target column if present.
    """

    def __init__(self, drop_id: bool = True):
        self.drop_id = drop_id

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer (stateless transformation)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        df = clean_total_charges(X)
        
        # Add engineered features
        df["service_count"] = compute_service_count(df)
        if "tenure" in df.columns:
            df["tenure_group"] = compute_tenure_group(df["tenure"])
        else:
            df["tenure_group"] = "Unknown"
            
        # Ensure SeniorCitizen is treated as categorical string
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
            
        # Drop ID and target columns if present to avoid leakage
        cols_to_drop = []
        if self.drop_id and ID_COLUMN in df.columns:
            cols_to_drop.append(ID_COLUMN)
        if TARGET_COLUMN in df.columns:
            cols_to_drop.append(TARGET_COLUMN)
            
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
        return df

