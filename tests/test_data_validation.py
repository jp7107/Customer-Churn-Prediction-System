"""Tests for data loading, provenance, schema validation, and quality checks."""

import pytest
import pandas as pd
from src.churn_prediction.config import TARGET_COLUMN, RAW_FEATURE_COLUMNS
from src.churn_prediction.data import compute_file_sha256, get_data_provenance, load_raw_data
from src.churn_prediction.validation import (
    DataValidationError,
    run_data_quality_audit,
    validate_schema,
)


@pytest.fixture
def sample_valid_df():
    """Create a minimal valid dataframe representing raw Telco records."""
    data = {col: ["No"] * 5 for col in RAW_FEATURE_COLUMNS}
    data["customerID"] = [f"ID_{i}" for i in range(5)]
    data["tenure"] = [1, 12, 24, 0, 48]
    data["MonthlyCharges"] = [29.85, 50.0, 75.5, 20.0, 90.0]
    data["TotalCharges"] = ["29.85", "600.0", "1800.0", " ", "4320.0"]
    data["SeniorCitizen"] = [0, 0, 1, 0, 0]
    data[TARGET_COLUMN] = ["No", "Yes", "No", "No", "Yes"]
    return pd.DataFrame(data)


def test_validate_schema_success(sample_valid_df):
    """Test that schema validation passes for a properly structured dataset."""
    result = validate_schema(sample_valid_df, require_target=True)
    assert result["status"] == "PASSED"
    assert result["columns_verified"] == len(RAW_FEATURE_COLUMNS)


def test_validate_schema_missing_column(sample_valid_df):
    """Test that dropping a required feature raises DataValidationError."""
    df_missing = sample_valid_df.drop(columns=["MonthlyCharges"])
    with pytest.raises(DataValidationError, match="Missing required feature columns"):
        validate_schema(df_missing, require_target=True)


def test_validate_schema_invalid_target(sample_valid_df):
    """Test that an unexpected target label raises DataValidationError."""
    df_bad_target = sample_valid_df.copy()
    df_bad_target.loc[0, TARGET_COLUMN] = "Maybe"
    with pytest.raises(DataValidationError, match="Unexpected target labels"):
        validate_schema(df_bad_target, require_target=True)


def test_run_data_quality_audit(sample_valid_df):
    """Test that data quality audit detects blank TotalCharges and zero tenure."""
    audit = run_data_quality_audit(sample_valid_df)
    assert audit["total_rows"] == 5
    assert audit["exact_duplicates"] == 0
    assert audit["total_charges_blank_count"] == 1
    assert audit["all_blanks_zero_tenure"] is True

