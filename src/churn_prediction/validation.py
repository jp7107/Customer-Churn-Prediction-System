"""Dataset schema validation, data quality checks, and anomaly detection."""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.churn_prediction.config import (
    ID_COLUMN,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAPPING,
)
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)


class DataValidationError(Exception):
    """Raised when dataset validation checks fail."""
    pass


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> Dict[str, Any]:
    """Validate that the dataset contains expected columns and valid values."""
    issues: List[str] = []
    
    # Check required feature columns
    missing_features = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        issues.append(f"Missing required feature columns: {missing_features}")

    # Check target if required
    if require_target:
        if TARGET_COLUMN not in df.columns:
            issues.append(f"Target column '{TARGET_COLUMN}' missing from dataframe.")
        else:
            unique_targets = set(df[TARGET_COLUMN].dropna().unique())
            valid_targets = set(TARGET_MAPPING.keys())
            invalid_targets = unique_targets - valid_targets
            if invalid_targets:
                issues.append(f"Unexpected target labels found: {invalid_targets}")

    if issues:
        logger.error("Schema validation failed with %d issues: %s", len(issues), issues)
        raise DataValidationError("; ".join(issues))

    logger.info("Schema validation passed.")
    return {"status": "PASSED", "columns_verified": len(RAW_FEATURE_COLUMNS)}


def run_data_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform a comprehensive data quality audit."""
    report: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "exact_duplicates": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else 0,
        "null_counts": df.isna().sum().to_dict(),
    }

    # Inspect TotalCharges blank strings
    if "TotalCharges" in df.columns:
        total_charges_raw = df["TotalCharges"].astype(str).str.strip()
        blank_mask = total_charges_raw == ""
        blank_count = int(blank_mask.sum())
        report["total_charges_blank_count"] = blank_count
        
        if blank_count > 0 and "tenure" in df.columns:
            blank_tenures = df.loc[blank_mask, "tenure"].value_counts().to_dict()
            report["blank_total_charges_tenures"] = blank_tenures
            report["all_blanks_zero_tenure"] = bool(all(t == 0 for t in blank_tenures.keys()))
        else:
            report["blank_total_charges_tenures"] = {}
            report["all_blanks_zero_tenure"] = False

    # Check value ranges for key numeric columns
    numeric_checks = {}
    if "tenure" in df.columns:
        numeric_checks["tenure_min"] = float(df["tenure"].min())
        numeric_checks["tenure_max"] = float(df["tenure"].max())
    if "MonthlyCharges" in df.columns:
        numeric_checks["monthly_charges_min"] = float(df["MonthlyCharges"].min())
        numeric_checks["monthly_charges_max"] = float(df["MonthlyCharges"].max())
    report["numeric_ranges"] = numeric_checks

    logger.info("Data quality audit completed. Exact duplicates: %d, Blanks in TotalCharges: %d",
                report["exact_duplicates"], report.get("total_charges_blank_count", 0))
    return report

