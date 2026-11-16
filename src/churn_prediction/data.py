"""Data loading, provenance tracking, and dataset fingerprinting."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.churn_prediction.config import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
)
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Compute and return the SHA-256 checksum of a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found for hashing: {filepath}")
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Telco customer churn dataset."""
    path = Path(filepath) if filepath else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/."
        )
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def get_data_provenance(df: pd.DataFrame, filepath: Path) -> Dict[str, Any]:
    """Extract metadata and provenance information from dataset."""
    sha256 = compute_file_sha256(filepath)
    target_counts = df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {}
    target_pct = (
        (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()
        if TARGET_COLUMN in df.columns
        else {}
    )

    return {
        "source_url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "filepath": str(filepath.resolve()),
        "filename": filepath.name,
        "sha256": sha256,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "target_counts": target_counts,
        "target_distribution_percent": target_pct,
    }
