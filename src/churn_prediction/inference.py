"""Inference service for loading trained pipeline and producing predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.churn_prediction.config import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    INVERSE_TARGET_MAPPING,
)
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)


class ChurnPredictor:
    """Production inference wrapper around the serialized Scikit-learn churn pipeline."""

    _instance: Optional["ChurnPredictor"] = None

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.pipeline: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.decision_threshold: float = 0.50
        self.model_name: str = "Unknown"
        self._load()

    def _load(self) -> None:
        """Load pipeline artifact and associated metadata."""
        if not self.model_path.exists():
            logger.error("Model artifact not found at %s", self.model_path)
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Train model first via python -m src.churn_prediction.training"
            )

        logger.info("Loading churn model pipeline from %s", self.model_path)
        self.pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.decision_threshold = float(self.metadata.get("selected_threshold", 0.50))
            self.model_name = self.metadata.get("model_name", "Trained Churn Model")
        else:
            logger.warning("Metadata file not found at %s; using default threshold 0.50", self.metadata_path)
            self.decision_threshold = 0.50
            self.model_name = "Trained Churn Model"

        logger.info("Model '%s' successfully loaded. Decision threshold: %.2f", self.model_name, self.decision_threshold)

    @classmethod
    def get_instance(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> "ChurnPredictor":
        """Singleton accessor for cached predictor instance."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, metadata_path=metadata_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful in tests)."""
        cls._instance = None

    def predict_records(
        self,
        records: List[Dict[str, Any]],
        custom_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Predict churn probabilities and classifications for a batch of raw records."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")

        threshold = custom_threshold if custom_threshold is not None else self.decision_threshold
        df_input = pd.DataFrame(records)

        # Pipeline handles feature engineering, imputation, encoding, scaling, and classification
        probabilities = self.pipeline.predict_proba(df_input)[:, 1]

        results = []
        for idx, prob in enumerate(probabilities):
            prob_float = float(round(prob, 4))
            binary_pred = int(prob >= threshold)
            label = INVERSE_TARGET_MAPPING.get(binary_pred, "Unknown")

            results.append({
                "prediction": label,
                "probability": prob_float,
                "churn_probability": prob_float,
                "decision_threshold": threshold,
                "is_churn": bool(binary_pred == 1),
            })
        return results

    def predict_single(
        self,
        record: Dict[str, Any],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict churn probability and classification for a single customer record."""
        return self.predict_records([record], custom_threshold=custom_threshold)[0]
