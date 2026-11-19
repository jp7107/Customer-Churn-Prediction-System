"""Model explainability using SHAP values for churn prediction.

Cognirex Solutions AI/ML Technical Assessment
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.churn_prediction.config import DEFAULT_MODEL_PATH, FIGURES_DIR
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)


class ChurnExplainer:
    """SHAP-based explainability for XGBoost churn prediction model."""

    def __init__(self, pipeline_path: str = DEFAULT_MODEL_PATH):
        """Load trained pipeline and initialize SHAP explainer.
        
        Args:
            pipeline_path: Path to trained .joblib pipeline
        """
        logger.info(f"Loading pipeline for explainability from {pipeline_path}")
        self.pipeline: Pipeline = joblib.load(pipeline_path)
        
        # Extract the final XGBoost model from pipeline
        self.model = self.pipeline.named_steps["classifier"]
        
        # Get feature names after preprocessing
        preprocessor = self.pipeline.named_steps["preprocessor"]
        self.feature_names = self._get_feature_names(preprocessor)
        
        logger.info(f"Initialized explainer with {len(self.feature_names)} features")
        self.explainer: Optional[shap.TreeExplainer] = None

    def _get_feature_names(self, preprocessor) -> List[str]:
        """Extract feature names from ColumnTransformer."""
        feature_names = []
        
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or name == "drop":
                continue
            
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
        
        return feature_names

    def fit_explainer(self, X_train: pd.DataFrame, sample_size: int = 100):
        """Fit SHAP TreeExplainer on training data sample.
        
        Args:
            X_train: Training dataframe
            sample_size: Number of samples for background distribution
        """
        logger.info("Transforming training data through preprocessing pipeline...")
        
        # Transform through preprocessing steps only (not final classifier)
        X_transformed = self.pipeline[:-1].transform(X_train)
        
        # Sample for computational efficiency
        if len(X_transformed) > sample_size:
            indices = np.random.choice(len(X_transformed), sample_size, replace=False)
            background = X_transformed[indices]
        else:
            background = X_transformed
        
        logger.info(f"Fitting SHAP TreeExplainer with background size {len(background)}...")
        self.explainer = shap.TreeExplainer(self.model, background)
        logger.info("SHAP explainer fitted successfully")

    def explain_prediction(
        self, 
        customer_data: Dict, 
        show_plot: bool = False
    ) -> Dict:
        """Generate SHAP explanation for a single customer prediction.
        
        Args:
            customer_data: Dictionary with customer features
            show_plot: Whether to display waterfall plot
            
        Returns:
            Dictionary with prediction, probability, and top feature contributions
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")
        
        # Convert to DataFrame and preprocess
        df = pd.DataFrame([customer_data])
        X_transformed = self.pipeline[:-1].transform(df)
        
        # Get prediction and probability
        pred_proba = self.model.predict_proba(X_transformed)[0]
        pred_class = int(pred_proba[1] > 0.5)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_transformed)
        
        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1][0]
        else:
            shap_values_positive = shap_values[0]
        
        # Get top contributing features
        feature_importance = list(zip(self.feature_names, shap_values_positive))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = [
            {"feature": feat, "shap_value": float(val), "impact": "increase" if val > 0 else "decrease"}
            for feat, val in feature_importance[:5]
        ]
        
        result = {
            "prediction": "Churn" if pred_class == 1 else "No Churn",
            "churn_probability": float(pred_proba[1]),
            "base_value": float(self.explainer.expected_value),
            "top_contributors": top_features,
            "explanation": self._generate_text_explanation(top_features, pred_proba[1])
        }
        
        if show_plot:
            self._plot_waterfall(shap_values_positive, pred_proba[1])
        
        return result

    def _generate_text_explanation(self, top_features: List[Dict], prob: float) -> str:
        """Generate human-readable explanation."""
        risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        
        explanation = f"This customer has a {risk_level} churn risk ({prob:.1%} probability). "
        explanation += "Key factors: "
        
        factors = []
        for feat_dict in top_features[:3]:
            feat = feat_dict["feature"]
            impact = "increases" if feat_dict["impact"] == "increase" else "decreases"
            factors.append(f"{feat} {impact} risk")
        
        explanation += ", ".join(factors) + "."
        return explanation

    def _plot_waterfall(self, shap_values: np.ndarray, probability: float):
        """Display SHAP waterfall plot."""
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=None,
                feature_names=self.feature_names
            )
        )
        plt.title(f"SHAP Explanation (Churn Probability: {probability:.2%})")
        plt.tight_layout()
        plt.show()

    def generate_global_feature_importance(
        self, 
        X_test: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Generate global feature importance using SHAP summary.
        
        Args:
            X_test: Test dataframe
            save_path: Optional path to save summary plot
            
        Returns:
            DataFrame with mean absolute SHAP values per feature
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")
        
        logger.info("Computing SHAP values for test set...")
        X_transformed = self.pipeline[:-1].transform(X_test)
        shap_values = self.explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_transformed,
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP summary plot to {save_path}")
        
        plt.close()
        
        return importance_df


def main():
    """Demo: Generate SHAP explanations for sample predictions."""
    from src.churn_prediction.data import load_raw_data
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = load_raw_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize explainer
    explainer = ChurnExplainer()
    explainer.fit_explainer(X_train, sample_size=100)
    
    # Explain a high-risk customer
    high_risk_customer = {
        "customerID": "EXPLAIN-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 89.85
    }
    
    explanation = explainer.explain_prediction(high_risk_customer, show_plot=True)
    print("\n=== Customer Churn Explanation ===")
    print(json.dumps(explanation, indent=2))
    
    # Generate global importance
    importance_df = explainer.generate_global_feature_importance(
        X_test, 
        save_path=FIGURES_DIR / "shap_summary.png"
    )
    print("\n=== Top 10 Most Important Features (SHAP) ===")
    print(importance_df.head(10))


if __name__ == "__main__":
    main()
