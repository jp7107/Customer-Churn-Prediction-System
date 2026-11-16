"""Model evaluation, metrics computation, threshold optimization, and visualization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.churn_prediction.config import FIGURES_DIR
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute all required classification metrics at a given decision threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
            "total": int(len(y_true)),
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """Find the threshold that maximizes the chosen metric (e.g., F1) on validation data."""
    best_threshold = 0.50
    best_score = -1.0
    threshold_metrics = {}

    thresholds = np.linspace(0.10, 0.90, 81)
    for t in thresholds:
        m = compute_metrics(y_true, y_pred_proba, threshold=t)
        score = m[metric]
        threshold_metrics[round(t, 2)] = {
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "accuracy": m["accuracy"],
        }
        if score > best_score:
            best_score = score
            best_threshold = float(round(t, 2))

    logger.info("Optimal threshold by %s: %.2f (Score: %.4f)", metric, best_threshold, best_score)
    return best_threshold, threshold_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save labeled confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    labels = ["Stay (0)", "Churn (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    model_name: str = "Final Model",
) -> None:
    """Plot and save ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved ROC curve to %s", output_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    model_name: str = "Final Model",
) -> None:
    """Plot and save Precision-Recall curve with Average Precision."""
    prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    baseline_churn_rate = y_true.mean()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#2ca02c", lw=2, label=f"{model_name} (PR-AUC / AP = {ap_score:.3f})")
    plt.axhline(y=baseline_churn_rate, color="grey", linestyle="--", label=f"Baseline Churn Rate ({baseline_churn_rate:.1%})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved Precision-Recall curve to %s", output_path)


def plot_threshold_curves(
    threshold_metrics: Dict[float, Dict[str, float]],
    optimal_threshold: float,
    output_path: Path,
) -> None:
    """Plot precision, recall, and F1 curves across decision thresholds."""
    df_thresh = pd.DataFrame.from_dict(threshold_metrics, orient="index")
    
    plt.figure(figsize=(7, 5))
    plt.plot(df_thresh.index, df_thresh["f1"], label="F1-Score", color="#1f77b4", lw=2)
    plt.plot(df_thresh.index, df_thresh["precision"], label="Precision", color="#2ca02c", lw=2)
    plt.plot(df_thresh.index, df_thresh["recall"], label="Recall", color="#d62728", lw=2)
    plt.axvline(x=optimal_threshold, color="black", linestyle="--", label=f"Selected Threshold ({optimal_threshold:.2f})")
    plt.axvline(x=0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Threshold vs. Precision, Recall & F1-Score", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="center right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved threshold curves to %s", output_path)


def plot_feature_importance(
    pipeline: Any,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Extract and plot top features from the trained pipeline."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        
        # Get feature names from column transformer
        cat_encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        cat_features = preprocessor.transformers_[1][2]
        cat_feature_names = list(cat_encoder.get_feature_names_out(cat_features))
        num_feature_names = list(preprocessor.transformers_[0][2])
        all_feature_names = num_feature_names + cat_feature_names
        
        importances = None
        label = "Importance"
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            label = "Gini Feature Importance"
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
            label = "Absolute Logistic Coefficient"
            
        if importances is not None and len(importances) == len(all_feature_names):
            df_imp = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
            df_top = df_imp.sort_values(by="Importance", ascending=False).head(top_n)
            
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df_top, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
            plt.title(f"Top {top_n} Most Predictive Features ({label})", fontsize=13, fontweight="bold", pad=12)
            plt.xlabel(label, fontsize=11)
            plt.ylabel("Feature", fontsize=11)
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info("Saved feature importance plot to %s", output_path)
    except Exception as e:
        logger.warning("Could not generate feature importance plot: %s", e)
