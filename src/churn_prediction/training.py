"""End-to-end model training, cross-validation, comparison, and serialization pipeline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

from src.churn_prediction.config import (
    CV_FOLDS,
    DEFAULT_DATA_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    FIGURES_DIR,
    ID_COLUMN,
    INVERSE_TARGET_MAPPING,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_FEATURE_COLUMNS,
    REPORTS_DIR,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.churn_prediction.data import compute_file_sha256, get_data_provenance, load_raw_data
from src.churn_prediction.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_curves,
)
from src.churn_prediction.logging_config import setup_logger
from src.churn_prediction.preprocessing import build_full_pipeline
from src.churn_prediction.validation import run_data_quality_audit, validate_schema

logger = setup_logger(__name__)


def prepare_data_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a leakage-safe stratified train/test split.
    
    Splitting occurs before any transformation or imputation.
    """
    logger.info("Splitting dataset: test_size=%.2f, random_state=%d", TEST_SIZE, RANDOM_STATE)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)

    if y.isna().any():
        raise ValueError("Target contains unmapped or missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split complete. Train: %d records (%.1f%% churn), Test: %d records (%.1f%% churn)",
        len(X_train),
        (y_train.mean() * 100),
        len(X_test),
        (y_test.mean() * 100),
    )
    return X_train, X_test, y_train, y_test


def evaluate_cv_pipeline(
    pipeline_factory: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate a pipeline factory using 5-Fold Stratified Cross Validation.
    
    Returns CV summary statistics and out-of-fold predicted probabilities.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y_train))
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        pipeline = pipeline_factory()
        pipeline.fit(X_tr, y_tr)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba

        m = compute_metrics(y_val.to_numpy(), val_proba, threshold=0.50)
        fold_metrics.append({
            "fold": fold,
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    df_folds = pd.DataFrame(fold_metrics)
    summary = {
        "model_name": model_name,
        "cv_roc_auc_mean": float(df_folds["roc_auc"].mean()),
        "cv_roc_auc_std": float(df_folds["roc_auc"].std()),
        "cv_pr_auc_mean": float(df_folds["pr_auc"].mean()),
        "cv_pr_auc_std": float(df_folds["pr_auc"].std()),
        "cv_f1_mean": float(df_folds["f1"].mean()),
        "cv_f1_std": float(df_folds["f1"].std()),
        "cv_accuracy_mean": float(df_folds["accuracy"].mean()),
        "cv_precision_mean": float(df_folds["precision"].mean()),
        "cv_recall_mean": float(df_folds["recall"].mean()),
    }
    logger.info(
        "CV for %s: ROC-AUC = %.4f +/- %.4f | PR-AUC = %.4f | F1 = %.4f | Recall = %.4f",
        model_name,
        summary["cv_roc_auc_mean"],
        summary["cv_roc_auc_std"],
        summary["cv_pr_auc_mean"],
        summary["cv_f1_mean"],
        summary["cv_recall_mean"],
    )
    return summary, oof_preds


def plot_cv_comparison(df_comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot cross-validation comparison across candidates."""
    metrics_to_plot = ["cv_roc_auc_mean", "cv_pr_auc_mean", "cv_f1_mean", "cv_recall_mean"]
    labels = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall"]
    
    df_plot = df_comparison.copy()
    df_plot = df_plot.rename(columns=dict(zip(metrics_to_plot, labels)))
    df_melt = pd.melt(
        df_plot,
        id_vars=["model_name"],
        value_vars=labels,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="model_name", palette="Set2")
    plt.title("Cross-Validation Model Comparison (5-Fold Stratified CV)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Cross-Validation Mean Score", fontsize=11)
    plt.ylim(0.0, 1.0)
    plt.legend(title="Candidate Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved CV comparison plot to %s", output_path)


def run_training_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    model_save_path: Path = DEFAULT_MODEL_PATH,
    metadata_save_path: Path = DEFAULT_METADATA_PATH,
) -> Dict[str, Any]:
    """Execute complete end-to-end training, model selection, evaluation, and serialization."""
    logger.info("=== Starting Churn Prediction Model Pipeline ===")
    
    # 1. Load data & Provenance
    raw_df = load_raw_data(data_path)
    provenance = get_data_provenance(raw_df, data_path)
    validate_schema(raw_df, require_target=True)
    quality_audit = run_data_quality_audit(raw_df)

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = prepare_data_split(raw_df)

    # 3. Calculate class imbalance ratio on TRAINING set only
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    logger.info("Train class distribution: Negative=%d, Positive=%d (Ratio: %.2f)", neg_count, pos_count, scale_pos_weight)

    # 4. Define candidate models
    candidate_defs = [
        # Candidate 1: Logistic Regression (StandardScaler applied)
        {
            "name": "Logistic Regression (Balanced)",
            "factory": lambda: build_full_pipeline(
                classifier=LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
                scale_numeric=True,
            ),
        },
        {
            "name": "Logistic Regression (Standard)",
            "factory": lambda: build_full_pipeline(
                classifier=LogisticRegression(
                    C=0.1,
                    class_weight=None,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
                scale_numeric=True,
            ),
        },
        # Candidate 2: Random Forest Classifier
        {
            "name": "Random Forest (Balanced)",
            "factory": lambda: build_full_pipeline(
                classifier=RandomForestClassifier(
                    n_estimators=150,
                    max_depth=7,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
        },
        {
            "name": "Random Forest (Standard)",
            "factory": lambda: build_full_pipeline(
                classifier=RandomForestClassifier(
                    n_estimators=150,
                    max_depth=7,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
        },
        # Candidate 3: XGBoost Classifier
        {
            "name": "XGBoost (Weighted)",
            "factory": lambda: build_full_pipeline(
                classifier=xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
        },
        {
            "name": "XGBoost (Standard)",
            "factory": lambda: build_full_pipeline(
                classifier=xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=1.0,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
        },
    ]

    # 5. Run Stratified Cross-Validation on all candidates
    cv_results: List[Dict[str, Any]] = []
    candidate_oof_preds: Dict[str, np.ndarray] = {}

    for cand in candidate_defs:
        summary, oof_preds = evaluate_cv_pipeline(
            pipeline_factory=cand["factory"],
            X_train=X_train,
            y_train=y_train,
            model_name=cand["name"],
        )
        cv_results.append(summary)
        candidate_oof_preds[cand["name"]] = oof_preds

    df_comparison = pd.DataFrame(cv_results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    plot_cv_comparison(df_comparison, FIGURES_DIR / "cv_model_comparison.png")

    # 6. Select Best Performing Model based on Training CV ROC-AUC and PR-AUC
    # We sort by cv_roc_auc_mean descending
    df_sorted = df_comparison.sort_values(by=["cv_roc_auc_mean", "cv_pr_auc_mean"], ascending=False)
    best_candidate_name = df_sorted.iloc[0]["model_name"]
    best_candidate_def = next(c for c in candidate_defs if c["name"] == best_candidate_name)
    logger.info("Selected winning model based on CV performance: %s", best_candidate_name)

    # 7. Optimize Decision Threshold using Training Out-Of-Fold predictions ONLY
    best_oof_preds = candidate_oof_preds[best_candidate_name]
    optimal_threshold, threshold_curve_metrics = find_optimal_threshold(
        y_train.to_numpy(), best_oof_preds, metric="f1"
    )
    plot_threshold_curves(threshold_curve_metrics, optimal_threshold, FIGURES_DIR / "threshold_analysis.png")

    # 8. Refit Winning Pipeline on Entire Training Set (80%)
    logger.info("Refitting winning model '%s' on full training set (%d samples)...", best_candidate_name, len(X_train))
    final_pipeline = best_candidate_def["factory"]()
    final_pipeline.fit(X_train, y_train)

    # 9. One-time Final Evaluation on Untouched Holdout Test Set (20%)
    logger.info("Evaluating final pipeline once on untouched holdout test set (%d samples)...", len(X_test))
    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    # Compute metrics at default threshold 0.50
    test_metrics_default = compute_metrics(y_test_arr, test_proba, threshold=0.50)
    # Compute metrics at optimal threshold
    test_metrics_optimal = compute_metrics(y_test_arr, test_proba, threshold=optimal_threshold)

    logger.info("=== Holdout Test Results (Default Threshold 0.50) ===")
    logger.info("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f | PR-AUC: %.4f",
                test_metrics_default["accuracy"], test_metrics_default["precision"],
                test_metrics_default["recall"], test_metrics_default["f1"],
                test_metrics_default["roc_auc"], test_metrics_default["pr_auc"])

    logger.info("=== Holdout Test Results (Optimized Threshold %.2f) ===", optimal_threshold)
    logger.info("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f | PR-AUC: %.4f",
                test_metrics_optimal["accuracy"], test_metrics_optimal["precision"],
                test_metrics_optimal["recall"], test_metrics_optimal["f1"],
                test_metrics_optimal["roc_auc"], test_metrics_optimal["pr_auc"])

    # 10. Generate Test Visualizations
    cm = np.array([
        [test_metrics_optimal["confusion_matrix"]["tn"], test_metrics_optimal["confusion_matrix"]["fp"]],
        [test_metrics_optimal["confusion_matrix"]["fn"], test_metrics_optimal["confusion_matrix"]["tp"]],
    ])
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix_test.png", title=f"Confusion Matrix — {best_candidate_name} (Thresh={optimal_threshold:.2f})")
    plot_roc_curve(y_test_arr, test_proba, FIGURES_DIR / "roc_curve_test.png", model_name=best_candidate_name)
    plot_precision_recall_curve(y_test_arr, test_proba, FIGURES_DIR / "precision_recall_curve_test.png", model_name=best_candidate_name)
    plot_feature_importance(final_pipeline, FIGURES_DIR / "feature_importance.png")

    # 11. Save Final Model Artifact and Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving complete model pipeline artifact to %s", model_save_path)
    joblib.dump(final_pipeline, model_save_path)

    metadata: Dict[str, Any] = {
        "model_name": best_candidate_name,
        "algorithm": final_pipeline.named_steps["classifier"].__class__.__name__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "selected_threshold": optimal_threshold,
        "default_threshold": 0.50,
        "dataset_provenance": provenance,
        "data_quality_audit": quality_audit,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "target_mapping": TARGET_MAPPING,
        "inverse_target_mapping": INVERSE_TARGET_MAPPING,
        "raw_features": RAW_FEATURE_COLUMNS,
        "cv_performance": next(c for c in cv_results if c["model_name"] == best_candidate_name),
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics_optimal_threshold": test_metrics_optimal,
    }

    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", metadata_save_path)

    # Save final evaluation summary
    with open(REPORTS_DIR / "final_evaluation.json", "w") as f:
        json.dump({
            "model_name": best_candidate_name,
            "selected_threshold": optimal_threshold,
            "test_metrics_optimal_threshold": test_metrics_optimal,
            "test_metrics_default_threshold": test_metrics_default,
        }, f, indent=2)

    logger.info("=== Pipeline Training & Evaluation Successfully Completed ===")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Telco Churn Prediction models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw dataset CSV.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for joblib model artifact.")
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_PATH, help="Output path for model metadata JSON.")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        model_save_path=args.model_out,
        metadata_save_path=args.metadata_out,
    )


if __name__ == "__main__":
    main()

