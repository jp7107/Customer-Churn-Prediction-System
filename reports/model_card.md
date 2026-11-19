# Model Card — Telco Customer Churn Predictor

**Model Name:** XGBoost (Weighted) Customer Churn Pipeline  
**Model Version:** `1.0.0`  
**Model Type:** Gradient Boosted Decision Trees (`xgboost.XGBClassifier`) with Scikit-learn Pipeline  
**Author / Developer:** Jyotiprakash  
**Assessment:** Cognirex Solutions AI/ML Technical Assessment (Round 2)  
**Date:** November 2026  

---

## 1. Model Overview & Intended Use

- **Intended Use:** Predict the probability that an individual telecommunications customer will terminate their service contract within the subsequent billing window (`Churn = Yes`).
- **Target Meaning:**
  - Class `1` (`Likely to Churn`): Customer cancels their subscription.
  - Class `0` (`Not Likely to Churn`): Customer maintains active subscription.
- **Operational Output:** Continuous probability $P(\text{Churn} = 1 \mid X)$ bounded in $[0.0, 1.0]$, accompanied by a binary decision flag based on an optimized threshold ($\tau = 0.54$).
- **Out-of-Scope Uses:** This model is not intended for credit scoring, underwriting, pricing discrimination, or legal decisions.

---

## 2. Training Data & Preprocessing Pipeline

- **Source Dataset:** IBM Telco Customer Churn (7,043 rows, 21 columns).
- **Training Partition:** 5,634 records (80% stratified holdout split, random state = 42).
- **Test Partition:** 1,409 records (20% stratified holdout split, random state = 42).
- **Feature Pipeline:**
  1. `TelcoFeatureEngineer`: Deterministic cleaning of `TotalCharges` (tenure=0 rule), calculation of `service_count` (0 to 8 active services), binning into `tenure_group`, and dropping `customerID`.
  2. `ColumnTransformer`:
     - Numeric: `SimpleImputer(strategy="median")`
     - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
  3. `XGBClassifier`:
     - `n_estimators`: 100
     - `max_depth`: 3
     - `learning_rate`: 0.08
     - `subsample`: 0.8
     - `colsample_bytree`: 0.8
     - `scale_pos_weight`: 2.77 (ratio of negative to positive training examples)
     - `eval_metric`: `"logloss"`

---

## 3. Evaluation Metrics (Actual Holdout Test Set)

Evaluated once on the untouched holdout test partition of **1,409 records** (1,035 Retained, 374 Churned):

| Metric | Selected Threshold ($\tau = 0.54$) | Default Threshold ($\tau = 0.50$) |
|:---|:---:|:---:|
| **ROC-AUC** | **0.8477** | 0.8477 |
| **PR-AUC (Avg Precision)** | **0.6646** | 0.6646 |
| **F1-Score** | **0.6281** | 0.6253 |
| **Recall (Sensitivity)** | **77.01%** (288 / 374) | **79.41%** (297 / 374) |
| **Precision** | **53.04%** (288 / 543) | **51.56%** (297 / 576) |
| **Accuracy** | **75.80%** | 74.73% |

### Confusion Matrix ($\tau = 0.54$)
- **True Negatives (TN):** 780
- **False Positives (FP):** 255
- **False Negatives (FN):** 86
- **True Positives (TP):** 288

---

## 4. Decision Threshold Rationale

The default threshold of $\tau = 0.50$ produces a high recall (79.41%) but lower precision (51.56%). By tuning the decision threshold on **training out-of-fold cross-validation predictions** to $\tau = 0.54$, the model achieves an optimal balance:
- Recovers **77.01%** of all churners (missing only 86 out of 374 at-risk subscribers).
- Reduces false alarms by 24 accounts compared to the default threshold, increasing precision to 53.04% and lifting F1-score to 0.6281.

---

## 5. Limitations & Production Considerations

1. **Static Cross-Sectional Nature:** The dataset lacks longitudinal timestamped time-series events (e.g., call drops, customer support ticket frequency).
2. **Concept Drift & Market Dynamics:** Telecommunications market offers, competitor pricing, and network upgrades change over time.
3. **Monitoring Requirements:** In a production setting, continuous monitoring of input feature drift (e.g., using Kolmogorov-Smirnov test for numeric features and Population Stability Index for categories) and prediction drift must be implemented.

