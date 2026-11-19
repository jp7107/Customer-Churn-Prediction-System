# System Architecture & Technical Workflow

**Project:** Customer Churn Prediction System  
**Organization:** Cognirex Solutions AI/ML Technical Assessment  
**Author:** Jyotiprakash  

---

## 1. Architectural Overview

The Telco Customer Churn Prediction System is designed as an end-to-end, modular, and leakage-safe machine learning engineering pipeline. It guarantees strict separation between offline training and online serving, while sharing deterministic preprocessing components to eliminate training-serving skew.

```mermaid
flowchart TD
    subgraph Data Layer
        RAW["Raw Dataset (data/raw/*.csv)"] --> PROV["Provenance & SHA-256 Validation"]
        PROV --> AUDIT["Data Quality Audit (validation.py)"]
    end

    subgraph Training & ML Lifecycle
        AUDIT --> SPLIT["Stratified 80/20 Train/Test Split"]
        SPLIT -->|Training Split (80%)| CV["Stratified 5-Fold Cross Validation"]
        
        subgraph Pipeline Composition
            FE["TelcoFeatureEngineer (features.py)"] --> PRE["ColumnTransformer (preprocessing.py)"]
            PRE --> EST["Classifier (LR, RF, XGBoost)"]
        end
        
        CV --> Pipeline Composition
        Pipeline Composition --> COMP["Model Benchmarking & Comparison"]
        COMP --> WINNER["Winning Model Selection: XGBoost (Weighted)"]
        WINNER --> THRESH["Threshold Optimization on OOF Val Data"]
        THRESH --> FULLTRAIN["Refit Pipeline on 100% Training Partition"]
        FULLTRAIN --> TESTEVAL["Single Final Evaluation on Holdout Test Split (20%)"]
        TESTEVAL --> ART["Serialized Pipeline (.joblib) + Metadata (.json)"]
    end

    subgraph Production Serving Layer
        ART --> LOADER["Inference Engine (inference.py)"]
        LOADER --> FASTAPI["FastAPI REST Application (api/main.py)"]
        CLIENT["API Client / Frontend / Curl"] -->|JSON Customer Payload| FASTAPI
        FASTAPI -->|Validated Schema| LOADER
        LOADER -->|Prediction & Churn Probability| FASTAPI
        FASTAPI -->|JSON Response| CLIENT
    end
```

---

## 2. Core Subsystems & Responsibilities

### 2.1 Configuration & Logging (`src/churn_prediction/config.py`, `logging_config.py`)
- Centralizes all paths, constants, random seeds (`RANDOM_STATE = 42`), test split ratios (`TEST_SIZE = 0.20`), and column definitions.
- Configures uniform structured logging output across all modules and API endpoints.

### 2.2 Data Integrity & Provenance (`src/churn_prediction/data.py`, `validation.py`)
- Verifies SHA-256 cryptographic hashes (`16320c9c1ec7...`) on the source dataset.
- Executes comprehensive data quality audits: checks for exact duplicates, customerID uniqueness, column types, target distributions, and detects blank string anomalies in `TotalCharges`.

### 2.3 Shared Feature Engineering (`src/churn_prediction/features.py`)
- Implements `TelcoFeatureEngineer`, an `sklearn.base.BaseEstimator` transformer.
- Safely converts `TotalCharges` using the domain-verified `tenure == 0` rule.
- Generates `service_count` (active subscribed services count, 0 to 8).
- Discretizes continuous `tenure` into 4 cohorts (`tenure_group`).
- Drops non-predictive identifiers (`customerID`) and target labels (`Churn`) to guarantee zero leakage.

### 2.4 Preprocessing & Estimators (`src/churn_prediction/preprocessing.py`)
- Assembles Scikit-learn `ColumnTransformer` pipelines.
- Imputes missing numeric values with fold medians and categorical values with fold modes.
- One-hot encodes nominal categorical features with `handle_unknown="ignore"`.
- Applies `StandardScaler` selectively to numeric features for linear models (Logistic Regression).

### 2.5 Training & Evaluation Engine (`src/churn_prediction/training.py`, `evaluation.py`)
- Orchestrates Stratified 5-Fold Cross Validation across candidate algorithms.
- Evaluates class imbalance mitigation strategies (`class_weight="balanced"` and `scale_pos_weight = 2.77`).
- Optimizes decision threshold ($\tau = 0.54$) exclusively on out-of-fold training predictions.
- Refits winning pipeline and evaluates once on the untouched holdout test partition.
- Serializes `models/churn_pipeline.joblib` and comprehensive `models/churn_pipeline_metadata.json`.

### 2.6 Serving & API Integration (`api/main.py`, `schemas.py`, `inference.py`)
- Production-grade FastAPI service exposing `/health`, `/model-info`, `/predict`, and `/predict/batch`.
- Pydantic models validate input bounds, numeric types, and categorical strings.
- Returns churn prediction string (`"Likely to Churn"` / `"Not Likely to Churn"`), explicit churn probability float, and decision threshold.

