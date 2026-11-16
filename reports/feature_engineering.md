# Telco Customer Churn — Feature Engineering Documentation

**Project:** Customer Churn Prediction System  
**Organization:** Cognirex Solutions AI/ML Technical Assessment  
**Author:** Jyotiprakash  

---

## 1. Feature Engineering Rationale & Philosophy

Feature engineering was implemented under strict ML safety constraints:
1. **Zero Data Leakage:** All transformations are deterministic or fit strictly on training partitions. No statistics from the test partition are ever used.
2. **Deterministic Parity:** Feature generation is encapsulated in `src.churn_prediction.features.TelcoFeatureEngineer`, an `sklearn.base.BaseEstimator` transformer that operates identically in training, cross-validation, and API inference.
3. **Traceability over Arbitrary Proliferation:** Rather than generating hundreds of ungrounded interaction variables, features were created based on concrete empirical patterns identified during Exploratory Data Analysis.

---

## 2. Engineered Features

### 2.1 `service_count` (Discrete Numeric)
- **Definition:** The integer sum of distinct active services to which a customer subscribes (range: 0 to 8 services).
- **Constituent Services Evaluated:**
  - `PhoneService` (`Yes` = 1, `No` = 0)
  - `MultipleLines` (`Yes` = 1, `No` / `No phone service` = 0)
  - `InternetService` (`DSL` or `Fiber optic` = 1, `No` = 0)
  - `OnlineSecurity` (`Yes` = 1, `No` / `No internet service` = 0)
  - `OnlineBackup` (`Yes` = 1, `No` / `No internet service` = 0)
  - `DeviceProtection` (`Yes` = 1, `No` / `No internet service` = 0)
  - `TechSupport` (`Yes` = 1, `No` / `No internet service` = 0)
  - `StreamingTV` (`Yes` = 1, `No` / `No internet service` = 0)
  - `StreamingMovies` (`Yes` = 1, `No` / `No internet service` = 0)
- **Domain Rationale:** High-service subscribers possess higher switching costs (ecosystem lock-in) and exhibit lower churn compared to single-service users.
- **Handling of Special Categories:** Crucially, `"No internet service"` and `"No phone service"` are strictly counted as `0` rather than treated as missing values or affirmative subscriptions.

### 2.2 `tenure_group` (Ordinal / Nominal Categorical)
- **Definition:** Discretization of continuous `tenure` (in months) into four customer life-cycle cohorts:
  - `0-12 Month`: High-risk onboarding cohort.
  - `13-24 Month`: Transition / second-year cohort.
  - `25-48 Month`: Established subscribers.
  - `49-72 Month`: Mature / loyal cohort.
- **Domain Rationale:** Linear models cannot capture the sharp non-linear decay of churn risk after the first year without non-linear binning or non-linear basis expansions.

### 2.3 Safe `TotalCharges` Numeric Transformation
- **Rule:** Blank strings (`" "`) are verified to occur exclusively when `tenure == 0`.
- **Implementation:** Replaces blank values with `0.0` when `tenure == 0`, and converts all other valid representations to standard float64. Any anomalous non-numeric strings are converted to `np.nan` and imputed via fold-specific median imputation.

---

## 3. Predictor Hygiene & Leakage Prevention

- **`customerID` Exclusion:** `customerID` is an arbitrary hexadecimal string identifier assigned upon account creation. It carries zero predictive customer behavior information. Retaining it would allow tree models to memorize customer hashes. It is strictly excluded from all model pipelines and retained only as optional trace metadata in API responses.
- **Target Exclusion:** `Churn` is mapped to binary integer format (`Yes` = 1, `No` = 0) in one centralized location and is strictly segregated from input feature matrices ($X$).
