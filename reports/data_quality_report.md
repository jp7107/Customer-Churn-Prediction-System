# Telco Customer Churn — Data Provenance & Quality Report

**Project:** Customer Churn Prediction System  
**Organization:** Cognirex Solutions AI/ML Technical Assessment  
**Author:** Jyotiprakash  
**Date of Audit:** November 2026  

---

## 1. Dataset Provenance

| Attribute | Value / Specification |
|:---|:---|
| **Original Dataset Name** | IBM Telco Customer Churn |
| **Kaggle Source URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Local File Path** | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **SHA-256 Checksum** | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| **Total Record Count** | 7,043 rows |
| **Total Column Count** | 21 columns (20 features + 1 target) |
| **Target Column** | `Churn` ("Yes" = Churned, "No" = Retained) |

---

## 2. Target Variable Verification

| Target Class | Count | Percentage (%) | Representation |
|:---|:---:|:---:|:---|
| **No (Retained)** | 5,174 | 73.46% | Majority Class (Negative, mapped to 0) |
| **Yes (Churned)** | 1,869 | 26.54% | Minority Class (Positive, mapped to 1) |
| **Total** | **7,043** | **100.00%** | Imbalance Ratio: ~2.77 : 1 |

---

## 3. Data Integrity & Schema Audit

1. **Schema Integrity:**
   - All 20 expected raw features and the 1 target column (`Churn`) are present.
   - Identifier column: `customerID` (string format: `XXXX-XXXXX`).

2. **Duplicate Analysis:**
   - **Exact Duplicate Rows:** 0 rows (no exact full-row duplicates).
   - **Duplicate Customer IDs:** 0 (all 7,043 customer IDs are strictly unique).

3. **Missing Value Diagnostics & The `TotalCharges` Special Case:**
   - Standard Pandas `isna().sum()` reports 0 null values across all columns.
   - However, `TotalCharges` is stored as an `object` data type due to blank spaces (`" "`) in 11 records.
   - **Analysis of Blank Records:**
     - Blank count: exactly 11 records.
     - Distribution of `tenure` for all 11 records: `tenure == 0` (100% of cases).
     - Business Context: Customers with `tenure == 0` are brand new subscribers who have joined during the current billing cycle and have not yet been issued their first monthly invoice.
     - **Defensible Treatment Rule:** When `TotalCharges` is blank and `tenure == 0`, `TotalCharges` is deterministically imputed to `0.0`. If any other unparseable values occur, they are converted to `np.nan` and imputed using training-fold median imputation to guarantee zero leakage.

4. **Categorical Semantics:**
   - Values such as `No internet service` (for add-on security and streaming services) and `No phone service` (for `MultipleLines`) represent active domain semantics rather than missingness. They are explicitly preserved as distinct categories for one-hot encoding.
