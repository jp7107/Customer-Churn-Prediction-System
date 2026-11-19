# Telco Customer Churn — Exploratory Data Analysis Summary

**Project:** Customer Churn Prediction System  
**Organization:** Cognirex Solutions AI/ML Technical Assessment  
**Author:** Jyotiprakash  

---

## 1. Key Business & Empirical Findings

Our exploratory data analysis was conducted on all 7,043 customer records. All findings presented below are empirical and evidence-based.

### Finding 1: Severe Imbalance in Customer Retention
- **73.46%** of customers in the dataset remain active, while **26.54%** have churned.
- *ML Implication:* Evaluating models strictly by accuracy would yield an inflated ~73.5% score simply by predicting that no customer ever churns. Churn models must be prioritized on **ROC-AUC, PR-AUC, F1-Score, and Recall**.

### Finding 2: Contract Type is the Primary Association with Churn
- **Month-to-month contracts:** **42.71%** churn rate (3,875 customers).
- **One-year contracts:** **11.27%** churn rate (1,473 customers).
- **Two-year contracts:** **2.83%** churn rate (1,695 customers).
- *Insight:* Month-to-month subscribers exhibit over 15x higher churn probability than two-year subscribers. Long-term contract commitments serve as a powerful retention stabilizer.

### Finding 3: The "First-Year Fragility" Phenomenon
- Customers in the **0–12 Month** cohort exhibit a **47.68%** churn rate.
- Customers in the **13–24 Month** cohort drop to **28.71%**.
- Customers in the **25–48 Month** cohort drop to **19.38%**.
- Customers in the **49–72 Month** cohort experience a churn rate of only **9.51%**.
- *Insight:* Early onboarding (months 1 through 6) is the most critical window for retention interventions.

### Finding 4: Fiber Optic and Support Service Disconnect
- Customers with **Fiber Optic** internet churn at **41.89%**, despite paying significantly higher monthly charges ($70–$110).
- Customers with **DSL** internet churn at **18.96%**.
- In-depth sub-group analysis reveals that Fiber Optic customers **without Tech Support or Online Security** churn at over **49%**, whereas Fiber Optic customers **with Tech Support** churn at less than **22%**.
- *Insight:* High billing without adequate technical resolution drives subscriber dissatisfaction.

### Finding 5: Payment Method Dynamics
- Customers paying via **Electronic Check** churn at **45.29%** (1,071 out of 2,365).
- Customers paying via **Automatic Credit Card** (15.24%), **Automatic Bank Transfer** (16.71%), or **Mailed Check** (19.11%) exhibit substantially lower attrition.
- *Insight:* Friction in monthly manual payments correlates strongly with churn.

---

## 2. Generated EDA Figures

All figures have been rendered at 300 DPI into `reports/figures/`:

1. `target_distribution.png`: Shows class counts (5,174 No vs. 1,869 Yes) and exact percentages.
2. `numeric_distributions.png`: Kernel density estimates of `tenure`, `MonthlyCharges`, and `TotalCharges` stratified by churn status.
3. `churn_by_contract.png`: Barplot of churn rates across Contract categories (Month-to-month, One year, Two year).
4. `churn_by_tenure_cohort.png`: Churn attrition across discretized customer life stages.
5. `churn_by_internet_techsupport.png`: Bivariate interaction of internet connection type and technical support availability.
6. `correlation_heatmap.png`: Spearman rank correlations between numeric/engineered variables and churn.

