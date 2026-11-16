# Screen-Recording Demonstration Script (3–5 Minutes)

**Project:** Customer Churn Prediction System  
**Role:** AI/ML Engineer Technical Assessment  
**Candidate:** Jyotiprakash  
**Target Video Duration:** 3 to 5 minutes  

---

## Video Walkthrough Timeline

### Segment 1: Introduction & Problem Context (0:00 – 0:45)
- **Visual:** Display the project root in the IDE / terminal and show the repository structure.
- **Narrative:**
  > *"Hello, my name is Jyotiprakash. In this video, I will demonstrate the end-to-end Customer Churn Prediction System built for the Cognirex Solutions AI/ML Engineer technical assessment. Our objective is to predict telecom subscriber churn on the IBM Telco Customer Churn dataset of 7,043 customers. The core priority of this project is machine learning correctness: zero preprocessing data leakage, handling class imbalance, optimizing decision thresholds, and serving reproducible predictions through a FastAPI endpoint."*

---

### Segment 2: Dataset Exploration & Key Empirical Insights (0:45 – 1:30)
- **Visual:** Open `notebooks/01_eda.ipynb` or the figures in `reports/figures/` (`target_distribution.png`, `churn_by_contract.png`, `churn_by_tenure_cohort.png`).
- **Narrative:**
  > *"Starting with Exploratory Data Analysis: our dataset contains 7,043 customer records with an imbalanced target—73.5% retained vs. 26.5% churned. This 2.77:1 ratio means accuracy alone is an uninformative metric.
  > A key data cleaning insight was finding 11 records where TotalCharges was stored as blank whitespace. We proved that all 11 occur when tenure equals 0—brand new subscribers. We handled this with a deterministic rule setting them to 0.0, avoiding any dropping of records.
  > In our behavioral analysis, contract type showed the strongest association: month-to-month contracts churn at 42.7%, while two-year contracts churn at only 2.8%. Furthermore, tenure cohort analysis revealed that first-year subscribers experience a 47.7% churn rate."*

---

### Segment 3: ML Pipeline, Model Comparison & Selection (1:30 – 2:30)
- **Visual:** Display `reports/figures/cv_model_comparison.png` and `reports/model_comparison.csv`.
- **Narrative:**
  > *"For model development, we benchmarked three distinct model families using 5-Fold Stratified Cross-Validation on the training partition only: Logistic Regression, Random Forest, and XGBoost. For each algorithm, we evaluated both unweighted and class-weighted strategies.
  > Standard unweighted models showed high accuracy but low recall around 48% to 53%, missing half of all churners. Weighting by class imbalance dramatically improved recall to ~80%.
  > XGBoost (Weighted) emerged as the superior model, achieving a cross-validation ROC-AUC of 0.8493 and PR-AUC of 0.6704.
  > We then tuned our decision threshold on training out-of-fold predictions to 0.54, optimizing the F1-score to balance precision and recall."*

---

### Segment 4: Test Evaluation & Confusion Matrix (2:30 – 3:15)
- **Visual:** Display `reports/figures/roc_curve_test.png`, `precision_recall_curve_test.png`, and `confusion_matrix_test.png`.
- **Narrative:**
  > *"We refit the winning pipeline on the full training split and evaluated it once on the untouched holdout test set of 1,409 records.
  > On the test set, our model achieved an ROC-AUC of 0.8477, PR-AUC of 0.6646, and an F1-score of 0.6281.
  > Looking at our test confusion matrix at the 0.54 threshold: out of 374 actual churners, our model successfully detected 288—a recall of 77.01%—while maintaining a precision of 53.04%. This provides strong operational value for retention interventions."*

---

### Segment 5: Automated Testing & Live FastAPI Demonstration (3:15 – 4:30)
- **Visual:** 
  1. In the terminal, execute `pytest -v` and highlight that all 24 automated tests pass.
  2. Start the FastAPI server: `uvicorn api.main:app --host 127.0.0.1 --port 8000`.
  3. Open browser to `http://127.0.0.1:8000/docs` (Swagger UI).
  4. Send a test prediction request for a high-risk customer (Month-to-month contract, tenure 1, Fiber Optic, no tech support).
  5. Show the response:
     ```json
     {
       "prediction": "Likely to Churn",
       "probability": 0.9374,
       "churn_probability": 0.9374,
       "decision_threshold": 0.54,
       "is_churn": true
     }
     ```
  6. Send a second test prediction request for a low-risk loyal customer (Two-year contract, tenure 65, DSL, support add-ons). Show the response:
     ```json
     {
       "prediction": "Not Likely to Churn",
       "probability": 0.0306,
       "churn_probability": 0.0306,
       "decision_threshold": 0.54,
       "is_churn": false
     }
     ```
- **Narrative:**
  > *"All 24 automated tests covering validation, feature engineering, pipeline inference, and API endpoints pass cleanly.
  > In the live API, our serialized pipeline calculates real probabilities. For a high-risk subscriber on a month-to-month contract with fiber and no support, the model outputs a 93.7% churn probability, classifying them as 'Likely to Churn'. For an established subscriber on a two-year contract, the churn probability drops to 3.1%, correctly classifying them as 'Not Likely to Churn'."*

---

### Segment 6: Conclusion (4:30 – 4:50)
- **Visual:** Display the repository README.md and wrap up.
- **Narrative:**
  > *"In summary, the system delivers an end-to-end, reproducible, leakage-safe churn prediction pipeline with verified test metrics, thorough documentation, and a working production API. Thank you for your consideration."*
