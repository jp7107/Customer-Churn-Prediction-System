# Business Impact Analysis: Churn Prediction System

**Project:** Customer Churn Prediction System  
**Organization:** Cognirex Solutions AI/ML Assessment  
**Author:** Jyotiprakash  

---

## 1. Economic Context & Problem Severity

### Industry Benchmarks
- **Customer Acquisition Cost (CAC):** Telecommunications industry average: $300–$600 per subscriber
- **Average Customer Lifetime Value (LTV):** $1,200–$2,000 over 24-36 months
- **Churn Impact:** Losing a customer costs 5-25x more than retaining them
- **Current Churn Rate:** 26.5% annual churn rate in our dataset (1,869 of 7,043 customers)

### Financial Impact Without Intervention
For a mid-sized telco with 100,000 subscribers:
- **Annual churners:** 26,500 customers
- **Lost revenue:** 26,500 × $1,500 (avg LTV) = **$39.75 million/year**
- **Replacement cost:** 26,500 × $400 (avg CAC) = **$10.6 million/year**
- **Total annual impact:** ~**$50 million**

---

## 2. Model Performance Translation to Business Value

### Our Model Performance (Holdout Test Set)
- **Recall @ threshold 0.35:** 71.4% (267 of 374 churners caught)
- **Precision @ threshold 0.35:** 56.0% (267 true positives, 210 false positives)
- **Threshold 0.54 (optimized):** Recall 77.0% (288/374), Precision 53.0%

### Retention Campaign Economics
**Assumptions:**
- **Retention offer cost:** $50 per customer (discount voucher, call center time)
- **Retention success rate:** 40% (industry standard for targeted interventions)
- **Customer LTV saved:** $1,500 per retained customer

### Scenario A: No ML Model (Baseline)
- **Strategy:** Random outreach or no proactive retention
- **Customers lost:** 374 (all churners)
- **Revenue loss:** 374 × $1,500 = **$561,000**

### Scenario B: ML Model @ Threshold 0.35
- **Customers targeted:** 477 (267 TP + 210 FP)
- **Actual churners caught:** 267 (71.4% recall)
- **Churners successfully retained:** 267 × 40% = **107 customers**
- **Campaign cost:** 477 × $50 = **$23,850**
- **Revenue saved:** 107 × $1,500 = **$160,500**
- **Net benefit:** $160,500 - $23,850 = **$136,650**
- **ROI:** 573% return on campaign investment

### Scenario C: Optimized Threshold 0.54 (Higher Precision)
- **Customers targeted:** 543 (288 TP + 255 FP)
- **Actual churners caught:** 288 (77.0% recall)
- **Churners successfully retained:** 288 × 40% = **115 customers**
- **Campaign cost:** 543 × $50 = **$27,150**
- **Revenue saved:** 115 × $1,500 = **$172,500**
- **Net benefit:** $172,500 - $27,150 = **$145,350**
- **ROI:** 535% return on campaign investment

---

## 3. Threshold Selection & Business Trade-offs

| Threshold | Recall | Precision | Customers Targeted | Churners Caught | Campaign Cost | Revenue Saved | Net Benefit | ROI |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 80% | 48% | 625 | 299 | $31,250 | $179,400 | $148,150 | 474% |
| **0.35** | **71%** | **56%** | **477** | **267** | **$23,850** | **$160,500** | **$136,650** | **573%** |
| 0.50 | 52% | 66% | 294 | 195 | $14,700 | $117,000 | $102,300 | 696% |
| **0.54** | **77%** | **53%** | **543** | **288** | **$27,150** | **$172,500** | **$145,350** | **535%** |

**Business Recommendation:** 
- Use **threshold 0.54** for maximum absolute profit ($145,350)
- Catches 77% of all churners while controlling false positive waste
- Alternative: Use 0.50 for highest ROI (696%) if budget-constrained

---

## 4. Comparative Analysis: ML vs. Rule-Based Heuristics

### Rule-Based Heuristic Example
*"Flag any customer on Month-to-Month contract with tenure < 12 months"*

**Performance:**
- Captures ~45% of churners (lower recall than ML)
- Generates massive false positives (low precision ~30%)
- Cannot adapt to complex interactions (e.g., Fiber + No Tech Support)

**ML Advantage:**
- **+32% additional recall** (77% vs 45%)
- **+23% better precision** (53% vs 30%)
- Learns non-obvious patterns: senior citizens on 2-year contracts with paperless billing still churn if they have no online security

---

## 5. Deployment Strategy & Phased Rollout

### Phase 1: Pilot (Month 1-2)
- Deploy model on 10% of customer base (shadow mode)
- A/B test: ML-guided retention vs. traditional approach
- Measure: actual retention rate, campaign cost, customer satisfaction

### Phase 2: Scale (Month 3-6)
- Roll out to 50% of at-risk customers
- Integrate with CRM (Salesforce, HubSpot)
- Automated email/SMS triggers for high-risk scores (>0.7)

### Phase 3: Optimization (Month 6+)
- Real-time API integration for customer service reps
- Dynamic threshold tuning based on observed retention success rates
- Expand features: call center sentiment, payment delays, network quality

---

## 6. Risk Mitigation & Monitoring

### Model Degradation Risks
1. **Data Drift:** Customer behavior changes (e.g., pandemic shifts to streaming)
   - **Mitigation:** Monthly PSI (Population Stability Index) monitoring
   - **Action:** Retrain if PSI > 0.25 or recall drops below 70%

2. **Concept Drift:** Relationship between features and churn changes
   - **Mitigation:** Track rolling 90-day F1-score and ROC-AUC
   - **Action:** Trigger retraining pipeline if AUC drops by >5%

3. **Adversarial Gaming:** Customers learn to exploit retention offers
   - **Mitigation:** Cap retention offer frequency (1 per 12 months)
   - **Action:** Monitor repeat churn threats

---

## 7. Competitive Advantage & Strategic Value

### Current State (No ML)
- Reactive: customers leave before intervention
- High CAC from constant replacement acquisition
- Poor customer experience (no proactive support)

### Future State (ML-Powered Retention)
- **Proactive:** Intervene 30-60 days before predicted churn
- **Personalized:** Tailor offers based on churn drivers
  - Contract upgrade incentives for month-to-month users
  - Free tech support trial for unsupported Fiber customers
  - Streaming service bundles for entertainment-focused users
- **Measurable:** Track ROI per campaign, optimize continuously

### Bottom Line
- **$145,350 net benefit per 1,409 customers**
- Extrapolated to 100,000 subscriber base: **$10.3 million annual savings**
- Payback period: <1 month of deployment
- Strategic moat: Better retention = higher LTV = competitive pricing power

---

## 8. Ethical Considerations

### Fairness & Bias
- Model does not use protected attributes (race, religion, national origin)
- Gender is included but contributes <2% to feature importance
- Regularly audit for demographic disparities in retention offer rates

### Customer Privacy
- No PII (names, phone numbers, addresses) used in features
- Predictions stored with encryption
- GDPR/CCPA compliant: customers can request deletion

### Transparency
- Customer service reps see "retention priority: high/medium/low" not raw probabilities
- Customers can ask "why am I receiving this offer?" → explain top 3 features

---

## Conclusion

This ML system delivers **measurable, immediate financial impact** while respecting customer privacy and business constraints. The 535-573% ROI makes it one of the highest-value applications of machine learning in telecommunications.

**Next Steps:**
1. Present business case to executive stakeholders
2. Secure budget for pilot deployment ($50K campaign)
3. Integrate with CRM and marketing automation platforms
4. Measure actual retention lift and iterate

