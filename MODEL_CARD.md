# Model card: Retention Radar

- **Task:** predict the probability a retail bank customer leaves (`Exited`).
- **Model:** Hist Gradient Boosting in a scikit-learn pipeline (scaling + one-hot), isotonic-calibrated. Seed 42, scikit-learn 1.8.0 (pinned), data hash feb55f9965e9065b. Exact training time and versions are in `artifacts/metrics.json`.
- **Data:** 3,999 customers, churn 21.2%. Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
- **Performance (20% hold-out):** AUC 0.8382, PR-AUC 0.6813, Brier 0.1109; at threshold 0.15: precision 0.4198, recall 0.8047.
- **Intended use:** prioritising retention outreach, as a demo. **Not for** credit, pricing or any decision that affects customers' access to services.

## Fairness audit (hold-out, threshold 0.15; full table in `reports/fairness.md`)
- **Age band** shows the largest gaps. Customers aged 46-60 are flagged 78.5% of the time (base churn 54.1%) versus 12.0% for those 30 or under (base churn 5.1%). The model is following real differences in churn in this dataset, but the false-positive rate is also much higher for 46-60 (0.565 vs 0.100), so many non-churners in that band would receive offers. The 60+ band has only 38 customers, so its numbers are noisy.
- **Geography:** USA customers are flagged 55.6% versus about 35% for Australia and Japan (base churn 29.4% vs 18.1%); false-positive rate 0.424 vs about 0.26.
- **Gender:** small gaps (selection rate 0.372 vs 0.441, recall 0.813 vs 0.798), in line with base churn 0.180 vs 0.245.

## Risks and limitations
- Age, Gender and Geography are model inputs, so outreach intensity differs across groups. For an offer-only retention use the harm is low, but any real deployment would need review, and decisions affecting access to services must not use this model.
- The audit is descriptive on ~800 hold-out customers; it neither proves nor rules out unfair treatment.
- The economics (offer cost 40, customer value 800, 30% accept) are assumptions, not bank data.
- Risk is calibrated on one snapshot with no time split and may drift: check PSI on new batches.
