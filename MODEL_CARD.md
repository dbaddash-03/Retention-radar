# Model card: Retention Radar

- **Task:** predict the probability a retail bank customer leaves (`Exited`).
- **Model:** Hist Gradient Boosting in a scikit-learn pipeline (scaling + one-hot), isotonic-calibrated. Trained 2026-10-06 09:26:09, seed 42, scikit-learn 1.8.0, data hash feb55f9965e9065b.
- **Data:** 3,999 customers, churn 21.2%. Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
- **Performance (20% hold-out):** AUC 0.8382, PR-AUC 0.6813, Brier 0.1109; at threshold 0.15: precision 0.4198, recall 0.8047.
- **Intended use:** prioritising retention outreach, as a demo. **Not for** credit, pricing or any decision that affects customers' access to services.
- **Risks:** Gender and Geography are included as features, so outcomes may differ across groups. A fairness audit is not done. The economics are assumptions. Risk is calibrated on one snapshot and may drift: check PSI on new batches.
