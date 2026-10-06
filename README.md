# Retention Radar: bank churn predictor

An end-to-end churn project: reproducible training, honest evaluation, a business-cost decision rule, a tested API, drift monitoring, Docker and CI, with a custom radar-style interface.
Dataset and idea from [Soumilgit/XYZ-Bank-Customer-Churn-Predictor](https://github.com/Soumilgit/XYZ-Bank-Customer-Churn-Predictor) (MIT licence). The code here is a fresh implementation.

## Results (real data, 3,999 customers, 21.2% churn)
5-fold cross-validation on the training split:

| Model | AUC | PR-AUC | F1 |
| --- | --- | --- | --- |
| Logistic Regression | 0.8228 ± 0.0224 | 0.5704 | 0.5621 |
| Random Forest | 0.8467 ± 0.02 | 0.6415 | 0.5927 |
| Hist Gradient Boosting | 0.84 ± 0.029 | 0.6487 | 0.5793 |

Selected by PR-AUC (better than accuracy for an imbalanced target): **Hist Gradient Boosting**, then isotonic-calibrated.
Hold-out (20%, touched once): AUC 0.8382, PR-AUC 0.6813, Brier 0.1109.
At the cost-optimised threshold of 15% risk: recall 0.8047, precision 0.4198.
Assuming an offer costs 40, a customer is worth 800 and 30% accept: profit **19,680** versus 8,560 for contacting everyone and 0 for no outreach.
Top drivers (permutation importance): Age, NumOfProducts, IsActiveMember, Geography.

![diagnostics](reports/figures/diagnostics.png)

## Data science side
- `notebooks/analysis.py` (cells) and `analysis.ipynb`: EDA, imbalance, CV comparison, PR curve, calibration, profit curve, permutation importance and SHAP, conclusions. Figures land in `reports/figures/`.
- Leakage control: `RowNumber`, `CustomerId`, `Surname` are never features.
- Limitations: one dataset snapshot (no time split), no fairness audit yet, assumed business costs. See `MODEL_CARD.md`.

## ML engineering side
| Piece | Where |
| --- | --- |
| Schema and validation | `churn/data.py` |
| Training: CV, calibration, threshold, artifact, metrics, seed and data hash | `churn/train.py` -> `artifacts/` |
| Drift (PSI) | `churn/drift.py`, returned by `/api/batch` |
| API: `/api/predict`, `/api/batch`, `/api/health`, `/api/meta` | `app.py` |
| Tests (6, ~15 s) | `tests/` |
| Container and CI | `Dockerfile`, `.github/workflows/ci.yml` |

The model trains once (`python -m churn.train`) and is loaded from `artifacts/model.joblib` at serve time.

## Run
    python -m venv .venv ; source .venv/bin/activate        (Windows: .venv\Scripts\activate)
    pip install -r requirements-dev.txt
    python -m churn.train      # trains, writes artifacts/
    python app.py             # http://localhost:5000
    python -m pytest -q
    docker build -t retention-radar . && docker run -p 8000:8000 retention-radar

Batch scoring: upload a CSV with the 10 feature columns in the UI, or `curl -X POST --data-binary @file.csv -H "Content-Type: text/csv" localhost:5000/api/batch`.

Educational project, not for real lending or account decisions.
