# %% [markdown]
# # Bank churn: analysis, modelling and business case
# Run cell by cell in VS Code (Python Interactive) or convert with `python notebooks/make_ipynb.py`.
# Questions: who churns, which model is trustworthy, and how many customers should we contact?
# %%
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd())
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from churn.data import FEATS, TARGET, ROOT, load_data
from churn.train import SEED, profit
FIG = os.path.join(ROOT, "reports", "figures"); os.makedirs(FIG, exist_ok=True)
df = load_data(); print(df.shape, "churn rate:", round(df[TARGET].mean(), 3))
# %% [markdown]
# ## 1. Exploratory analysis
# Churn is a minority class, so accuracy alone is misleading. A model that predicts "stays" for everyone would still score well on accuracy. We use PR-AUC, recall and money instead.
# %%
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
df.groupby("Geography")[TARGET].mean().sort_values().plot.barh(ax=ax[0, 0], title="Churn by country")
df.groupby("NumOfProducts")[TARGET].mean().plot.bar(ax=ax[0, 1], title="Churn by products held")
df.groupby("IsActiveMember")[TARGET].mean().plot.bar(ax=ax[0, 2], title="Churn: active (1) vs inactive (0)")
df.groupby(pd.cut(df.Age, [17, 30, 40, 50, 60, 100]), observed=True)[TARGET].mean().plot.bar(ax=ax[1, 0], title="Churn by age band")
df.groupby(pd.qcut(df.Balance.rank(method="first"), 5, labels=False))[TARGET].mean().plot.bar(ax=ax[1, 1], title="Churn by balance quintile")
df.groupby("Gender")[TARGET].mean().plot.bar(ax=ax[1, 2], title="Churn by gender")
plt.tight_layout(); plt.savefig(f"{FIG}/eda.png", dpi=110); plt.close()
print(df.groupby("Geography")[TARGET].agg(["mean", "count"]).round(3).sort_values("mean", ascending=False).head(8))
# %% [markdown]
# ## 2. Model comparison (stratified 5-fold CV, in `churn/train.py`)
# Three model families compete on cross-validated PR-AUC. The winner is probability-calibrated and evaluated once on a held-out 20 percent.
# %%
art = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib")); m = art["metrics"]
print(pd.DataFrame(m["cv_leaderboard"])); print(m["holdout"])
# %% [markdown]
# ## 3. Hold-out diagnostics: PR curve, calibration, and the cost-based threshold
# %%
X, y = df[FEATS], df[TARGET]
_, Xte, _, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
p = art["model"].predict_proba(Xte)[:, 1]
pr, rc, _ = precision_recall_curve(yte, p); fp_, mp_ = calibration_curve(yte, p, n_bins=8)
grid = np.arange(.05, .96, .01); pf = [profit(yte.values, p >= t) for t in grid]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(rc, pr); ax[0].set(title="Precision-recall", xlabel="recall", ylabel="precision")
ax[1].plot(mp_, fp_, "o-"); ax[1].plot([0, 1], [0, 1], "--", c="grey"); ax[1].set(title="Calibration", xlabel="predicted", ylabel="observed")
ax[2].plot(grid, pf); ax[2].axvline(art["threshold"], c="r", ls="--"); ax[2].set(title="Profit vs threshold", xlabel="contact if risk >=", ylabel="profit")
plt.tight_layout(); plt.savefig(f"{FIG}/diagnostics.png", dpi=110); plt.close()
print("chosen threshold:", art["threshold"], m["economics"])
# %% [markdown]
# ## 4. Explainability
# Permutation importance is model-agnostic and cheap. SHAP gives per-customer explanations when installed (`pip install shap`).
# %%
pi = pd.Series(m["permutation_importance"]).sort_values(); pi.plot.barh(title="Permutation importance (PR-AUC drop)"); plt.tight_layout(); plt.savefig(f"{FIG}/importance.png", dpi=110); plt.close()
print(pi.sort_values(ascending=False).head(5))
try:
    import shap
    cal = art["model"].calibrated_classifiers_[0].estimator; pre, clf = cal[0], cal[-1]; Xs = Xte.sample(150, random_state=SEED)
    Xt = pre.transform(Xs); Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt; names = [n.split("__")[-1] for n in pre.get_feature_names_out()]
    ex = shap.Explainer(lambda d: clf.predict_proba(d)[:, 1], shap.maskers.Independent(Xt, max_samples=100), feature_names=names)
    shap.plots.beeswarm(ex(Xt[:100]), show=False); plt.tight_layout(); plt.savefig(f"{FIG}/shap.png", dpi=110); plt.close(); print("SHAP saved")
except Exception as e:
    print("SHAP skipped:", type(e).__name__, e)
# %% [markdown]
# ## 5. Conclusions and next steps
# - Inactive members, customers with 3 or more products, older customers and certain countries churn most (see `eda.png`).
# - Recall is favoured over precision because a wasted offer costs far less than a lost customer.
# - Next: time-based validation, uplift modelling (who is *persuadable*, not just likely to leave), fairness checks across gender and country, and monitoring PSI in production.
