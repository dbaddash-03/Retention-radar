"""Reproducible training: CV model selection, probability calibration, cost-based threshold, artifact + metrics.
Run:  python -m churn.train"""
import hashlib, json, os, time
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import BIN, CAT, FEATS, NUM, ROOT, TARGET, load_data
from .drift import make_ref

SEED = 42
# Business assumptions (currency-neutral units). Change them to match your bank.
OFFER_COST, CUSTOMER_VALUE, SAVE_RATE = 40.0, 800.0, 0.30

def pipeline(model):
    pre = ColumnTransformer([("n", StandardScaler(), NUM + BIN), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", model)])

def candidates():
    return {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", max_depth=4, learning_rate=.05, max_iter=200, random_state=SEED)}

def profit(y, flag):
    tp, fp = int(((y == 1) & flag).sum()), int(((y == 0) & flag).sum())
    return tp * (SAVE_RATE * CUSTOMER_VALUE - OFFER_COST) - fp * OFFER_COST

def train(data=None, out=None, path=None):
    df = data if data is not None else load_data(path)
    out = out or os.path.join(ROOT, "artifacts"); os.makedirs(out, exist_ok=True)
    X, y = df[FEATS], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED); board = []
    for name, m in candidates().items():
        r = cross_validate(pipeline(m), Xtr, ytr, cv=cv, scoring={"auc": "roc_auc", "ap": "average_precision", "f1": "f1"})
        board.append({"model": name, **{k: round(float(r["test_" + k].mean()), 4) for k in ("auc", "ap", "f1")},
                      "auc_std": round(float(r["test_auc"].std()), 4)})
    best = max(board, key=lambda b: b["ap"])["model"]          # PR-AUC suits an imbalanced target
    model = CalibratedClassifierCV(pipeline(candidates()[best]), method="isotonic", cv=5).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    grid = np.round(np.arange(.05, .96, .01), 2)
    pf = [profit(yte.values, p >= t) for t in grid]; thr = float(grid[int(np.argmax(pf))]); flag = p >= thr
    imp = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=5, random_state=SEED)
    metrics = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df), "churn_rate": round(float(y.mean()), 4),
        "best_model": best, "cv_leaderboard": board, "threshold": thr,
        "holdout": {"auc": round(roc_auc_score(yte, p), 4), "pr_auc": round(average_precision_score(yte, p), 4),
            "brier": round(brier_score_loss(yte, p), 4), "precision": round(precision_score(yte, flag, zero_division=0), 4),
            "recall": round(recall_score(yte, flag), 4), "f1": round(f1_score(yte, flag), 4)},
        "economics": {"assumptions": {"offer_cost": OFFER_COST, "customer_value": CUSTOMER_VALUE, "save_rate": SAVE_RATE},
            "profit_at_threshold": round(max(pf), 1), "profit_contact_everyone": round(profit(yte.values, np.ones(len(yte), bool)), 1),
            "profit_contact_nobody": 0.0},
        "permutation_importance": {f: round(float(v), 4) for f, v in sorted(zip(FEATS, imp.importances_mean), key=lambda t: -t[1])},
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "seed": SEED},
        "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]}
    base = {c: (Xtr[c].mode()[0] if c in CAT else float(Xtr[c].median())) for c in FEATS}
    joblib.dump({"model": model, "threshold": thr, "base": base, "ref": make_ref(Xtr), "metrics": metrics,
                 "geos": sorted(Xtr["Geography"].unique().tolist())}, os.path.join(out, "model.joblib"))
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    return metrics

if __name__ == "__main__":
    m = train(); print(json.dumps({k: m[k] for k in ("best_model", "threshold", "holdout", "economics")}, indent=2)); print(m["cv_leaderboard"])
