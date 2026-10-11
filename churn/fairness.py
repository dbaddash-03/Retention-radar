"""Group fairness audit on the SAME 20% hold-out the model was evaluated on (touched once, never trained on).
Run:  python -m churn.fairness      -> prints a table and writes reports/fairness.md

Reports, per group, at the deployed outreach threshold: selection rate (share flagged), precision, recall and
false-positive rate. Gaps are descriptive: small groups give noisy numbers, and the audit does not by itself
prove or rule out unfair treatment. Gender and Geography are model inputs, so review the result before any real use."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import FEATS, ROOT, TARGET, load_data
from .train import SEED

GROUPS = ["Geography", "Gender", "AgeBand"]


def holdout_scores(model_path: str | None = None, data_path: str | None = None) -> pd.DataFrame:
    art = joblib.load(model_path or os.path.join(ROOT, "artifacts", "model.joblib"))
    df = load_data(data_path)
    X, y = df[FEATS], df[TARGET]
    _, Xte, _, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)  # identical split to train()
    out = Xte.copy()
    out["y"] = yte.values
    out["risk"] = art["model"].predict_proba(Xte)[:, 1]
    out["flag"] = out["risk"] >= art["threshold"]
    out["AgeBand"] = pd.cut(out["Age"], [0, 30, 45, 60, 200], labels=["<=30", "31-45", "46-60", "60+"])
    return out


def audit(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        for val, d in scored.groupby(g, observed=True):
            tp = int((d.flag & (d.y == 1)).sum()); fp = int((d.flag & (d.y == 0)).sum())
            pos, neg = int((d.y == 1).sum()), int((d.y == 0).sum())
            rows.append({"group": g, "value": str(val), "n": len(d), "base_churn": round(pos / len(d), 3),
                         "selection_rate": round(float(d.flag.mean()), 3),
                         "precision": round(tp / max(tp + fp, 1), 3),
                         "recall": round(tp / pos, 3) if pos else np.nan,
                         "fpr": round(fp / neg, 3) if neg else np.nan})
    return pd.DataFrame(rows)


def to_markdown(t: pd.DataFrame, thr: float) -> str:
    lines = [f"# Fairness audit (hold-out, threshold {thr:.2f})", "",
             "| Group | Value | n | Base churn | Selection rate | Precision | Recall | FPR |", "|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r.group} | {r.value} | {r.n} | {r.base_churn} | {r.selection_rate} | {r.precision} | {r.recall} | {r.fpr} |")
    lines += ["", "Largest gaps within each attribute:", ""]
    for g, d in t.groupby("group"):
        lines.append(f"- **{g}**: selection-rate gap {d.selection_rate.max() - d.selection_rate.min():.3f}, "
                     f"recall gap {d.recall.max() - d.recall.min():.3f}, FPR gap {d.fpr.max() - d.fpr.min():.3f}")
    lines += ["", "Caveat: hold-out is ~800 customers, so small groups are noisy. Differences in selection rate can reflect",
              "real differences in base churn rather than bias; compare recall and FPR too."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    art = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib"))
    t = audit(holdout_scores())
    md = to_markdown(t, art["threshold"])
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    open(os.path.join(ROOT, "reports", "fairness.md"), "w", encoding="utf-8").write(md)
    print(md)
