"""Print the README results block straight from artifacts/metrics.json so docs never drift from the model.
Run:  python -m churn.report"""
import json
import os

from .data import ROOT


def results_block(m: dict) -> str:
    h, e = m["holdout"], m["economics"]
    best = m["best_model"]
    out = [f"## Results (real data, {m['rows']:,} customers, {m['churn_rate'] * 100:.1f}% churn)",
           "5-fold cross-validation on the training split:", "", "| Model | AUC | PR-AUC | F1 |", "| --- | --- | --- | --- |"]
    out += [f"| {b['model']} | {b['auc']} ± {b['auc_std']} | {b['ap']} | {b['f1']} |" for b in m["cv_leaderboard"]]
    out += ["", f"Selected by PR-AUC (better than accuracy for an imbalanced target): **{best}**, then isotonic-calibrated.",
            f"Hold-out (20%, touched once): AUC {h['auc']}, PR-AUC {h['pr_auc']}, Brier {h['brier']}.",
            f"At the cost-optimised threshold of {m['threshold'] * 100:.0f}% risk: recall {h['recall']}, precision {h['precision']}.",
            f"Assuming an offer costs {e['assumptions']['offer_cost']:g}, a customer is worth {e['assumptions']['customer_value']:g} and "
            f"{e['assumptions']['save_rate'] * 100:.0f}% accept: profit **{e['profit_at_threshold']:,.0f}** versus "
            f"{e['profit_contact_everyone']:,.0f} for contacting everyone and 0 for no outreach.",
            "Top drivers (permutation importance): " + ", ".join(list(m["permutation_importance"])[:4]) + ".",
            f"<sub>Generated from artifacts/metrics.json (trained {m['trained_at']}, sklearn {m['versions']['sklearn']}, seed {m['versions']['seed']}).</sub>"]
    return "\n".join(out)


if __name__ == "__main__":
    print(results_block(json.load(open(os.path.join(ROOT, "artifacts", "metrics.json"), encoding="utf-8"))))
