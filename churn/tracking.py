"""Optional MLflow experiment tracking. Enable with:  python -m churn.train --mlflow
Then browse runs with:  mlflow ui   (reads ./mlruns by default)"""
import os
import re


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\- /]", "_", name).replace(" ", "_")


def log_run(metrics: dict, artifact_dir: str, experiment: str = "retention-radar", tracking_uri: str | None = None) -> str:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "best_model": metrics["best_model"], "rows": metrics["rows"], "seed": metrics["versions"]["seed"],
            "threshold": metrics["threshold"], "data_sha256": metrics["data_sha256"],
            **{f"assume_{k}": v for k, v in metrics["economics"]["assumptions"].items()},
        })
        mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics["holdout"].items()})
        mlflow.log_metrics({
            "profit_at_threshold": metrics["economics"]["profit_at_threshold"],
            "profit_contact_everyone": metrics["economics"]["profit_contact_everyone"],
        })
        for row in metrics["cv_leaderboard"]:
            n = _clean(row["model"])
            mlflow.log_metrics({f"cv_auc__{n}": row["auc"], f"cv_prauc__{n}": row["ap"], f"cv_f1__{n}": row["f1"]})
        for f in ("metrics.json", "model.joblib"):
            p = os.path.join(artifact_dir, f)
            if os.path.exists(p):
                mlflow.log_artifact(p)
        return run.info.run_id
