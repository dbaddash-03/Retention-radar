"""Tests for the API edge cases, fairness audit, report generator and (optional) MLflow tracking."""
import json

import pandas as pd
import pytest

from app import create_app
from churn.data import load_data
from churn.fairness import audit, holdout_scores, to_markdown
from churn.report import results_block
from churn.train import train

ROW = dict(CreditScore=620, Geography="Japan", Gender="Female", Age=45, Tenure=3, Balance=80000,
           NumOfProducts=1, HasCrCard=1, IsActiveMember=0, EstimatedSalary=70000)


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    d = tmp_path_factory.mktemp("art2")
    m = train(data=load_data().sample(1500, random_state=3), out=str(d))
    return d, m


def test_meta_matches_ui_contract(art):
    d, m = art
    j = create_app(str(d / "model.joblib")).test_client().get("/api/meta").json
    assert {"geos", "rows", "churn_rate", "best", "holdout", "threshold", "board", "economics"} <= set(j)
    assert j["economics"]["profit_at_threshold"] >= 0 and j["board"][0]["auc"] > 0.5


def test_root_serves_ui_and_predict_is_monotone_sensible(art):
    d, _ = art
    c = create_app(str(d / "model.joblib")).test_client()
    assert b"Retention Radar" in c.get("/").data
    r = c.post("/api/predict", json=ROW).json
    assert r["contact"] == (r["risk"] >= r["threshold"])
    assert all(set(x) == {"feature", "value", "impact"} for x in r["drivers"])
    imp = [abs(x["impact"]) for x in r["drivers"]]
    assert imp == sorted(imp, reverse=True)


def test_batch_errors_are_json(art):
    d, _ = art
    c = create_app(str(d / "model.joblib")).test_client()
    assert c.post("/api/batch", data="").status_code == 400
    assert c.post("/api/batch", data="a,b\n1,2\n").json["error"].startswith("missing columns")
    assert c.post("/api/predict", data="not json").status_code == 400


def test_fairness_audit_covers_all_groups(art):
    d, m = art
    t = audit(holdout_scores(str(d / "model.joblib")))
    assert {"Geography", "Gender", "AgeBand"} == set(t.group)
    assert t.n.groupby(t.group).sum().nunique() == 1  # each attribute partitions the same hold-out set
    assert "Largest gaps" in to_markdown(t, m["threshold"])


def test_report_block_matches_metrics(art):
    _, m = art
    block = results_block(m)
    assert f"{m['rows']:,}" in block and str(m["holdout"]["auc"]) in block


def test_mlflow_logging(art, tmp_path):
    pytest.importorskip("mlflow")
    import mlflow
    from churn.tracking import log_run
    d, m = art
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_id = log_run(m, str(d), tracking_uri=uri)
    got = mlflow.tracking.MlflowClient(uri).get_run(run_id)
    assert got.data.metrics["holdout_auc"] == pytest.approx(m["holdout"]["auc"])
    assert got.data.params["best_model"] == m["best_model"]
