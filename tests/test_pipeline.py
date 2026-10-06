import numpy as np, pandas as pd, pytest
from churn.data import FEATS, load_data, validate
from churn.drift import make_ref, psi
from churn.train import train
from app import create_app

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("art"); df = load_data().sample(1200, random_state=1)
    m = train(data=df, out=str(d)); assert 0 < m["threshold"] < 1
    return create_app(str(d / "model.joblib")).test_client()

ROW = dict(CreditScore=620, Geography="Japan", Gender="Female", Age=45, Tenure=3, Balance=80000, NumOfProducts=1, HasCrCard=1, IsActiveMember=0, EstimatedSalary=70000)

def test_real_data_loads_without_identifiers():
    df = load_data(); assert len(df) > 3000 and "CustomerId" not in df and set(df["Exited"]) == {0, 1}

def test_validation_rejects_bad_input():
    with pytest.raises(ValueError, match="outside"): validate(pd.DataFrame([{**ROW, "Age": 5}]))
    with pytest.raises(ValueError, match="missing"): validate(pd.DataFrame([{"Age": 30}]))

def test_psi_zero_for_same_and_high_for_shifted():
    s = pd.DataFrame({c: np.random.default_rng(0).normal(50, 10, 2000) for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]})
    r = make_ref(s)["Age"]; assert psi(r, s["Age"].values) < .01 and psi(r, s["Age"].values + 30) > .25

def test_health_and_predict(client):
    assert client.get("/api/health").json["status"] == "ok"
    r = client.post("/api/predict", json=ROW).json; assert 0 <= r["risk"] <= 1 and len(r["drivers"]) == 6

def test_predict_rejects_invalid(client):
    assert client.post("/api/predict", json={**ROW, "Age": -1}).status_code == 400

def test_batch_scores_and_reports_drift(client):
    csv = pd.DataFrame([ROW] * 30).to_csv(index=False)
    r = client.post("/api/batch", data=csv).json; assert r["n"] == 30 and "Age" in r["drift"]
