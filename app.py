"""Retention Radar API + UI.

Endpoints (the contract used by index.html and tests/):
  GET  /               the radar UI
  GET  /api/health     liveness
  GET  /api/meta       model + business summary for the UI header and leaderboard
  POST /api/predict    JSON customer row -> risk, top drivers, outreach decision
  POST /api/batch      CSV body -> scored summary, top risks, PSI drift report

Run:  python app.py          (dev, http://localhost:5000)
      gunicorn app:app       (production; the model loads lazily on the first request)
"""
import io
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from churn.data import FEATS, ROOT, validate
from churn.drift import report

MODEL_PATH = os.path.join(ROOT, "artifacts", "model.joblib")
MAX_BATCH_ROWS = 50_000
N_DRIVERS = 6


def create_app(model_path: str | None = None) -> Flask:
    art = joblib.load(model_path or MODEL_PATH)
    model, thr, base, ref, m = art["model"], art["threshold"], art["base"], art["ref"], art["metrics"]

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

    def bad(msg: str, code: int = 400):
        return jsonify({"error": msg}), code

    @app.get("/")
    def index():
        return send_from_directory(ROOT, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model": m["best_model"], "trained_at": m["trained_at"]})

    @app.get("/api/meta")
    def meta():
        return jsonify({
            "geos": art["geos"], "rows": m["rows"], "churn_rate": round(m["churn_rate"] * 100, 1),
            "best": m["best_model"], "holdout": m["holdout"], "threshold": thr,
            "board": m["cv_leaderboard"], "economics": m["economics"],
            "importance": m["permutation_importance"],
        })

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return bad("send a JSON object with the 10 customer fields")
        try:
            row = validate(pd.DataFrame([payload]))
        except ValueError as e:
            return bad(str(e))
        # Driver impact = risk(this customer) - risk(same customer with ONE feature set to the
        # typical value). Positive impact => that feature pushes risk up versus a typical customer.
        variants = [row]
        for f in FEATS:
            v = row.copy()
            v[f] = base[f]
            variants.append(v)
        p = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
        risk = float(p[0])
        drivers = sorted(
            ({"feature": f, "value": row[f].iloc[0].item() if hasattr(row[f].iloc[0], "item") else row[f].iloc[0],
              "impact": round(risk - float(p[i + 1]), 4)} for i, f in enumerate(FEATS)),
            key=lambda d: -abs(d["impact"]),
        )[:N_DRIVERS]
        return jsonify({"risk": round(risk, 4), "contact": bool(risk >= thr), "threshold": thr, "drivers": drivers})

    @app.post("/api/batch")
    def batch():
        raw = request.get_data(as_text=True)
        if not raw.strip():
            return bad("empty body: send a CSV with a header row")
        try:
            df = pd.read_csv(io.StringIO(raw))
        except Exception:
            return bad("could not parse CSV")
        if len(df) == 0:
            return bad("CSV has no rows")
        if len(df) > MAX_BATCH_ROWS:
            return bad(f"too many rows (max {MAX_BATCH_ROWS})")
        try:
            x = validate(df)
        except ValueError as e:
            return bad(str(e))
        risk = model.predict_proba(x)[:, 1]
        ids = df["CustomerId"].tolist() if "CustomerId" in df else list(range(1, len(df) + 1))
        order = risk.argsort()[::-1][:10]
        return jsonify({
            "n": int(len(df)), "flagged": int((risk >= thr).sum()), "mean_risk": round(float(risk.mean()), 4),
            "drift": report(ref, x), "top": [{"id": ids[i], "risk": round(float(risk[i]), 4)} for i in order],
        })

    @app.errorhandler(413)
    def too_big(_):
        return bad("file too large (max 5 MB)", 413)

    return app


_real = None


def app(environ, start_response):
    """WSGI entry point for gunicorn. Lazy so importing this module never needs a trained model."""
    global _real
    if _real is None:
        _real = create_app()
    return _real(environ, start_response)


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
