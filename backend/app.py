"""CardioSense Flask API.

Endpoints:
    GET  /api/health      -> liveness
    GET  /api/schema      -> the input fields the form should render
    POST /api/predict     -> {feature: value, ...} -> risk result
    GET  /api/metrics      -> the honest test-set metrics for the About page

Run locally:  python app.py   (serves on http://127.0.0.1:5000)
Production:   gunicorn app:app
"""
import json
from pathlib import Path
from flask import Flask, request, jsonify

from cardiosense_api.services import model_service as svc

app = Flask(__name__)

# CORS: the frontend is served from a different origin in dev (Vite :5173) and
# on free-tier hosting (Netlify/Vercel front, Render/Railway back). We enable it
# with a tiny after_request hook so the backend needs no extra dependency. If
# flask-cors is installed it would do the same job; this keeps the app runnable
# and the deploy image smaller.
@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

REPORTS = Path(__file__).resolve().parent.parent / "reports"


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/schema")
def schema():
    return jsonify(svc.feature_schema())


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    # mode=clinician returns the richer breakdown; default is patient
    mode = request.args.get("mode", "patient")
    if mode not in ("patient", "clinician"):
        mode = "patient"
    try:
        result = svc.predict(payload, mode=mode)
    except Exception as e:  # never 500 on a bad field; tell the client what broke
        return jsonify({"error": f"Could not score input: {e}"}), 400
    return jsonify(result)


@app.get("/api/metrics")
def metrics():
    path = REPORTS / "evaluation.json"
    if not path.exists():
        return jsonify({"error": "metrics not generated yet"}), 404
    return jsonify(json.loads(path.read_text()))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
