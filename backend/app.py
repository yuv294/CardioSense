"""CardioSense Flask API.

Endpoints (model = framingham | nhanes):
    GET  /api/health                     liveness + loaded models
    GET  /api/models                     the calculators on offer, with headline results
    GET  /api/schema?model=              the form for one calculator
    POST /api/predict?model=&mode=       {input: value} -> risk, explanation, what-if, published scores
    POST /api/batch?model=               {"records": [...]} -> one risk per record (clinician CSV upload)
    GET  /api/evidence?model=            evaluation, nested CV, ablation, CV and data reports

Run locally:  python app.py   (http://127.0.0.1:5000)
Production:   gunicorn app:app
If frontend/dist exists (npm run build), it is served from / as well, so one
process runs the whole app.
"""
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from cardiosense_api.services import model_service as svc

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
app = Flask(__name__, static_folder=None)
MAX_BATCH = 5000


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def _model():
    return svc.get(request.args.get("model", "framingham"))


@app.errorhandler(KeyError)
def _unknown(e):
    return jsonify({"error": str(e).strip("'\"")}), 404


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "models": list(svc.REGISTRY)})


@app.get("/api/models")
def models():
    return jsonify([m.info() for m in svc.REGISTRY.values()])


@app.get("/api/schema")
def schema():
    return jsonify(_model().schema())


@app.post("/api/predict")
def predict():
    m = _model()
    mode = request.args.get("mode", "patient")
    try:
        return jsonify(m.predict(request.get_json(silent=True) or {}, mode="clinician" if mode == "clinician" else "patient"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/batch")
def batch():
    m = _model()
    records = (request.get_json(silent=True) or {}).get("records", [])
    if not isinstance(records, list) or not records:
        return jsonify({"error": "Send {\"records\": [ {...}, ... ]}."}), 400
    if len(records) > MAX_BATCH:
        return jsonify({"error": f"At most {MAX_BATCH} rows per upload."}), 400
    return jsonify({"results": m.predict_many(records)})


@app.get("/api/evidence")
def evidence():
    return jsonify(_model().evidence())


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path):
    if not DIST.exists():
        return jsonify({"error": "Frontend not built. Run `npm run build` in frontend/, or use the Vite dev server."}), 404
    if path and (DIST / path).is_file():
        return send_from_directory(DIST, path)
    return send_from_directory(DIST, "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
