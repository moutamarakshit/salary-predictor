"""
Flask API + static frontend server for the STEM base-salary predictor.

Endpoints
---------
GET  /                 -> serves the frontend (backend/static/index.html)
GET  /api/metadata      -> dropdown options, numeric ranges, model metrics
POST /api/predict       -> { predicted_salary, range_low, range_high }

Run locally:
    python backend/app.py

The app loads model/salary_model.pkl and model/metadata.json, which are
produced by src/train.py. Run that script first.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "salary_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

REQUIRED_FIELDS = [
    "company",
    "title",
    "nation",
    "gender",
    "race",
    "education",
    "yearsofexperience",
    "yearsatcompany",
]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

_model = None
_metadata = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run "
                "`python src/train.py --data data/<your-file>.xlsx` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metadata():
    global _metadata
    if _metadata is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"No metadata found at {METADATA_PATH}. Run "
                "`python src/train.py --data data/<your-file>.xlsx` first."
            )
        _metadata = json.loads(METADATA_PATH.read_text())
    return _metadata


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/metadata")
def metadata():
    try:
        return jsonify(get_metadata())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/predict")
def predict():
    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    payload = request.get_json(silent=True) or {}
    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, "")]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        years_experience = float(payload["yearsofexperience"])
        years_at_company = float(payload["yearsatcompany"])
    except (TypeError, ValueError):
        return jsonify({"error": "yearsofexperience and yearsatcompany must be numbers"}), 400

    if years_experience < 0 or years_at_company < 0:
        return jsonify({"error": "Experience values cannot be negative"}), 400

    experience_before_company = max(years_experience - years_at_company, 0)

    row = pd.DataFrame(
        [
            {
                "company": str(payload["company"]),
                "title": str(payload["title"]),
                "nation": str(payload["nation"]),
                "gender": str(payload["gender"]),
                "race": str(payload["race"]),
                "education": str(payload["education"]),
                "yearsofexperience": years_experience,
                "yearsatcompany": years_at_company,
                "experience_before_company": experience_before_company,
            }
        ]
    )

    prediction = float(model.predict(row)[0])
    prediction = max(prediction, 0.0)

    meta = get_metadata()
    rmse = meta.get("metrics", {}).get("rmse", 0)

    return jsonify(
        {
            "predicted_salary": round(prediction, 2),
            "range_low": round(max(prediction - rmse, 0), 2),
            "range_high": round(prediction + rmse, 2),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
