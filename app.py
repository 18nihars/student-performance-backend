from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import sys
import os

from db import init_db, save_prediction, get_latest_predictions

# 🔥 Correct import path for Render & local
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model.explain import explain_single   # now works everywhere


app = Flask(__name__)
CORS(app)
init_db()

# 🔥 Load model correctly (Render uses working directory)
try:
    model = pickle.load(open(os.path.join("model", "best_advanced_model.pkl"), "rb"))


except Exception as e:
    print("Error loading model:", e)
    model = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return jsonify({
        "message": "Welcome — available endpoints: /health (GET), /predict (POST), /predictions (GET)",
        "note": "Use POST /predict with JSON data."
    })


@app.get("/predictions")
def predictions():
    rows = get_latest_predictions(20)

    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "timestamp": row[1],
            "attendance_percent": row[2],
            "study_hours": row[3],
            "internal_marks": row[4],
            "assignments_submitted": row[5],
            "activities_participation": row[6],
            "prediction": row[7],
            "confidence": row[8]
        })

    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        output = explain_single(data)

        # Save to DB
        save_prediction(
            data,
            output["prediction"],
            output["confidence"]
        )

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
