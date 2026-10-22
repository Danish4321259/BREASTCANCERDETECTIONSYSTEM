"""
Flask Web Application for Breast Cancer Detection System
Provides complete web pages (Home, Predict, Result, Dashboard, About)
and REST API endpoints for real-time inference and metric queries.
"""

import os
import json
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

from src.data_preprocessing import FEATURE_NAMES, load_data
from src.predict import get_predictor, MODEL_MAPPING, POPULATION_BENCHMARKS

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "breast_cancer_mca_project_2026_secret")

# Warm up predictor and models at application startup
predictor = get_predictor()

# Group features logically for clean 3-column UI layout
FEATURE_GROUPS = {
    "Nuclear Means": [f for f in FEATURE_NAMES if f.startswith("mean ")],
    "Standard Errors (SE)": [f for f in FEATURE_NAMES if f.endswith(" error")],
    "Worst / Largest Measurements": [f for f in FEATURE_NAMES if f.startswith("worst ")]
}

# Preload sample records for demonstration
_sample_cache = {}

def get_cached_sample(sample_type: str) -> List[float]:
    global _sample_cache
    if not _sample_cache:
        X, y, _ = load_data()
        malig_idx = (y == 1).idxmax()
        benign_idx = (y == 0).idxmax()
        _sample_cache["malignant"] = [float(v) for v in X.iloc[malig_idx].values]
        _sample_cache["benign"] = [float(v) for v in X.iloc[benign_idx].values]
    return _sample_cache.get(sample_type, _sample_cache["benign"])

@app.route("/")
def index():
    """Home landing page with project overview, architecture, and disclaimer."""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form page (GET) and form submission processing (POST)."""
    if request.method == "POST":
        model_choice = request.form.get("model_choice", "best")
        features_list = []
        errors = []
        
        # Collect and validate all 30 features
        for feature_name in FEATURE_NAMES:
            raw_val = request.form.get(feature_name)
            if raw_val is None or raw_val.strip() == "":
                errors.append(f"Missing required feature: '{feature_name}'")
                continue
            try:
                val = float(raw_val)
                features_list.append(val)
            except ValueError:
                errors.append(f"Invalid numeric input for '{feature_name}': {raw_val}")
                
        if errors:
            return render_template(
                "predict.html",
                feature_groups=FEATURE_GROUPS,
                feature_names=FEATURE_NAMES,
                benchmarks=POPULATION_BENCHMARKS,
                selected_model=model_choice,
                errors=errors,
                form_values=request.form
            )
            
        try:
            result = predictor.predict(features_list, model_name=model_choice)
            return render_template("result.html", result=result, model_choice=model_choice)
        except Exception as e:
            return render_template(
                "predict.html",
                feature_groups=FEATURE_GROUPS,
                feature_names=FEATURE_NAMES,
                benchmarks=POPULATION_BENCHMARKS,
                selected_model=model_choice,
                errors=[str(e)],
                form_values=request.form
            )

    return render_template(
        "predict.html",
        feature_groups=FEATURE_GROUPS,
        feature_names=FEATURE_NAMES,
        benchmarks=POPULATION_BENCHMARKS,
        selected_model="best",
        errors=None,
        form_values={}
    )

@app.route("/dashboard")
def dashboard():
    """Model performance dashboard loading saved metrics, confusion matrices, and ROC curves."""
    metrics_path = "results/evaluation_metrics.json"
    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            
    ann_history_path = "results/ann_training_history.json"
    ann_history = {}
    if os.path.exists(ann_history_path):
        with open(ann_history_path, "r") as f:
            ann_history = json.load(f)
            
    return render_template(
        "dashboard.html",
        metrics=metrics_data,
        ann_history=ann_history
    )

@app.route("/about")
def about():
    """Academic project documentation, dataset explanation, and clinical considerations."""
    return render_template("about.html")

# REST API Endpoints
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for prediction. Accepts:
    { "features": [30 numbers] or { "mean radius": 14.2, ... }, "model": "best"|"ann"|"svm"|... }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body provided."}), 400
        
    raw_features = data.get("features")
    if raw_features is None:
        # Check if features are provided directly as top-level keys
        if any(k in data for k in FEATURE_NAMES):
            raw_features = {k: data[k] for k in FEATURE_NAMES if k in data}
        else:
            return jsonify({"error": "Missing 'features' in request payload."}), 400
            
    model_choice = data.get("model", "best")
    
    try:
        result = predictor.predict(raw_features, model_name=model_choice)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": "Validation Error", "details": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Inference Error", "details": str(e)}), 500

@app.route("/api/sample/<sample_type>")
def api_sample(sample_type):
    """Returns sample record values ('benign' or 'malignant') for interactive demonstration."""
    sample_type = sample_type.lower()
    if sample_type not in ["benign", "malignant"]:
        return jsonify({"error": "Invalid sample type. Must be 'benign' or 'malignant'."}), 400
        
    vals = get_cached_sample(sample_type)
    sample_dict = {name: val for name, val in zip(FEATURE_NAMES, vals)}
    return jsonify({
        "sample_type": sample_type,
        "features_array": vals,
        "features_dict": sample_dict
    })

@app.route("/api/metrics")
def api_metrics():
    """Returns saved comparison metrics."""
    metrics_path = "results/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found."}), 404

@app.route("/api/models")
def api_models():
    """Returns supported models."""
    return jsonify({
        "available_models": list(MODEL_MAPPING.keys()),
        "default": "best",
        "best_model_name": "Support Vector Machine / ANN"
    })

if __name__ == "__main__":
    # Support running standalone on port 5000 or custom port
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
