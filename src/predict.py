"""
Prediction Inference Module for Breast Cancer Detection System
Loads trained models (best ML model or ANN) and fitted preprocessor,
validates 30 numerical inputs, standardizes them, and returns structured predictions.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
try:
    import tensorflow as tf
except Exception:
    tf = None

from src.data_preprocessing import load_preprocessor, FEATURE_NAMES

MODEL_MAPPING = {
    "best": "models/best_model.joblib",
    "logistic_regression": "models/logistic_regression.joblib",
    "svm": "models/support_vector_machine.joblib",
    "random_forest": "models/random_forest.joblib",
    "knn": "models/k-nearest_neighbors.joblib",
    "decision_tree": "models/decision_tree.joblib",
    "gradient_boosting": "models/gradient_boosting.joblib",
    "ann": "models/ann_model.keras"
}

# Normative reference values derived from WDBC dataset (mean and std for benign and malignant)
POPULATION_BENCHMARKS = {
    "mean radius": {"benign_mean": 12.15, "malignant_mean": 17.46, "unit": "µm"},
    "mean texture": {"benign_mean": 17.91, "malignant_mean": 21.60, "unit": "gray-level SD"},
    "mean perimeter": {"benign_mean": 78.08, "malignant_mean": 115.37, "unit": "µm"},
    "mean area": {"benign_mean": 462.79, "malignant_mean": 978.38, "unit": "µm²"},
    "mean smoothness": {"benign_mean": 0.092, "malignant_mean": 0.103, "unit": "ratio"},
    "mean compactness": {"benign_mean": 0.080, "malignant_mean": 0.145, "unit": "ratio"},
    "mean concavity": {"benign_mean": 0.046, "malignant_mean": 0.161, "unit": "ratio"},
    "mean concave points": {"benign_mean": 0.026, "malignant_mean": 0.088, "unit": "count/ratio"},
    "worst radius": {"benign_mean": 13.38, "malignant_mean": 21.13, "unit": "µm"},
    "worst area": {"benign_mean": 558.85, "malignant_mean": 1422.29, "unit": "µm²"},
    "worst concave points": {"benign_mean": 0.074, "malignant_mean": 0.182, "unit": "ratio"}
}

KEY5_FEATURES = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

class CancerPredictor:
    """Wrapper class managing preprocessor and model cache for inference."""
    
    def __init__(self, scaler_path: str = "models/preprocessing_pipeline.joblib"):
        self.scaler_path = scaler_path
        self._scaler = None
        self._scaler5 = None
        self._model5 = None
        self._models_cache = {}
        
    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = load_preprocessor(self.scaler_path)
        return self._scaler

    @property
    def scaler5(self):
        if self._scaler5 is None:
            if os.path.exists("models/key5_scaler.joblib"):
                self._scaler5 = joblib.load("models/key5_scaler.joblib")
        return self._scaler5

    @property
    def model5(self):
        if self._model5 is None:
            if os.path.exists("models/key5_model.joblib"):
                self._model5 = joblib.load("models/key5_model.joblib")
        return self._model5

    def get_model(self, model_key: str = "best"):
        if model_key in self._models_cache:
            return self._models_cache[model_key]
            
        model_path = MODEL_MAPPING.get(model_key, MODEL_MAPPING["best"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the models first.")
            
        if model_key == "ann" or model_path.endswith(".keras"):
            if os.path.exists("models/ann_weights.joblib"):
                model = joblib.load("models/ann_weights.joblib")
            elif tf is not None and os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
            else:
                # Fallback to best classical model if neither is present
                model = joblib.load(MODEL_MAPPING["best"])
        else:
            model = joblib.load(model_path)
            
        self._models_cache[model_key] = model
        return model

    def validate_inputs(self, raw_features: Union[List[float], np.ndarray, Dict[str, float]]) -> Tuple[np.ndarray, bool]:
        """
        Validates feature vector:
        - Supports 5 key features (radius, texture, perimeter, area, smoothness)
        - Or all 30 standard features
        Returns (features_array, is_5_features)
        """
        if isinstance(raw_features, dict):
            # Check if 5 key features are provided
            simple_keys = ['radius', 'texture', 'perimeter', 'area', 'smoothness']
            mean_simple_keys = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
            
            # Check if it has the 5 keys
            has_simple = all(k in raw_features for k in simple_keys)
            has_mean_simple = all(k in raw_features for k in mean_simple_keys)
            
            if has_simple and len(raw_features) == 5:
                ordered_vals = [raw_features[k] for k in simple_keys]
                features_array = np.array(ordered_vals, dtype=float).reshape(1, -1)
                return features_array, True
            elif has_mean_simple and len(raw_features) == 5:
                ordered_vals = [raw_features[k] for k in mean_simple_keys]
                features_array = np.array(ordered_vals, dtype=float).reshape(1, -1)
                return features_array, True
            else:
                ordered_values = []
                for name in FEATURE_NAMES:
                    if name not in raw_features:
                        raise ValueError(f"Missing required feature: '{name}'")
                    ordered_values.append(raw_features[name])
                features_array = np.array(ordered_values, dtype=float).reshape(1, -1)
                return features_array, False
        else:
            features_array = np.array(raw_features, dtype=float)

        if features_array.ndim == 1:
            if len(features_array) == 5:
                features_array = features_array.reshape(1, -1)
                is_5_features = True
            elif len(features_array) == 30:
                features_array = features_array.reshape(1, -1)
                is_5_features = False
            else:
                raise ValueError(f"Input features count mismatch: Expected 5 or 30 features, got {len(features_array)}.")
        elif features_array.ndim == 2:
            if features_array.shape[1] == 5:
                is_5_features = True
            elif features_array.shape[1] == 30:
                is_5_features = False
            else:
                raise ValueError(f"Expected 5 or 30 features per row, got {features_array.shape[1]}.")
        else:
            raise ValueError("Input features must be a 1D or 2D array-like object.")

        if not np.all(np.isfinite(features_array)):
            raise ValueError("Input features contain NaN, infinite, or non-numeric values.")

        return features_array, is_5_features

    def predict(
        self,
        features: Union[List[float], np.ndarray, Dict[str, float]],
        model_name: str = "best"
    ) -> Dict[str, Any]:
        """
        Performs end-to-end prediction:
        1. Input validation (5 or 30 features)
        2. Standardization
        3. Model scoring
        4. Structured interpretation with risk factors and population benchmarks
        """
        X_raw, is_5_features = self.validate_inputs(features)
        
        if is_5_features:
            X_scaled = self.scaler5.transform(X_raw)
            model = self.model5
            model_display_name = "Key 5-Feature SVM Classifier"
            probs = model.predict_proba(X_scaled)[0]
            prob_benign = float(probs[0])
            prob_malignant = float(probs[1])
            pred_class = int(prob_malignant >= 0.5)
        else:
            X_scaled = self.scaler.transform(X_raw)
            model = self.get_model(model_name)
            
            # Determine prediction and probability
            if model_name == "ann" or isinstance(model, dict) or (tf is not None and isinstance(model, getattr(tf, 'keras', object) and tf.keras.Model)):
                if isinstance(model, dict):
                    w1 = model['dense_hidden_1']['W']
                    b1 = model['dense_hidden_1']['b']
                    w2 = model['dense_hidden_2']['W']
                    b2 = model['dense_hidden_2']['b']
                    w3 = model['output_malignancy_prob']['W']
                    b3 = model['output_malignancy_prob']['b']
                    h1 = np.maximum(0, np.dot(X_scaled, w1) + b1)
                    h2 = np.maximum(0, np.dot(h1, w2) + b2)
                    raw_sig = 1.0 / (1.0 + np.exp(-(np.dot(h2, w3) + b3)))
                    prob_malignant = float(raw_sig.ravel()[0])
                else:
                    prob_malignant = float(model.predict(X_scaled, verbose=0).ravel()[0])
                pred_class = int(prob_malignant >= 0.5)
                prob_benign = 1.0 - prob_malignant
                model_display_name = "Artificial Neural Network (ANN)"
            else:
                pred_class = int(model.predict(X_scaled)[0])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)[0]
                    prob_benign = float(probs[0])
                    prob_malignant = float(probs[1])
                elif hasattr(model, "decision_function"):
                    dec = float(model.decision_function(X_scaled)[0])
                    prob_malignant = 1.0 / (1.0 + np.exp(-dec))
                    prob_benign = 1.0 - prob_malignant
                else:
                    prob_malignant = 1.0 if pred_class == 1 else 0.0
                    prob_benign = 1.0 - prob_malignant
                model_display_name = type(model).__name__

        label = "Malignant" if pred_class == 1 else "Benign"
        risk_level = (
            "High Risk" if prob_malignant >= 0.70
            else "Moderate Risk" if prob_malignant >= 0.40
            else "Low Risk"
        )
        
        # Analyze prominent contributing features against population benchmarks
        feature_insights = []
        raw_vals = X_raw[0]
        feature_list = KEY5_FEATURES if is_5_features else FEATURE_NAMES
        for idx, name in enumerate(feature_list):
            if name in POPULATION_BENCHMARKS:
                bench = POPULATION_BENCHMARKS[name]
                user_val = raw_vals[idx]
                b_diff = abs(user_val - bench["benign_mean"])
                m_diff = abs(user_val - bench["malignant_mean"])
                closer_to = "Malignant Profile" if m_diff < b_diff else "Benign Profile"
                feature_insights.append({
                    "feature": name,
                    "value": round(float(user_val), 4),
                    "unit": bench.get("unit", ""),
                    "benign_mean": bench["benign_mean"],
                    "malignant_mean": bench["malignant_mean"],
                    "closer_to": closer_to
                })
                
        return {
            "prediction": label,
            "is_malignant": bool(pred_class == 1),
            "probability_malignant": round(prob_malignant, 4),
            "probability_benign": round(prob_benign, 4),
            "confidence_percentage": round(max(prob_malignant, prob_benign) * 100, 2),
            "model_used": model_display_name,
            "model_key": model_name,
            "risk_assessment": risk_level,
            "key_features_analysis": feature_insights[:6],
            "disclaimer": (
                "DISCLAIMER: This system is an academic research prototype built on the Wisconsin Diagnostic Breast Cancer (WDBC) "
                "dataset for educational demonstration. It is NOT certified as a medical diagnostic device and does not "
                "constitute medical advice. Clinical decisions must always be made by a qualified pathologist and oncologist."
            )
        }

# Global predictor instance
_predictor_instance: Optional[CancerPredictor] = None

def get_predictor() -> CancerPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CancerPredictor()
    return _predictor_instance

def predict_sample(features: List[float], model_name: str = "best") -> Dict[str, Any]:
    """Convenience helper function for single-sample prediction."""
    return get_predictor().predict(features, model_name=model_name)
