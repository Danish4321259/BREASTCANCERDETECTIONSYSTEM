"""
Unit Tests for Prediction Inference Module
Tests model inference, probability calibration, output formatting,
input validation, and medical disclaimer attachment.
"""

import pytest
import numpy as np
from src.predict import CancerPredictor, get_predictor, FEATURE_NAMES
from src.data_preprocessing import load_data

@pytest.fixture(scope="module")
def predictor():
    """Initializes a shared predictor instance."""
    return get_predictor()

@pytest.fixture(scope="module")
def sample_cases():
    """Loads a known benign and known malignant sample directly from WDBC."""
    X, y, _ = load_data()
    # Find index of first malignant and first benign
    malig_idx = (y == 1).idxmax()
    benign_idx = (y == 0).idxmax()
    return {
        "malignant": list(X.iloc[malig_idx].values),
        "benign": list(X.iloc[benign_idx].values)
    }

def test_prediction_output_structure(predictor, sample_cases):
    """Verifies that prediction dictionary matches all schema fields."""
    res = predictor.predict(sample_cases["benign"], model_name="best")
    
    assert "prediction" in res
    assert "is_malignant" in res
    assert "probability_malignant" in res
    assert "probability_benign" in res
    assert "confidence_percentage" in res
    assert "model_used" in res
    assert "risk_assessment" in res
    assert "disclaimer" in res
    assert "key_features_analysis" in res
    
    # Check types and ranges
    assert res["prediction"] in ["Benign", "Malignant"]
    assert isinstance(res["is_malignant"], bool)
    assert 0.0 <= res["probability_malignant"] <= 1.0
    assert 0.0 <= res["probability_benign"] <= 1.0
    assert abs((res["probability_malignant"] + res["probability_benign"]) - 1.0) < 0.01
    assert "DISCLAIMER" in res["disclaimer"]

def test_benign_sample_classification(predictor, sample_cases):
    """Verifies that a typical benign case is classified as Benign."""
    res = predictor.predict(sample_cases["benign"], model_name="best")
    assert res["prediction"] == "Benign"
    assert res["is_malignant"] is False
    assert res["probability_benign"] > res["probability_malignant"]

def test_malignant_sample_classification(predictor, sample_cases):
    """Verifies that an aggressive malignant case is classified as Malignant."""
    res = predictor.predict(sample_cases["malignant"], model_name="best")
    assert res["prediction"] == "Malignant"
    assert res["is_malignant"] is True
    assert res["probability_malignant"] > res["probability_benign"]

def test_ann_model_prediction(predictor, sample_cases):
    """Verifies inference via the Deep Learning ANN model."""
    res = predictor.predict(sample_cases["malignant"], model_name="ann")
    assert res["prediction"] == "Malignant"
    assert "Neural Network" in res["model_used"] or "ANN" in res["model_used"]
    assert 0.0 <= res["probability_malignant"] <= 1.0

def test_input_validation_invalid_length(predictor):
    """Verifies rejection of vectors with fewer or more than 30 features."""
    with pytest.raises(ValueError, match="Input features count mismatch"):
        predictor.predict([1.0] * 29)
        
    with pytest.raises(ValueError, match="Input features count mismatch"):
        predictor.predict([1.0] * 31)

def test_input_validation_nan_or_inf(predictor):
    """Verifies rejection of NaN or infinite inputs."""
    invalid_nan = [1.0] * 29 + [float("nan")]
    with pytest.raises(ValueError, match="NaN, infinite"):
        predictor.predict(invalid_nan)
        
    invalid_inf = [1.0] * 29 + [float("inf")]
    with pytest.raises(ValueError, match="NaN, infinite"):
        predictor.predict(invalid_inf)

def test_dict_input_handling(predictor, sample_cases):
    """Verifies that input provided as a feature-name dictionary functions correctly."""
    dict_features = {name: val for name, val in zip(FEATURE_NAMES, sample_cases["benign"])}
    res = predictor.predict(dict_features, model_name="best")
    assert res["prediction"] == "Benign"
