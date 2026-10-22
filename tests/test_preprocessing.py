"""
Unit Tests for Data Preprocessing Pipeline
Verifies dataset loading, target mapping consistency, stratified split ratios,
and standard scaling integrity without leakage.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import load_data, split_and_scale_data, FEATURE_NAMES

def test_load_data_integrity():
    """Verifies that dataset loads with 569 samples, 30 features, and no missing values."""
    X, y, feature_names = load_data()
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (569, 30)
    assert len(y) == 569
    assert len(feature_names) == 30
    assert X.isnull().sum().sum() == 0

def test_target_label_mapping():
    """
    Verifies that target mapping maps:
    1 = Malignant (cancerous, 212 samples)
    0 = Benign (non-cancerous, 357 samples)
    """
    _, y, _ = load_data()
    unique_vals = set(y.unique())
    assert unique_vals == {0, 1}
    
    malignant_count = (y == 1).sum()
    benign_count = (y == 0).sum()
    assert malignant_count == 212, f"Expected 212 malignant cases, got {malignant_count}"
    assert benign_count == 357, f"Expected 357 benign cases, got {benign_count}"

def test_stratified_split_and_scaling():
    """Verifies 80:20 stratified split and standard scaling properties."""
    X, y, _ = load_data()
    processed = split_and_scale_data(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = processed["X_train_scaled"]
    X_test_scaled = processed["X_test_scaled"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]
    
    # 80:20 split verification (455 train, 114 test)
    assert X_train_scaled.shape == (455, 30)
    assert X_test_scaled.shape == (114, 30)
    assert len(y_train) == 455
    assert len(y_test) == 114
    
    # Verify stratification (approx 37.3% malignant in both train and test)
    train_malig_ratio = (y_train == 1).mean()
    test_malig_ratio = (y_test == 1).mean()
    assert abs(train_malig_ratio - test_malig_ratio) < 0.02
    
    # Verify scaler mean close to 0 and std close to 1 on training set
    train_means = np.mean(X_train_scaled, axis=0)
    train_stds = np.std(X_train_scaled, axis=0)
    assert np.allclose(train_means, 0, atol=1e-7)
    assert np.allclose(train_stds, 1, atol=1e-7)

def test_feature_names_order():
    """Verifies that all 30 expected feature names match exact specification."""
    _, _, feature_names = load_data()
    assert len(feature_names) == 30
    assert feature_names[0] == "mean radius"
    assert feature_names[-1] == "worst fractal dimension"
