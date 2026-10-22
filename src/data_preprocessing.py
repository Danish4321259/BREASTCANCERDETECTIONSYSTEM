"""
Data Preprocessing Module for Breast Cancer Detection System
Loads Wisconsin Diagnostic Breast Cancer (WDBC) dataset from Scikit-Learn,
applies stratified train-test splitting, and fits a StandardScaler on training data only.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def load_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Loads Wisconsin Diagnostic Breast Cancer dataset.
    Returns:
        X (DataFrame): 569 samples with 30 features
        y (Series): Binary target mapped to 1 = Malignant (cancerous), 0 = Benign (non-cancerous)
        feature_names (List[str]): List of 30 feature names
    """
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    
    # Note on target mapping:
    # In sklearn's load_breast_cancer: 0 = 'malignant', 1 = 'benign'
    # We map 1 = Malignant (positive class for clinical sensitivity), 0 = Benign (negative class)
    y_raw = raw.target
    y = pd.Series(np.where(y_raw == 0, 1, 0), name="target")
    
    return X, y, list(raw.feature_names)

def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits data using stratified sampling (80:20) and scales features.
    Scaler is fitted ONLY on training data to strictly avoid data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }

def save_preprocessor(scaler: StandardScaler, filepath: str = "models/preprocessing_pipeline.joblib") -> None:
    """Saves fitted scaler to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)

def load_preprocessor(filepath: str = "models/preprocessing_pipeline.joblib") -> StandardScaler:
    """Loads fitted scaler from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fitted preprocessor not found at {filepath}")
    return joblib.load(filepath)

if __name__ == "__main__":
    print("Testing data preprocessing pipeline...")
    X, y, f_names = load_data()
    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"Class distribution - Malignant (1): {(y == 1).sum()}, Benign (0): {(y == 0).sum()}")
    
    processed = split_and_scale_data(X, y)
    print(f"Train size: {processed['X_train_scaled'].shape[0]}, Test size: {processed['X_test_scaled'].shape[0]}")
    save_preprocessor(processed['scaler'])
    print("Preprocessor saved successfully to models/preprocessing_pipeline.joblib.")
