"""
Training and Evaluation Module for Machine Learning Models
Trains 6 classification algorithms on the Wisconsin Diagnostic Breast Cancer dataset:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Decision Tree Classifier
6. Gradient Boosting Classifier

Saves individual models and the best-performing model to disk.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.data_preprocessing import load_data, split_and_scale_data, save_preprocessor

def build_model_dictionary() -> Dict[str, Any]:
    """Returns the candidate machine learning models with tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        ),
        "Support Vector Machine": SVC(
            C=1.5, kernel="rbf", gamma="scale", probability=True, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=4, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="minkowski"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=6, criterion="gini", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    }

def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes standard and clinical metrics.
    In our mapping:
        1 = Malignant (cancerous, positive class)
        0 = Benign (non-cancerous, negative class)
    Recall for class 1 is Sensitivity.
    Recall for class 0 is Specificity.
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    return {
        "accuracy": float(round(accuracy, 4)),
        "precision": float(round(precision, 4)),
        "recall_sensitivity": float(round(recall, 4)),
        "specificity": float(round(specificity, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

def train_and_evaluate_all_ml() -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Orchestrates the entire ML training pipeline, cross-validation, test evaluation,
    and model serialization.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print(">>> 1. Loading and preprocessing WDBC dataset...")
    X, y, feature_names = load_data()
    processed = split_and_scale_data(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = processed["X_train_scaled"]
    X_test_scaled = processed["X_test_scaled"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]
    
    # Save the fitted scaler
    save_preprocessor(processed["scaler"], "models/preprocessing_pipeline.joblib")
    
    models = build_model_dictionary()
    results_list = []
    trained_models = {}
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(">>> 2. Training and evaluating 6 ML models with 5-fold CV...")
    for name, model in models.items():
        # Stratified Cross Validation on training fold
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strat, scoring="roc_auc")
        cv_auc_mean = float(np.mean(cv_scores))
        
        # Fit on full training set
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = round(time.time() - t0, 4)
        
        # Predict on test set
        t_infer0 = time.time()
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test_scaled)
            y_prob = 1 / (1 + np.exp(-decision))
        else:
            y_prob = y_pred.astype(float)
        infer_time_ms = round(((time.time() - t_infer0) / len(X_test_scaled)) * 1000, 4)
        
        metrics = calculate_clinical_metrics(y_test, y_pred, y_prob)
        metrics["model_name"] = name
        metrics["cv_roc_auc"] = round(cv_auc_mean, 4)
        metrics["train_time_sec"] = train_time
        metrics["infer_time_ms"] = infer_time_ms
        
        # Save individual model
        filename = f"models/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, filename)
        trained_models[name] = model
        
        results_list.append(metrics)
        print(f"  - {name:25s} | Acc: {metrics['accuracy']:.4f} | Recall(Sens): {metrics['recall_sensitivity']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | CV-AUC: {cv_auc_mean:.4f}")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results/model_comparison.csv", index=False)
    
    # Model Selection Strategy:
    # In clinical oncology, false negatives (missing malignant tumors) carry extreme morbidity.
    # Therefore, we prioritize Recall (Sensitivity) for Malignant tumor detection, breaking ties
    # by ROC-AUC and F1-score.
    results_df["selection_score"] = (
        results_df["recall_sensitivity"] * 0.40 +
        results_df["roc_auc"] * 0.35 +
        results_df["f1_score"] * 0.15 +
        results_df["accuracy"] * 0.10
    )
    best_row = results_df.sort_values(by="selection_score", ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    print(f"\n>>> Selected Best ML Model: {best_model_name} (Selection Score: {best_row['selection_score']:.4f})")
    
    # Save best model to designated best_model.joblib
    joblib.dump(trained_models[best_model_name], "models/best_model.joblib")
    
    # Export evaluation metadata
    summary = {
        "best_model": best_model_name,
        "features": feature_names,
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "positive_class": "Malignant (1)",
        "negative_class": "Benign (0)",
        "models": {r["model_name"]: r for r in results_list}
    }
    with open("results/ml_evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(">>> ML models trained and artifacts saved to /models and /results successfully.")
    return results_df, summary, best_model_name

if __name__ == "__main__":
    train_and_evaluate_all_ml()
