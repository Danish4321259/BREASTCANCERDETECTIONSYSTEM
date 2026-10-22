"""
Deep Learning Module: Artificial Neural Network (ANN) for Breast Cancer Detection
Built using TensorFlow/Keras.

Architecture:
- Input shape: (30,)
- Dense(64, activation='relu'): Extracts non-linear combinations from 30 cytological features
- Dropout(0.3): Regularization to prevent co-adaptation of hidden units
- Dense(32, activation='relu'): Second non-linear abstraction layer
- Dropout(0.2): Secondary regularization
- Dense(1, activation='sigmoid'): Outputs calibrated malignancy probability P(Y=1|X)

Training:
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall, AUC
- EarlyStopping & ModelCheckpoint on validation loss
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.data_preprocessing import load_data, split_and_scale_data

def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for TensorFlow and NumPy for reproducible experiments."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_ann_model(input_dim: int = 30) -> keras.Model:
    """
    Constructs the Artificial Neural Network as specified:
    30 inputs -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", name="dense_hidden_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(32, activation="relu", name="dense_hidden_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(1, activation="sigmoid", name="output_malignancy_prob")
    ], name="BreastCancer_ANN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model

def train_and_evaluate_ann() -> dict:
    """
    Trains the ANN on scaled training data with early stopping and saves the model.
    Evaluates on held-out test data.
    """
    set_seeds(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print(">>> 1. Loading and splitting data for ANN...")
    X, y, feature_names = load_data()
    processed = split_and_scale_data(X, y, test_size=0.2, random_state=42)
    
    X_train_full = processed["X_train_scaled"]
    y_train_full = processed["y_train"]
    X_test = processed["X_test_scaled"]
    y_test = processed["y_test"]
    
    # Split training into train and validation sets (80% train, 20% val)
    # The test set remains strictly untouched until final evaluation!
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    
    print(f"ANN Train samples: {len(y_train)}, Val samples: {len(y_val)}, Test samples: {len(y_test)}")
    
    model = build_ann_model(input_dim=30)
    model.summary()
    
    checkpoint_path = "models/ann_model.keras"
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    print(">>> 2. Training ANN with Adam optimizer and Binary Cross-Entropy...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate on held-out test set
    print(">>> 3. Evaluating ANN on held-out test dataset...")
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    ann_metrics = {
        "model_name": "Artificial Neural Network (ANN)",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "epochs_trained": len(history.history["loss"])
    }
    
    print(f"\nANN Test Results: Acc: {accuracy:.4f} | Recall(Sens): {recall:.4f} | Specificity: {specificity:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Plot training history curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    # Loss plot
    axes[0].plot(epochs_range, history.history["loss"], label="Train Loss", color="#2563eb", linewidth=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    axes[0].set_title("ANN Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Accuracy plot
    axes[1].plot(epochs_range, history.history["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2)
    axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", color="#059669", linewidth=2, linestyle="--")
    axes[1].set_title("ANN Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/ann_training_curves.png", dpi=300)
    plt.close()
    
    # Save history json
    hist_dict = {
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "precision": [float(v) for v in history.history.get("precision", [])],
        "val_precision": [float(v) for v in history.history.get("val_precision", [])],
        "recall": [float(v) for v in history.history.get("recall", [])],
        "val_recall": [float(v) for v in history.history.get("val_recall", [])],
        "auc": [float(v) for v in history.history.get("auc", [])],
        "val_auc": [float(v) for v in history.history.get("val_auc", [])]
    }
    with open("results/ann_training_history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)
        
    with open("results/ann_evaluation_metrics.json", "w") as f:
        json.dump(ann_metrics, f, indent=2)
        
    return ann_metrics

if __name__ == "__main__":
    train_and_evaluate_ann()
