"""
Comprehensive Model Evaluation and Comparison Module
Evaluates all trained ML algorithms and the Deep Learning ANN on held-out test data.
Generates:
1. Confusion matrices for all models
2. ROC Curves comparison plot
3. Accuracy comparison bar chart
4. Multi-metric comparison (Precision, Recall/Sensitivity, F1-Score)
5. Comprehensive comparison table (model_comparison.csv & evaluation_metrics.json)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import tensorflow as tf

from src.data_preprocessing import load_data, split_and_scale_data

def evaluate_all() -> pd.DataFrame:
    """Evaluates ML models + ANN and produces comparison charts and reports."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    
    X, y, feature_names = load_data()
    processed = split_and_scale_data(X, y, test_size=0.2, random_state=42)
    X_test_scaled = processed["X_test_scaled"]
    y_test = processed["y_test"]
    
    ml_model_files = {
        "Logistic Regression": "models/logistic_regression.joblib",
        "Support Vector Machine": "models/support_vector_machine.joblib",
        "Random Forest": "models/random_forest.joblib",
        "K-Nearest Neighbors": "models/k-nearest_neighbors.joblib",
        "Decision Tree": "models/decision_tree.joblib",
        "Gradient Boosting": "models/gradient_boosting.joblib"
    }
    
    results = []
    predictions_dict = {}
    probabilities_dict = {}
    
    # 1. Evaluate classical ML models
    for name, path in ml_model_files.items():
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(X_test_scaled)
                y_prob = 1 / (1 + np.exp(-dec))
            else:
                y_prob = y_pred.astype(float)
                
            predictions_dict[name] = y_pred
            probabilities_dict[name] = y_prob
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            results.append({
                "Model": name,
                "Type": "Machine Learning",
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall (Sensitivity)": round(rec, 4),
                "Specificity": round(spec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
            })

    # 2. Evaluate Deep Learning ANN
    ann_path = "models/ann_model.keras"
    if os.path.exists(ann_path):
        ann_model = tf.keras.models.load_model(ann_path)
        ann_prob = ann_model.predict(X_test_scaled).ravel()
        ann_pred = (ann_prob >= 0.5).astype(int)
        
        name = "Artificial Neural Network (ANN)"
        predictions_dict[name] = ann_pred
        probabilities_dict[name] = ann_prob
        
        cm = confusion_matrix(y_test, ann_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_test, ann_pred)
        prec = precision_score(y_test, ann_pred, zero_division=0)
        rec = recall_score(y_test, ann_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_test, ann_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, ann_prob)
        
        results.append({
            "Model": name,
            "Type": "Deep Learning",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
        })
        
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["Recall (Sensitivity)", "ROC-AUC", "F1-Score"], ascending=False)
    
    # Save CSV and JSON
    df_results.to_csv("results/model_comparison.csv", index=False)
    
    metrics_json = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
        "test_samples": len(y_test),
        "test_distribution": {
            "malignant": int(sum(y_test == 1)),
            "benign": int(sum(y_test == 0))
        },
        "models": df_results.to_dict(orient="records"),
        "best_overall_model": df_results.iloc[0]["Model"],
        "clinical_focus": (
            "Sensitivity (Recall for Malignant cases) is prioritized because false negatives "
            "delay critical oncology interventions, whereas false positives lead to standard follow-up biopsies."
        )
    }
    with open("results/evaluation_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
        
    # Generate Charts
    _plot_roc_curves(y_test, probabilities_dict)
    _plot_confusion_matrices(y_test, predictions_dict)
    _plot_metric_comparisons(df_results)
    
    print(">>> All evaluation charts and reports generated successfully.")
    return df_results

def _plot_roc_curves(y_test: np.ndarray, probabilities_dict: dict) -> None:
    plt.figure(figsize=(9, 7))
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5"]
    
    for i, (name, y_prob) in enumerate(probabilities_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", alpha=0.6)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves - Model Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=300)
    plt.savefig("static/images/roc_curves.png", dpi=300)
    plt.close()

def _plot_confusion_matrices(y_test: np.ndarray, predictions_dict: dict) -> None:
    n_models = len(predictions_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    class_labels = ["Benign (0)", "Malignant (1)"]
    
    for idx, (name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_labels, yticklabels=class_labels, ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        
    # Hide any unused subplots
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Confusion Matrices on Held-Out Test Set (N=114)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=300)
    plt.savefig("static/images/confusion_matrices.png", dpi=300)
    plt.close()

def _plot_metric_comparisons(df_results: pd.DataFrame) -> None:
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(10, 5))
    bars = plt.barh(df_results["Model"], df_results["Accuracy"], color="#3b82f6", edgecolor="#1d4ed8")
    plt.xlim(0.85, 1.01)
    plt.xlabel("Accuracy", fontsize=11)
    plt.title("Model Accuracy Comparison (Held-out Test Set)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.7)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width*100:.2f}%",
                 va="center", ha="left", fontsize=9, fontweight="bold")
                 
    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png", dpi=300)
    plt.savefig("static/images/accuracy_comparison.png", dpi=300)
    plt.close()

    # 2. Precision, Recall (Sensitivity), F1-Score grouped bar chart
    metrics_to_plot = ["Precision", "Recall (Sensitivity)", "F1-Score"]
    df_melted = df_results.melt(
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )
    
    plt.figure(figsize=(12, 6))
    palette = {"Precision": "#0ea5e9", "Recall (Sensitivity)": "#ef4444", "F1-Score": "#10b981"}
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=palette)
    plt.ylim(0.80, 1.02)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.ylabel("Score", fontsize=11)
    plt.title("Comparison of Precision, Clinical Sensitivity (Recall), and F1-Score", fontsize=13, fontweight="bold")
    plt.legend(title="Metric", loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/precision_recall_f1_comparison.png", dpi=300)
    plt.savefig("static/images/precision_recall_f1_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_all()
