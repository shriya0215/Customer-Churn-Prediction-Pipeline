"""
Model Evaluation and Business Impact Analysis Module.
"""

from typing import Dict, Any, List, Tuple
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Computes full set of classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def compute_business_impact(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    clv_saved_per_tp: float = 250.0,
    cost_per_fp: float = 50.0,
    loss_per_fn: float = 500.0,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Computes business profit/loss across thresholds.
    
    Business assumptions:
    - TP: At-risk customer retained (CLV benefit minus incentive cost) = +$250
    - FP: Unnecessary incentive sent to loyal customer = -$50
    - FN: Churned customer lost entirely = -$500
    - TN: Retained with zero intervention = $0
    
    Returns:
        best_threshold: The threshold maximizing total business value
        max_value: The maximum net business value
        curve_data: List of threshold, net value, precision, recall
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    curve_data = []
    best_threshold = 0.5
    max_value = -float("inf")

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        # Net value compared to no-intervention baseline
        net_value = (tp * clv_saved_per_tp) - (fp * cost_per_fp) - (fn * loss_per_fn)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)

        curve_data.append({
            "threshold": round(float(thresh), 2),
            "net_value": float(net_value),
            "precision": float(prec),
            "recall": float(rec),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

        if net_value > max_value:
            max_value = net_value
            best_threshold = float(thresh)

    return best_threshold, max_value, curve_data


def plot_evaluation_curves(models_eval: Dict[str, Dict[str, Any]], output_dir: str = "artifacts/plots"):
    """
    Plots comparative ROC Curves and Precision-Recall Curves.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, data in models_eval.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc = data["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", lw=1.5)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name, data in models_eval.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        pr_auc = data["metrics"]["pr_auc"]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)

    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=200)
    plt.close()


def plot_feature_importance(pipeline, feature_names: List[str], output_dir: str = "artifacts/plots", top_n: int = 15):
    """
    Extracts and plots top feature importances or coefficients.
    """
    os.makedirs(output_dir, exist_ok=True)
    classifier = pipeline.named_steps["classifier"]

    importances = None
    title = "Feature Importances"
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"{classifier.__class__.__name__} Feature Importances"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        title = f"{classifier.__class__.__name__} Absolute Coefficients"

    if importances is not None and len(importances) == len(feature_names):
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices][::-1]
        top_scores = [importances[i] for i in indices][::-1]

        plt.figure(figsize=(10, 7))
        bars = plt.barh(range(top_n), top_scores, color="#1f77b4", align="center")
        plt.yticks(range(top_n), top_features)
        plt.xlabel("Importance / Absolute Weight", fontsize=12)
        plt.title(f"Top {top_n} Predictive Features ({title})", fontsize=14, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=200)
        plt.close()
