import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def find_optimal_threshold(
    y_true, y_prob, metric: str = "gmean", min_thresh: float = 0.25, max_thresh: float = 0.55
) -> tuple[float, float]:
    """Find threshold that balances both classes (G-mean of recalls)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(min_thresh, max_thresh, 61):
        y_pred = (y_prob >= threshold).astype(int)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        if metric == "macro_f1":
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = float(np.sqrt(recalls[0] * recalls[1]))  # G-mean

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def save_metrics(metrics: dict, report: str, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / f"{prefix}_report.txt", "w") as f:
        f.write(report)


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-NAFLD", "NAFLD"],
        yticklabels=["Non-NAFLD", "NAFLD"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, output_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_classification_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, target_names=["Non-NAFLD", "NAFLD"], zero_division=0
    )
