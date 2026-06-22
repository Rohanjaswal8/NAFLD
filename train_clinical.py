"""Train and compare clinical NAFLD models; save the best by accuracy."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import ROOT_DIR, load_config
from src.data.clinical import load_clinical_data, split_clinical_data
from src.evaluation import (
    compute_metrics,
    format_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from src.models.clinical_model import CLINICAL_MODEL_TYPES, build_clinical_model


def main():
    parser = argparse.ArgumentParser(description="Compare clinical NAFLD models")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    clinical_cfg = cfg["clinical"]

    df = load_clinical_data(cfg["clinical_csv"])
    X_train, X_test, y_train, y_test = split_clinical_data(
        df,
        test_size=clinical_cfg["test_size"],
        random_state=clinical_cfg["random_state"],
    )

    models_dir = ROOT_DIR / cfg["models_dir"]
    results_dir = ROOT_DIR / cfg["results_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison = []
    best_name = None
    best_model = None
    best_accuracy = -1.0
    best_metrics = None
    best_pred = None
    best_prob = None
    best_report = None

    print(f"Comparing {len(CLINICAL_MODEL_TYPES)} clinical models...\n")

    for model_type in CLINICAL_MODEL_TYPES:
        print(f"Training: {model_type}")
        model = build_clinical_model(model_type)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        row = {"model": model_type, **metrics}
        comparison.append(row)

        print(
            f"  accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics.get('roc_auc', 0):.4f}"
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_name = model_type
            best_model = model
            best_metrics = metrics
            best_pred = y_pred
            best_prob = y_prob
            best_report = format_classification_report(y_test, y_pred)

    comparison_df = pd.DataFrame(comparison).sort_values(
        "accuracy", ascending=False
    )
    comparison_df.to_csv(results_dir / "clinical_model_comparison.csv", index=False)

    with open(results_dir / "clinical_model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    with open(results_dir / "clinical_best_model.json", "w") as f:
        json.dump({"best_model": best_name, "accuracy": best_accuracy}, f, indent=2)

    joblib.dump(best_model, models_dir / "clinical_model.joblib")
    save_metrics(best_metrics, best_report, results_dir, "clinical")
    plot_confusion_matrix(
        y_test,
        best_pred,
        results_dir / "clinical_confusion_matrix.png",
        f"Best Clinical Model ({best_name})",
    )
    plot_roc_curve(
        y_test,
        best_prob,
        results_dir / "clinical_roc_curve.png",
        f"Best Clinical Model ROC ({best_name})",
    )

    print("\n" + "=" * 50)
    print("Clinical model comparison complete.")
    print(f"Best model: {best_name} (accuracy={best_accuracy:.4f})")
    print("\nRanking by accuracy:")
    for _, row in comparison_df.iterrows():
        print(f"  {row['model']}: {row['accuracy']:.4f}")
    print(f"\nBest model saved to: {models_dir / 'clinical_model.joblib'}")


if __name__ == "__main__":
    main()
