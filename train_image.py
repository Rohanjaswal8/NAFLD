"""Train ultrasound NAFLD detection model."""

import argparse
import ssl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import ROOT_DIR, load_config
from src.data.imaging import create_dataloaders
from src.evaluation import (
    compute_metrics,
    format_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from src.models.image_model import build_image_model


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return all_labels, all_preds, all_probs, avg_loss


def load_model(backbone: str, pretrained: bool, device):
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        model = build_image_model(backbone, pretrained=pretrained).to(device)
        print(f"Loaded {backbone} (pretrained={pretrained})")
        return model
    except Exception:
        print(f"Pretrained load failed, using {backbone} from scratch.")
        return build_image_model(backbone, pretrained=False).to(device)


def main():
    parser = argparse.ArgumentParser(description="Train ultrasound NAFLD model")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_cfg = cfg["image"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_dataloaders(
        cfg["image_dirs"],
        img_size=img_cfg["img_size"],
        batch_size=img_cfg["batch_size"],
        val_split=img_cfg["val_split"],
        random_state=img_cfg["random_state"],
    )

    model = load_model(img_cfg["backbone"], img_cfg.get("pretrained", True), device)
    optimizer = Adam(model.parameters(), lr=img_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_gmean = 0.0
    models_dir = ROOT_DIR / cfg["models_dir"]
    results_dir = ROOT_DIR / cfg["results_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "image_model.pt"

    for epoch in range(img_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{img_cfg['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        y_true, y_pred, y_prob, val_loss = evaluate(model, val_loader, device, criterion)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        recalls = np.bincount(y_true, minlength=2)
        from sklearn.metrics import recall_score
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        gmean = float(np.sqrt(per_class_recall[0] * per_class_recall[1]))

        print(
            f"Epoch {epoch + 1}: train_loss={running_loss / len(train_loader.dataset):.4f}, "
            f"val_loss={val_loss:.4f}, gmean={gmean:.4f}, "
            f"non_nafld_recall={per_class_recall[0]:.3f}, nafld_recall={per_class_recall[1]:.3f}"
        )

        if gmean >= best_gmean:
            best_gmean = gmean
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": img_cfg["backbone"],
                    "img_size": img_cfg["img_size"],
                    "architecture": img_cfg["backbone"],
                },
                model_path,
            )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, y_prob, _ = evaluate(model, val_loader, device, criterion)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    report = format_classification_report(y_true, y_pred)
    save_metrics(metrics, report, results_dir, "image")
    plot_confusion_matrix(
        y_true, y_pred, results_dir / "image_confusion_matrix.png",
        f"Image Model ({img_cfg['backbone']})",
    )
    plot_roc_curve(
        y_true, y_prob, results_dir / "image_roc_curve.png",
        f"Image Model ROC ({img_cfg['backbone']})",
    )

    print(f"\nImage model ({img_cfg['backbone']}) training complete.")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
