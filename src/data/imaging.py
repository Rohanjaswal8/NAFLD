from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


def collect_image_paths(image_dirs: dict[str, str]) -> pd.DataFrame:
    records = []
    for label_name, label_value in [("nafld", 1), ("non_nafld", 0)]:
        folder = Path(image_dirs[label_name])
        if not folder.exists():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        for path in sorted(folder.glob("*.jpg")):
            records.append({"path": str(path), "label": label_value})
    return pd.DataFrame(records)


def compute_class_weights(labels: pd.Series, max_ratio: float = 3.0) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    total = len(labels)
    raw = torch.tensor(
        [total / (len(counts) * counts[i]) for i in range(len(counts))],
        dtype=torch.float32,
    )
    ratio = raw[0] / raw[1]
    if ratio > max_ratio:
        raw[0] = raw[1] * max_ratio
    elif ratio < 1 / max_ratio:
        raw[1] = raw[0] * max_ratio
    return raw


class UltrasoundDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def get_transforms(img_size: int, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(
    image_dirs: dict[str, str],
    img_size: int = 224,
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    df = collect_image_paths(image_dirs)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=random_state,
        stratify=df["label"],
    )

    class_weights = compute_class_weights(train_df["label"])

    train_labels = train_df["label"].values
    sample_weights = 1.0 / np.bincount(train_labels)[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )

    train_dataset = UltrasoundDataset(
        train_df, transform=get_transforms(img_size, train=True)
    )
    val_dataset = UltrasoundDataset(
        val_df, transform=get_transforms(img_size, train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_weights
