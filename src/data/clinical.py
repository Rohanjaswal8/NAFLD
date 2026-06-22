from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "Weight_kg",
    "Height_cm",
    "Glucose",
    "Cholesterol",
    "Triglycerides",
    "ALT",
    "AST",
    "Bilirubin",
    "Albumin",
    "Systolic_BP",
    "Diastolic_BP",
    "Diabetes_History",
]
TARGET_COLUMN = "NAFLD"


def load_clinical_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def split_clinical_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )
