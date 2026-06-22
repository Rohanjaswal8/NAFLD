from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.clinical import build_preprocessor

CLINICAL_MODEL_TYPES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "svm",
    "knn",
    "decision_tree",
    "naive_bayes",
]


def _build_classifier(model_type: str):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
    if model_type == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        )
    if model_type == "naive_bayes":
        return GaussianNB()
    raise ValueError(f"Unknown model type: {model_type}")


def build_clinical_model(model_type: str = "xgboost") -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", _build_classifier(model_type)),
        ]
    )
