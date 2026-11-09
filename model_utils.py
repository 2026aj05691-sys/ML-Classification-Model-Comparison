from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "sleep_debt_category"
NUMERIC_FEATURES = [
    "age", "bedtime_phone_minutes", "screen_brightness_pct",
    "blue_light_filter_active", "caffeine_post_5pm_mg", "physical_activity_min",
    "sleep_latency_min", "total_sleep_hours", "deep_sleep_pct", "rem_sleep_pct",
    "morning_alarm_snoozes", "next_day_fatigue_score",
]
CATEGORICAL_FEATURES = [
    "gender", "occupation_type", "chronotype", "primary_bedtime_app",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluation_metrics(y_true, y_pred, probabilities, classes):
    """Return six metrics; AUC is undefined unless all trained classes are present."""
    unknown_labels = set(y_true) - set(classes)
    if unknown_labels:
        raise ValueError(f"Unknown target labels: {sorted(unknown_labels)}")

    auc = None
    if set(y_true) == set(classes):
        auc = roc_auc_score(
            y_true, probabilities, labels=classes, multi_class="ovr", average="macro"
        )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": auc,
        "Precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }
