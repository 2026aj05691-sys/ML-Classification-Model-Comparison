import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_utils import (
    CATEGORICAL_FEATURES, FEATURES, MODEL_FILES, NUMERIC_FEATURES, ROOT, TARGET,
    evaluation_metrics,
)


def main():
    data_path = ROOT / "bedtime_screentime_sleep_debt.csv"
    df = pd.read_csv(data_path)
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    X_train, y_train = train_data[FEATURES], train_data[TARGET]
    X_test, y_test = test_data[FEATURES], test_data[TARGET]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    }

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    results = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        artifact_path = model_dir / MODEL_FILES[name]
        joblib.dump(pipeline, artifact_path, compress=3)
        saved_model = joblib.load(artifact_path)
        metrics = evaluation_metrics(
            y_test, saved_model.predict(X_test), saved_model.predict_proba(X_test),
            saved_model.classes_,
        )
        results.append({"Model": name, **metrics})
        print(f"Saved {artifact_path.name}")

    test_data.to_csv(ROOT / "test_data.csv", index=False)
    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(model_dir / "metrics.csv")
    metadata = {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]
        },
        "source_file": data_path.name,
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "random_state": 42,
        "test_size": 0.2,
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "target": TARGET,
        "classes": sorted(df[TARGET].unique().tolist()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(results_df.round(4).to_string())


if __name__ == "__main__":
    main()
