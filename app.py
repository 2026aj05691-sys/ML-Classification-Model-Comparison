import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from model_utils import (
    FEATURES, MODEL_FILES, NUMERIC_FEATURES, ROOT, TARGET, evaluation_metrics,
)

st.set_page_config(page_title="Sleep Debt Classification", layout="wide")
st.title("Sleep Debt Classification")
st.caption(
    "Compare observed pre-, during-, and post-sleep patterns across four sleep-debt "
    "categories. Educational use only; not a medical diagnosis."
)

model_name = st.sidebar.selectbox("Classifier", list(MODEL_FILES))
data_source = st.sidebar.radio("Data source", ["Bundled test data", "Upload CSV"])
test_path = ROOT / "test_data.csv"
if not test_path.is_file():
    st.error("test_data.csv is missing. Run: python train_models.py")
    st.stop()
st.sidebar.download_button(
    "Download test CSV", test_path.read_bytes(), "test_data.csv", "text/csv"
)

try:
    model = joblib.load(ROOT / "model" / MODEL_FILES[model_name])
except FileNotFoundError:
    st.error("Saved model is missing. Run: python train_models.py")
    st.stop()

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV with the 16 predictor columns", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV, or choose Bundled test data in the sidebar.")
        st.stop()
    csv_source = uploaded_file
else:
    csv_source = test_path

try:
    data = pd.read_csv(csv_source)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

missing_columns = sorted(set(FEATURES) - set(data.columns))
if missing_columns:
    st.error(f"Missing predictor columns: {', '.join(missing_columns)}")
    st.stop()
if data.empty:
    st.error("The CSV has no data rows.")
    st.stop()
if data[FEATURES].isna().any().any():
    st.error("Predictor columns contain missing values. Fill them before predicting.")
    st.stop()

X = data[FEATURES].copy()
try:
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        st.error("Numeric predictors must contain finite values.")
        st.stop()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
except (ValueError, TypeError) as error:
    st.error(f"Could not generate predictions: {error}")
    st.stop()

st.write(f"**Model:** {model_name} | **Rows:** {len(data):,}")
st.caption(
    "Precision, recall, and F1 use macro averaging over the four trained classes. "
    "AUC uses macro one-vs-rest probabilities."
)
if TARGET in data.columns:
    y = data[TARGET]
    if y.isna().any():
        st.error("Target labels contain missing values. Fill them or remove the target column.")
        st.stop()
    unknown_labels = set(y) - set(model.classes_)
    if unknown_labels:
        st.error(f"Unknown target labels: {sorted(unknown_labels)}")
        st.stop()
    metrics = evaluation_metrics(y, predictions, probabilities, model.classes_)
    if metrics["AUC Score"] is None:
        st.warning(
            "AUC is unavailable: the evaluation CSV must contain all four target "
            "classes. Other metrics are still shown."
        )
    for column, (name, value) in zip(st.columns(6), metrics.items()):
        column.metric(name, "N/A" if value is None else f"{value:.4f}")
    st.subheader("Confusion matrix")
    st.caption("Rows: actual categories. Columns: predicted categories.")
    st.dataframe(pd.DataFrame(
        confusion_matrix(y, predictions, labels=model.classes_),
        index=pd.Index(model.classes_, name="Actual"),
        columns=pd.Index(model.classes_, name="Predicted"),
    ))
else:
    st.info(f"Predictions only: add '{TARGET}' to the CSV to calculate metrics.")

output = data.copy()
output["predicted_sleep_debt_category"] = predictions
for i, label in enumerate(model.classes_):
    output[f"probability_{label}"] = probabilities[:, i]

st.subheader("Predictions")
st.dataframe(output.head(50), hide_index=True)
st.download_button(
    "Download all predictions",
    output.to_csv(index=False).encode("utf-8"),
    "sleep_debt_predictions.csv",
    "text/csv",
)
