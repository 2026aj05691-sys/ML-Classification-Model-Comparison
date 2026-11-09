# Sleep Debt Classification

## a. Problem statement

Classify a user's sleep-debt status from demographic information, before-sleep
habits, during-sleep measurements, and next-day outcomes. The target is
`sleep_debt_category`, with four classes: Optimal Recovery, Mild Deficit,
Moderate Debt, and Severe Sleep Debt.

Compare five classification algorithms using the same features and held-out
test samples, reporting accuracy, AUC, precision, recall, F1, and MCC. A Streamlit
app lets users select a model, upload a CSV, and view predictions and evaluation
metrics. This is classification of observed patterns, not before-sleep forecasting
or a medical diagnosis.

## b. Dataset description

The dataset is [bedtime_screentime_sleep_debt.csv](bedtime_screentime_sleep_debt.csv).

| Property | Description |
|---|---|
| Samples | 8,500 user records |
| Total columns | 18 |
| Predictors | 16: 12 numeric and 4 categorical |
| Excluded identifier | `user_id` |
| Target | `sleep_debt_category` |
| Training set | 6,800 rows (80%) |
| Test set | 1,700 rows (20%), saved in [test_data.csv](test_data.csv) |
| Split method | Stratified by target, with `random_state=42` |

Predictors cover age, gender, occupation, chronotype, bedtime phone use and app,
screen brightness, blue-light filter use, evening caffeine, physical activity,
sleep latency, total sleep, deep/REM sleep percentages, alarm snoozes, and
next-day fatigue.

Numeric predictors are standardized and categorical predictors are one-hot
encoded. Preprocessing is fitted only on the training set and saved inside
each model pipeline. All five models use the same split and inputs.

## c. GitHub repository link

https://github.com/2026aj05691-sys/ML-Classification-Model-Comparison

## d. Models used and evaluation metrics

The five models are:

1. **Logistic Regression:** `max_iter=2000`, with default `C=1.0`.
2. **Decision Tree Classifier:** `random_state=42`, with default depth settings.
3. **K-Nearest Neighbors (KNN):** `n_neighbors=5`.
4. **Gaussian Naive Bayes:** default GaussianNB settings.
5. **Support Vector Machine (SVM):** RBF kernel, default `C=1.0`,
   `probability=True`, and `random_state=42`.

### Comparison table

The following results come from the saved pipelines evaluated on the same
1,700 held-out rows. Values are rounded to four decimal places; full-precision
results are available in [model/metrics.csv](model/metrics.csv).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9812 | 0.9993 | 0.9784 | 0.9757 | 0.9770 | 0.9704 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.8118 | 0.9317 | 0.8060 | 0.7603 | 0.7797 | 0.7020 |
| Naive Bayes (Gaussian) | 0.8465 | 0.9797 | 0.8084 | 0.8674 | 0.8302 | 0.7685 |
| Support Vector Machine | 0.9494 | 0.9969 | 0.9374 | 0.9362 | 0.9367 | 0.9206 |



### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong runner-up: 98.12% accuracy, 0.9770 macro F1, and 0.9704 MCC. Its 0.9993 AUC indicates excellent one-vs-rest ranking on this split. It outperformed KNN, Gaussian Naive Bayes, and SVM despite its linear decision boundaries. |
| Decision Tree | Best observed performance, with 1.0000 for every metric and no classification errors on the held-out split. Investigate whether included sleep measurements directly define the target before interpreting this as independent predictive ability. |
| KNN | Lowest scores among the five models: 81.18% accuracy, 0.7797 macro F1, and 0.7020 MCC. Macro recall of 0.7603 shows that the five-neighbor configuration is less effective at recovering classes equally than the other tested models. |
| Naive Bayes (Gaussian) | Achieved 84.65% accuracy and 0.8302 macro F1, outperforming KNN but trailing Logistic Regression, Decision Tree, and SVM. Macro recall (0.8674) exceeds macro precision (0.8084), indicating a different precision/recall balance. Its feature-independence assumption may be limiting when sleep-related inputs are correlated. |
| Support Vector Machine | Strong performance: 94.94% accuracy, 0.9367 macro F1, 0.9206 MCC, and 0.9969 AUC. The RBF model outperformed KNN and Gaussian Naive Bayes, but did not improve on Logistic Regression or Decision Tree with these settings. |
| **Overall winner for this dataset** | **Decision Tree**, based on all six metrics on this held-out split. **Logistic Regression** is the runner-up. The perfect tree result is not proof of real-world generalization; verify label construction and evaluate independent data before drawing a broader conclusion. |

These observations compare the tested configurations, not every possible
hyperparameter setting. The test set was not used for fitting or tuning.

## Repository layout

```text
ML Assignment/
|-- app.py
|-- train_models.py
|-- model_utils.py
|-- test_project.py
|-- requirements.txt
|-- README.md
|-- bedtime_screentime_sleep_debt.csv
|-- test_data.csv
|-- sleep_debt_model_comparison.ipynb
|-- sleep_debt_logistic_regression.ipynb
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- gaussian_naive_bayes.joblib
|   |-- svm.joblib
|   |-- metrics.csv
|   |-- metadata.json
```

Both existing notebooks are preserved. The comparison notebook provides a simple
training example; `train_models.py` reproduces it and saves deployable models.

## Run locally

Use Python 3.13, matching the saved models. From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats false
```

If `.venv` already exists, skip its creation. The saved models are ready to use;
training is not performed when the app starts.

`requirements.txt` contains only the app and model dependencies. To also run the
notebooks locally, install the optional notebook and plotting tools:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel matplotlib
```

These optional tools are not needed on Streamlit Community Cloud.

Choose a classifier in the sidebar. Use the bundled test data or upload a CSV.
The app shows the six evaluation metrics, a confusion matrix, and downloadable
predictions with probabilities. Without target labels, it provides predictions
only. Uploaded data is processed in memory and is not saved by the app.

## Retrain and regenerate artifacts

```powershell
.\.venv\Scripts\python.exe train_models.py
```

This overwrites the five model artifacts, `test_data.csv`, `model/metrics.csv`,
and `model/metadata.json`. It does not change either notebook.

- A stratified 80/20 split with `random_state=42` gives 6,800 training rows and
  1,700 held-out test rows, matching the notebooks.
- Every classifier uses the same rows and predictors.
- Numeric columns are standardized; categorical columns are one-hot encoded.
  Preprocessing is fitted only on training data.
- Each `.joblib` contains the complete fitted preprocessing-and-classifier
  pipeline. No separate preprocessing file is needed.
- Models are Logistic Regression, Decision Tree, K-Nearest Neighbors,
  Gaussian Naive Bayes, and RBF Support Vector Machine with probability estimates.
- Test rows are not used to fit the saved models. Do not retrain on
  `test_data.csv` or use its scores to repeatedly tune models.

`model/metadata.json` records dependency versions, the source-data hash,
feature names, classes, and split settings. The core numerical dependencies in
`requirements.txt` match the saved artifacts. If using different versions,
retrain and verify rather than assuming old joblib files remain compatible.
Only load trusted model files: joblib/pickle artifacts can execute code.

## CSV format

The bundled `test_data.csv` is the actual held-out split, not synthetic examples.
Uploaded CSVs must contain these predictor columns:

| Numeric | Categorical |
|---|---|
| age | gender |
| bedtime_phone_minutes | occupation_type |
| screen_brightness_pct | chronotype |
| blue_light_filter_active | primary_bedtime_app |
| caffeine_post_5pm_mg | |
| physical_activity_min | |
| sleep_latency_min | |
| total_sleep_hours | |
| deep_sleep_pct | |
| rem_sleep_pct | |
| morning_alarm_snoozes | |
| next_day_fatigue_score | |

`sleep_debt_category` is optional for predictions and required for evaluation.
`user_id` and other extra columns are ignored by the model. Numeric inputs must
be finite and predictors must not contain missing values. Unseen categorical
values are accepted by the encoder, though predictions may be less reliable.
For evaluation, labels must match the four category names listed above.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_project -v
```

Tests check the held-out split, all five saved pipelines and six metrics, model
selection in the app, predictions without labels, unavailable AUC, and input errors.

## Interpretation and limitations

This is classification of observed patterns, not a forecast made before sleep.
The label-generation formula and dataset provenance have not been independently
verified. If sleep duration or other inputs were used to construct the target,
high accuracy may reflect reproducing that rule. A perfect Decision Tree score
on this split does not establish real-world generalization. These are dataset
associations, not causal findings or a clinically validated diagnosis.

## GitHub and deployment

The current folder is the repository root; no additional outer folder is needed.
Before publishing, confirm you have permission to redistribute the source and
held-out sleep data. No repository has been created or uploaded automatically.
Do not commit `.venv`, credentials, or private data.

Commit the source, README, requirements, authorized CSVs, notebooks, and the five
model pipelines. For Streamlit Community Cloud, select `app.py` as the entry point
and Python 3.13. The full training CSV is only needed for retraining; inference
uses the saved pipelines and the uploaded or bundled test CSV.
