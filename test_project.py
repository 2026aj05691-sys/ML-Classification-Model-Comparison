import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from model_utils import FEATURES, MODEL_FILES, ROOT, TARGET, evaluation_metrics


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.read_csv(ROOT / "test_data.csv")
        cls.saved_metrics = pd.read_csv(ROOT / "model" / "metrics.csv", index_col="Model")

    def run_app(self, data=None):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        if data is None:
            app.run()
        else:
            with patch("pandas.read_csv", return_value=data):
                app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_test_data_is_the_original_held_out_split(self):
        source = pd.read_csv(ROOT / "bedtime_screentime_sleep_debt.csv")
        train, test = train_test_split(
            source, test_size=0.2, random_state=42, stratify=source[TARGET]
        )
        pd.testing.assert_frame_equal(test.reset_index(drop=True), self.test_data)
        self.assertTrue(set(train["user_id"]).isdisjoint(test["user_id"]))
        self.assertEqual(len(FEATURES), 16)

    def test_saved_models_reproduce_all_six_metrics(self):
        X, y = self.test_data[FEATURES], self.test_data[TARGET]
        for name, filename in MODEL_FILES.items():
            with self.subTest(model=name):
                model = joblib.load(ROOT / "model" / filename)
                probabilities = model.predict_proba(X)
                self.assertEqual(probabilities.shape, (1700, 4))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                metrics = evaluation_metrics(y, model.predict(X), probabilities, model.classes_)
                self.assertEqual(len(metrics), 6)
                np.testing.assert_allclose(
                    list(metrics.values()), self.saved_metrics.loc[name].to_numpy(),
                    rtol=1e-10, atol=1e-12,
                )
                self.assertEqual(
                    model.named_steps["preprocessor"].named_transformers_["numeric"].n_samples_seen_,
                    6800,
                )

    def test_app_selects_every_model(self):
        app = self.run_app()
        for name in MODEL_FILES:
            with self.subTest(model=name):
                app.sidebar.selectbox[0].select(name).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(len(app.metric), 6)
                self.assertEqual(
                    [metric.value for metric in app.metric],
                    [f"{value:.4f}" for value in self.saved_metrics.loc[name]],
                )

    def test_unlabeled_csv_produces_predictions_only(self):
        app = self.run_app(self.test_data.head(10).drop(columns=[TARGET]))
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("Predictions only", app.info[0].value)
        self.assertEqual(len(app.dataframe), 1)

    def test_incomplete_class_coverage_explains_unavailable_auc(self):
        data = self.test_data.loc[self.test_data[TARGET] == "Moderate Debt"].head(10)
        app = self.run_app(data)
        self.assertEqual(len(app.error), 0)
        self.assertIn("AUC is unavailable", app.warning[0].value)
        self.assertEqual(app.metric[1].value, "N/A")

    def test_invalid_inputs_show_errors(self):
        missing_values = self.test_data.head(10).copy()
        missing_values.loc[missing_values.index[0], "age"] = np.nan
        nonnumeric = self.test_data.head(10).copy()
        nonnumeric["age"] = "not a number"
        infinite = self.test_data.head(10).copy()
        infinite["age"] = np.inf
        unknown_label = self.test_data.head(10).copy()
        unknown_label[TARGET] = "Unknown"
        cases = [
            (self.test_data.drop(columns=["age"]), "Missing predictor columns"),
            (self.test_data.iloc[:0], "no data rows"),
            (missing_values, "missing values"),
            (nonnumeric, "Could not generate predictions"),
            (infinite, "finite values"),
            (unknown_label, "Unknown target labels"),
        ]
        for data, message in cases:
            with self.subTest(message=message):
                app = self.run_app(data)
                self.assertEqual(len(app.error), 1)
                self.assertIn(message, app.error[0].value)

    def test_upload_mode_waits_for_a_file(self):
        app = self.run_app()
        app.sidebar.radio[0].set_value("Upload CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("Upload a CSV", app.info[0].value)


if __name__ == "__main__":
    unittest.main()
