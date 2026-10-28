import os
import unittest

import pandas as pd
from sklearn.datasets import load_breast_cancer

import predict
import train

HERE = os.path.dirname(os.path.abspath(__file__))


class DataTests(unittest.TestCase):
    def test_data_shape_and_no_missing(self):
        X, y, names = train.load_data()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(int(X.isnull().sum().sum()), 0)
        self.assertEqual(names, ["malignant", "benign"])

    def test_every_candidate_has_a_param_grid(self):
        self.assertEqual(set(train.candidate_models()), set(train.PARAM_GRIDS))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(HERE, "model.joblib")):
            train.main()
        cls.bundle = predict.load_model()

    def test_predict_returns_valid_labels(self):
        data = load_breast_cancer(as_frame=True)
        out = predict.predict_frame(self.bundle, data.data.iloc[:10])
        self.assertEqual(len(out), 10)
        self.assertTrue(set(out["prediction"]) <= {"malignant", "benign"})

    def test_accuracy_reasonable_on_full_data(self):
        # Sanity check only (includes training rows, so NOT a performance claim).
        data = load_breast_cancer(as_frame=True)
        out = predict.predict_frame(self.bundle, data.data)
        truth = [data.target_names[i] for i in data.target]
        acc = (out["prediction"].values == pd.Series(truth).values).mean()
        self.assertGreater(acc, 0.9)

    def test_missing_columns_raises(self):
        with self.assertRaises(ValueError):
            predict.predict_frame(self.bundle, pd.DataFrame({"a": [1]}))


if __name__ == "__main__":
    unittest.main()
