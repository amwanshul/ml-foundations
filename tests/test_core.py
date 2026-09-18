import unittest
import numpy as np

from ml_foundations.linear_regression import LinearRegression
from ml_foundations.logistic_regression import LogisticRegression
from ml_foundations.metrics import accuracy, precision, recall, f1_score
from ml_foundations.preprocessing import StandardScaler


class TestCore(unittest.TestCase):
    def test_scaler(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaled = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(scaled.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(scaled.std(axis=0), [1, 1], atol=1e-10)

    def test_linear_regression(self):
        X = np.arange(1, 6, dtype=float).reshape(-1, 1)
        y = 2 * X[:, 0] + 1
        model = LinearRegression(learning_rate=0.02, epochs=2000).fit(X, y)
        np.testing.assert_allclose(model.predict([[6]])[0], 13, atol=0.2)

    def test_logistic_regression(self):
        X = np.array([[-2], [-1], [1], [2]], dtype=float)
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(learning_rate=0.2, epochs=1500).fit(X, y)
        self.assertEqual(model.predict([[-3], [3]]).tolist(), [0, 1])

    def test_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.75)
        self.assertAlmostEqual(precision(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(recall(y_true, y_pred), 1.0)
        self.assertAlmostEqual(f1_score(y_true, y_pred), 0.8)


if __name__ == "__main__":
    unittest.main()
