import unittest

import numpy as np

from ml_foundations.metrics import accuracy, f1_score, precision, recall


class TestMetricEdgeCases(unittest.TestCase):
    def test_all_negative_predictions_return_zero_rates(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        self.assertEqual(precision(y_true, y_pred), 0.0)
        self.assertEqual(recall(y_true, y_pred), 0.0)
        self.assertEqual(f1_score(y_true, y_pred), 0.0)
        self.assertEqual(accuracy(y_true, y_pred), 1.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            accuracy(np.array([1, 0]), np.array([1]))


if __name__ == "__main__":
    unittest.main()
