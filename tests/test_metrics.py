import numpy as np

from corrai.metrics import cv_rmse, nmbe, smape


class TestMetrics:
    def test_nmbe(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.5, 2, 2.2, 3])

        expected = -13.0

        np.testing.assert_allclose(
            nmbe(y_pred=y_pred, y_true=y_true), expected, rtol=10 - 7
        )

    def test_cv_rmse(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.5, 2, 2.2, 3])

        expected = 30.0

        np.testing.assert_allclose(
            cv_rmse(y_pred=y_pred, y_true=y_true), expected, rtol=10 - 7
        )

    def test_smape(self):
        y_true = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        y_pred = np.array([[1.5, 2, 2.2, 3], [2.5, 3.0, 3.2, 4.0]])

        expected = 20.75

        np.testing.assert_allclose(
            smape(y_pred=y_pred, y_true=y_true), expected, rtol=10 - 7
        )
