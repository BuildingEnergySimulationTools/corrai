import numpy as np
from corrai.metrics import nmbe
from corrai.metrics import cv_rmse


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
