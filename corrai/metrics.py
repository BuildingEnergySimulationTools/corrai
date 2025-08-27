import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length


def nmbe(y_pred, y_true):
    """Normalized Mean Biased Error

    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :return:
    Normalized Mean biased error as float
    """
    check_consistent_length(y_pred, y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.DataFrame) else y_pred
    y_true = y_true.to_numpy() if isinstance(y_true, pd.DataFrame) else y_true
    # y_type, y_true, y_pred, multioutput = _check_reg_targets(
    #     y_true, y_pred, "uniform_average"
    # )

    return np.sum(y_pred - y_true) / np.sum(y_true) * 100


def cv_rmse(y_pred, y_true):
    """Coefficient of variation of root mean squared error

    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :return:
    Coefficient of variation of root mean squared error as float
    """
    check_consistent_length(y_pred, y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.DataFrame) else y_pred
    y_true = y_true.to_numpy() if isinstance(y_true, pd.DataFrame) else y_true
    # y_type, y_true, y_pred, multioutput = _check_reg_targets(
    #     y_true, y_pred, "uniform_average"
    # )
    return (
        (1 / np.mean(y_true))
        * np.sqrt(np.sum((y_true - y_pred) ** 2) / (y_true.shape[0] - 1))
        * 100
    )
