import numpy as np
import keras
from sklearn.utils import check_consistent_length
from sklearn.metrics._regression import _check_reg_targets


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
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, "uniform_average"
    )

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
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, "uniform_average"
    )
    return (
        (1 / np.mean(y_true))
        * np.sqrt(np.sum((y_true - y_pred) ** 2) / (y_true.shape[0] - 1))
        * 100
    )


def last_time_step_rmse(y_true, y_pred):
    """
    For sequence to sequence time forcasting models,
    returns the error on the last sequence.

    :param y_true: nd.array, with dimension []
    :param y_pred:
    :return:
    """
    return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])
