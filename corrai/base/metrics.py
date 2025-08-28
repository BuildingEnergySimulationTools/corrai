import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length


def nmbe(y_pred, y_true):
    """
    Normalized Mean Bias Error (NMBE).

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    Returns
    -------
    float
        Normalized mean bias error, expressed as a percentage.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([100, 200, 300])
    >>> y_pred = np.array([110, 190, 310])
    >>> nmbe(y_pred, y_true)
    1.6666666666666667

    >>> import pandas as pd
    >>> y_true = pd.Series([10, 20, 30])
    >>> y_pred = pd.Series([12, 18, 29])
    >>> nmbe(y_pred, y_true)
    -1.6666666666666667
    """
    check_consistent_length(y_pred, y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.DataFrame) else y_pred
    y_true = y_true.to_numpy() if isinstance(y_true, pd.DataFrame) else y_true
    return np.sum(y_pred - y_true) / np.sum(y_true) * 100


def cv_rmse(y_pred, y_true):
    """
    Coefficient of Variation of the Root Mean Squared Error (CV(RMSE)).

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    Returns
    -------
    float
        CV(RMSE), expressed as a percentage.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([100, 200, 300])
    >>> y_pred = np.array([110, 190, 310])
    >>> cv_rmse(y_pred, y_true)
     6.123724356957945


    >>> import pandas as pd
    >>> y_true = pd.Series([10, 20, 30])
    >>> y_pred = pd.Series([12, 18, 29])
    >>> cv_rmse(y_pred, y_true)
    10.606601717798213
    """
    check_consistent_length(y_pred, y_true)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.DataFrame) else y_pred
    y_true = y_true.to_numpy() if isinstance(y_true, pd.DataFrame) else y_true
    return (
        (1 / np.mean(y_true))
        * np.sqrt(np.sum((y_true - y_pred) ** 2) / (y_true.shape[0] - 1))
        * 100
    )
