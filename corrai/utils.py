import pandas as pd
import numpy as np


def _reshape_1d(sample):
    if isinstance(sample, pd.DataFrame):
        return sample.squeeze()
    elif isinstance(sample, np.ndarray):
        return sample.flatten()


def as_1_column_dataframe(X):
    """
    Converts a 1D array-like object to a pandas DataFrame with a single column.

    Parameters
    ----------
        X (list or numpy.ndarray or pandas.DataFrame): Input array-like object
        containing the data.

    Returns
    -------
    pandas.DataFrame: A pandas DataFrame object with a single column containing the
    input data.

    Raises
    ------
    ValueError: If input data is not a list, numpy.ndarray, or pandas.DataFrame.
    ValueError: If input data has more than one column.
    """

    if not isinstance(X, (list, np.ndarray, pd.DataFrame)):
        raise ValueError(
            f"X must be one of {list, np.array, pd.DataFrame}, " f"got {type(X)}"
        )

    if isinstance(X, list):
        X = pd.DataFrame(np.array(X))

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if X.shape[1] > 1:
        raise ValueError(
            f"X has {X.shape[1]} columns, cannot return 1 columns DataFrame "
        )

    return X


def check_datetime_index(X):
    """Check if X is an instance od DatFrame or Series and has a DatetimeIndex"""
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "A DataFrame or a Series was expected, got an instance of " f"{type(X)}"
        )
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X do not have a DateTimeIndex")
