import pandas as pd
import numpy as np
import datetime as dt
from collections.abc import Iterable


def _reshape_1d(sample):
    if isinstance(sample, pd.DataFrame):
        return sample.squeeze()
    elif isinstance(sample, np.ndarray):
        return sample.flatten()
    return None


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

    if not isinstance(X, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError(
            f"X must be one of {list, np.array, pd.DataFrame}, " f"got {type(X)}"
        )

    if isinstance(X, list):
        X = pd.DataFrame(np.array(X))

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if isinstance(X, pd.Series):
        X = X.to_frame()

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


def apply_transformation(x, function):
    """
    Utility function to apply elementwise transformation to list, ndarray,
    Series or DataFrame. If an object other than previously mentioned is passed,
    returns function(x)
    :param x: input to be transformed
    :param function: callable
    :return: transformed list, ndarray, Series or DataFrame or object, depending on
    the input
    """
    if isinstance(x, list):
        return [function(i) for i in x]
    elif isinstance(x, np.ndarray):
        return np.vectorize(function)(x)
    elif isinstance(x, pd.Series):
        return x.apply(function)
    elif isinstance(x, pd.DataFrame):
        # applymap will be deprecated in future version
        return x.map(function)

    return function(x)


def float_to_hour(hours_float):
    """
    Convert a float value representing hours to a string representation with the
    format "%H:%M".
    :param Float hours_float: Floating-point value representing hours.
    :return String representation of the hours in the format "%H:%M".
    """

    def f2s(x):
        h = int(x)
        minutes = int((x - h) * 60)
        time_str = dt.time(h, minutes).strftime("%H:%M")
        return time_str

    return apply_transformation(hours_float, f2s)


def hour_to_float(hours_string):
    """
    Convert hour string representation with the format "%H:%M"to a float value.
    :param hours_string: String, list, ndarray, Series, DataFrame value
    representing hours.
    :return float: Floating-point value representing hours.
    """

    def s2f(x):
        time_obj = dt.datetime.strptime(x, "%H:%M")
        return time_obj.hour + time_obj.minute / 60

    return apply_transformation(hours_string, s2f)


def get_reversed_dict(dictionary, values=None):
    """
    Reverses the key-value pairs in a dictionary and returns a new dictionary.

    :param dictionary: The original dictionary containing key-value pairs.

    :param values: (iterable or any) Optional. The values to filter the reversed
        dictionary by. If not provided, all values from the original dictionary will
        be used. Can be an iterable or a single value.

    :return: A new dictionary with reversed key-value pairs from the original
        dictionary. The new dictionary only includes key-value pairs where the value
        matches the specified values. If values are not provided, all key-value pairs
        from the original dictionary are included.
    """
    if values is None:
        values = dictionary.values()
    elif not isinstance(values, Iterable):
        values = [values]

    return {val: key for key, val in dictionary.items() if val in values}
