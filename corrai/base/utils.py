import pandas as pd
import numpy as np
import datetime as dt
from collections.abc import Iterable
from typing import Callable


def _reshape_1d(sample):
    """
    Reshape a 1D-like object to a 1D representation.

    Parameters
    ----------
    sample : pandas.DataFrame, numpy.ndarray or other
        The input object to reshape.

    Returns
    -------
    pandas.Series or numpy.ndarray or None
        - If DataFrame: squeezed to Series.
        - If ndarray: flattened to 1D.
        - Otherwise: returns None.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> _reshape_1d(np.array([[1, 2], [3, 4]]))
    array([1, 2, 3, 4])
    >>> _reshape_1d(pd.DataFrame({"a": [1, 2, 3]}))
    0    1
    1    2
    2    3
    Name: a, dtype: int64
    """
    if isinstance(sample, pd.DataFrame):
        return sample.squeeze()
    elif isinstance(sample, np.ndarray):
        return sample.flatten()
    return None


def as_1_column_dataframe(X):
    """
    Convert an array-like object to a pandas DataFrame with a single column.

    Parameters
    ----------
    X : list, numpy.ndarray, pandas.DataFrame or pandas.Series
        Input array-like object containing the data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with a single column containing the input data.

    Raises
    ------
    ValueError
        If input data type is invalid or if DataFrame has more than one column.

    Examples
    --------
    >>> as_1_column_dataframe([1, 2, 3])
       0
    0  1
    1  2
    2  3
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
    """
    Check that an object has a DatetimeIndex.

    Parameters
    ----------
    X : pandas.Series or pandas.DataFrame
        Input object.

    Raises
    ------
    ValueError
        If the input is not a Series or DataFrame, or if its index
        is not a DatetimeIndex.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2], index=pd.date_range("2020-01-01", periods=2))
    >>> check_datetime_index(s)  # passes without error

    >>> s2 = pd.Series([1, 2])
    >>> check_datetime_index(s2)
    Traceback (most recent call last):
        ...
    ValueError: X do not have a DateTimeIndex
    """
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "A DataFrame or a Series was expected, got an instance of " f"{type(X)}"
        )
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X do not have a DateTimeIndex")


def apply_transformation(x, function):
    """
    Apply a transformation function elementwise to array-like or scalar input.

    Parameters
    ----------
    x : list, numpy.ndarray, pandas.Series, pandas.DataFrame or any
        Input to be transformed.
    function : callable
        Transformation function.

    Returns
    -------
    list, numpy.ndarray, pandas.Series, pandas.DataFrame or any
        Object of same type as input with the transformation applied.

    Examples
    --------
    >>> apply_transformation([1, 2, 3], lambda x: x**2)
    [1, 4, 9]
    >>> import numpy as np
    >>> apply_transformation(np.array([1, 2, 3]), lambda x: x + 1)
    array([2, 3, 4])
    >>> import pandas as pd
    >>> apply_transformation(pd.Series([1, 2]), lambda x: x * 10)
    0    10
    1    20
    dtype: int64
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
    Convert float hours to string time format "%H:%M".

    Parameters
    ----------
    hours_float : float, list, numpy.ndarray, pandas.Series or pandas.DataFrame
        Floating-point hour values.

    Returns
    -------
    str, list, numpy.ndarray, pandas.Series or pandas.DataFrame
        Same type as input, with values formatted as "%H:%M".

    Examples
    --------
    >>> float_to_hour(12.5)
    '12:30'
    >>> float_to_hour([0.0, 1.25])
    ['00:00', '01:15']
    """

    def f2s(x):
        h = int(x)
        minutes = int((x - h) * 60)
        time_str = dt.time(h, minutes).strftime("%H:%M")
        return time_str

    return apply_transformation(hours_float, f2s)


def hour_to_float(hours_string):
    """
    Convert hour strings "%H:%M" to float hours.

    Parameters
    ----------
    hours_string : str, list, numpy.ndarray, pandas.Series or pandas.DataFrame
        String representation of hours.

    Returns
    -------
    float, list, numpy.ndarray, pandas.Series or pandas.DataFrame
        Same type as input, converted to float values of hours.

    Examples
    --------
    >>> hour_to_float("12:30")
    12.5
    >>> hour_to_float(["00:00", "01:15"])
    [0.0, 1.25]
    """

    def s2f(x):
        time_obj = dt.datetime.strptime(x, "%H:%M")
        return time_obj.hour + time_obj.minute / 60

    return apply_transformation(hours_string, s2f)


def get_reversed_dict(dictionary, values=None):
    """
    Reverse key-value pairs of a dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary to reverse.
    values : iterable or scalar, optional
        Subset of values to include in reversed dictionary. If None,
        all values are included.

    Returns
    -------
    dict
        Dictionary with values as keys and keys as values, filtered
        by ``values`` if specified.

    Examples
    --------
    >>> d = {"a": 1, "b": 2, "c": 1}
    >>> get_reversed_dict(d)
    {1: 'c', 2: 'b'}
    >>> get_reversed_dict(d, values=1)
    {1: 'c'}
    """

    if values is None:
        values = dictionary.values()
    elif not isinstance(values, Iterable):
        values = [values]

    return {val: key for key, val in dictionary.items() if val in values}


def check_indicators_configs(
    is_dynamic: bool,
    indicators_configs: list[str]
    | list[tuple[str, str | Callable] | tuple[str, str | Callable, pd.Series]]
    | None,
):
    if is_dynamic:
        if indicators_configs is None:
            raise ValueError(
                "Model is dynamic. At least one indicators and its aggregation "
                "method must be provided"
            )
        if isinstance(indicators_configs[0], str):
            raise ValueError(
                "Invalid 'indicators_configs'. Model is dynamic"
                "At least 'method' is required"
            )
    else:
        if indicators_configs is not None and isinstance(indicators_configs[0], tuple):
            raise ValueError(
                "Invalid 'indicators_configs'. Model is static. "
                "'indicators_configs' must be a list of string"
            )
