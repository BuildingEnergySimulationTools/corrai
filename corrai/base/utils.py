import pandas as pd
import numpy as np
import datetime as dt
from collections.abc import Iterable


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


def find_gaps(data, cols=None, timestep=None, return_combination=True):
    """
    Find gaps in time series data. Find individual columns gap and combined gap
    for all columns.

    Parameters:
    -----------
        data (pandas.DataFrame or pandas.Series): The time series data to check
            for gaps.

        cols (list, optional): The columns to check for gaps. Defaults to None,
            in which case all columns are checked.

        timestep (str or pandas.Timedelta, optional): The time step of the
            data. Can be either a string representation of a time period
            (e.g., '1H' for hourly data), or a pandas.Timedelta object.
            Defaults to None, in which case the time step is automatically
            determined using inferred_freq or using mean of timesteps if frequency
             cannot be inferred.

        return_combination (Bool, default True): wether or not to return a dict key
        "combination" that aggregate gaps of each columns in the DataFrame.

    Raises:
    -------

        ValueError: If cols or timestep are invalid.

    Returns:
    --------
        dict: A dictionary containing the duration of the gaps for each
        specified column, as well as the overall combination of columns.
    """

    check_datetime_index(data)
    if isinstance(data, pd.Series):
        data = as_1_column_dataframe(data)
    cols = data.columns if cols is None else cols
    timestep = get_mean_timestep(data) if timestep is None else timestep

    # Aggregate in a single columns to know overall quality
    df = data.copy()
    df = ~df.isnull()
    df["combination"] = df.all(axis=1)

    # Index are added at the beginning and at the end to account for
    # missing values and each side of the dataset
    first_index = df.index[0] - (df.index[1] - df.index[0])
    last_index = df.index[-1] - (df.index[-2] - df.index[-1])

    df.loc[first_index] = np.ones(df.shape[1], dtype=bool)
    df.loc[last_index] = np.ones(df.shape[1], dtype=bool)
    df.sort_index(inplace=True)

    # Compute gaps duration
    res = {}
    cols = list(cols) + ["combination"] if return_combination else list(cols)
    for col in cols:
        time_der = df[col].loc[df[col]].index.to_series().diff()
        res[col] = time_der[time_der > timestep]

    return res


def get_biggest_group(data: pd.Series):
    """
    Returns the largest continuous group of non-NaN values from a pandas Series
    with a datetime index.

    Parameters:
    -----------
    data : pd.Series
        A pandas Series with a datetime index. The Series may contain NaN values
        representing gaps.

    Returns:
    --------
    pd.Series
        The largest continuous segment of the input Series that does not contain
        NaN values.
    """
    check_datetime_index(data)
    valid_mask = data.notna()
    groups = (valid_mask != valid_mask.shift()).cumsum()
    largest_group_id = data[valid_mask].groupby(groups).size().idxmax()
    return data[groups == largest_group_id]


def gaps_describe(df_in, cols=None, timestep=None):
    res_find_gaps = find_gaps(df_in, cols, timestep)

    return pd.DataFrame({k: val.describe() for k, val in res_find_gaps.items()})


def get_mean_timestep(df):
    freq = df.index.inferred_freq
    if freq:
        freq = pd.to_timedelta("1" + freq) if freq.isalpha() else pd.to_timedelta(freq)
    else:
        freq = df.index.to_frame().diff().mean()[0]

    return freq


def missing_values_dict(df):
    return {
        "Number_of_missing": df.count(),
        "Percent_of_missing": (1 - df.count() / df.shape[0]) * 100,
    }
