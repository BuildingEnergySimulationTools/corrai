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
            f"A DataFrame or a Series was expected, got an instance of {type(X)}"
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


def get_data_gaps(
    data: pd.Series | pd.DataFrame,
    cols: str | list[str] = None,
    gap_threshold: str | dt.timedelta = None,
    return_combination=True,
):
    """
    Identifies gaps (consecutive NaN values) in the provided time series (pandas Series
    and DataFrames) data and returns them as groups of consecutive missing data
    points (NaNs).
    A gap threshold can be specified to filter gaps by their sizes.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The input time series data with a DateTime index. NaN values are
        considered gaps.
    cols : str or list[str], optional
        The columns in the DataFrame for which to detect gaps. If None (default), all
        columns are considered.
    gap_threshold : str or timedelta, optional
        The minimum duration of a gap for it to be considered valid.
        Can be passed as a string (e.g., '1d' for one day) or a `timedelta`.
        If None, no threshold is applied, NaN values are considered gaps.
    return_combination : bool, optional
        If True (default), a combination column is created that checks for NaNs
        across all columns in the DataFrame. Gaps in this combination column represent
        rows where NaNs are present in any of the columns.

    Returns
    -------
    dict[str, list[pd.DatetimeIndex]]
        A dictionary where the keys are the column names (or "combination" if
        `return_combination` is True) and the values are lists of `DatetimeIndex`
        objects.
        Each `DatetimeIndex` represents a group of one or several consecutive
        timestamps where the values in the corresponding column were NaN and
        exceeded the gap threshold.

    """

    check_datetime_index(data)
    if isinstance(data, pd.Series):
        data = as_1_column_dataframe(data)

    if isinstance(cols, str):
        cols = [cols]
    elif cols is None:
        cols = list(data.columns)

    if isinstance(gap_threshold, str):
        gap_threshold = pd.to_timedelta(gap_threshold)
    elif gap_threshold is None:
        gap_threshold = pd.to_timedelta(0)

    df = data.isnull()
    if return_combination:
        df["combination"] = df.any(axis=1)
        cols += ["combination"]

    def is_valid_gap(group):
        new_gap = pd.DatetimeIndex(group)
        return (new_gap.max() - new_gap.min()) >= gap_threshold

    def finalize_gap(current_group):
        new_gap_index = pd.DatetimeIndex(current_group)
        new_gap_index.freq = new_gap_index.inferred_freq
        return new_gap_index

    gap_dict = {}
    for col in cols:
        nan_groups = []
        current_group = []

        for timestamp in df.index:
            if df.loc[timestamp, col]:
                current_group.append(timestamp)
            else:
                if current_group and is_valid_gap(current_group):
                    nan_groups.append(finalize_gap(current_group))
                current_group = []

        # Append the last group if it exists and is valid
        if current_group and is_valid_gap(current_group):
            nan_groups.append(finalize_gap(current_group))

        gap_dict[col] = nan_groups

    return gap_dict


def get_biggest_group_valid(data: pd.Series):
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


def get_freq_delta_or_mean_time_interval(df: pd.Series | pd.DataFrame):
    check_datetime_index(df)
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


def get_outer_timestamps(idx: pd.DatetimeIndex, ref_index: pd.DatetimeIndex):
    try:
        out_start = ref_index[ref_index < idx[0]][-1]
    except IndexError:
        out_start = ref_index[0]

    try:
        out_end = ref_index[ref_index > idx[-1]][0]
    except IndexError:
        out_end = ref_index[-1]

    return out_start, out_end
