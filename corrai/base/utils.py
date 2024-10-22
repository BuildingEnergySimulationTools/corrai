import pandas as pd
import numpy as np
import datetime as dt
from collections.abc import Iterable


def _reshape_1d(sample):
    if isinstance(sample, pd.DataFrame):
        return sample.squeeze()
    elif isinstance(sample, np.ndarray):
        return sample.flatten()


def check_and_return_dt_index_df(X: pd.Series | pd.DataFrame):
    if not (isinstance(X, pd.Series) or isinstance(X, pd.DataFrame)):
        raise ValueError(
            f"Invalid X data, was expected an instance of pandas Dataframe "
            f"or Pandas Series. Got {type(X)}"
        )
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X index is not a pandas DateTime index")

    return X.to_frame() if isinstance(X, pd.Series) else X


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


def get_data_blocks(
    data: pd.Series | pd.DataFrame,
    is_null: bool = False,
    cols: str | list[str] = None,
    lower_td_threshold: str | dt.timedelta = None,
    upper_td_threshold: str | dt.timedelta = None,
    lower_threshold_inclusive: bool = True,
    upper_threshold_inclusive: bool = True,
    return_combination=True,
):
    """
    Identifies groups of valid data if is_null = False, or groups of nan if
    is_null = True (gaps in measurements).
    Returns them in a dictionary as list of DateTimeIndex. The keys values are
    data columns (or name if data is a Series).
    The groups can be filtered using lower_dt_threshold or higher_dt_threshold.
    The argument return indicates if an additional key must be set to the dictionary
    to account for all data presence.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The input time series data with a DateTime index. NaN values are
        considered gaps.
    is_null : Bool, default False
        Whether to return groups with valid data, or groups of Nan values
        (is_null = True)
    cols : str or list[str], optional
        The columns in the DataFrame for which to detect gaps. If None (default), all
        columns are considered.
    lower_td_threshold : str or timedelta, optional
        The minimum duration of a period for it to be considered valid.
        Can be passed as a string (e.g., '1d' for one day) or a `timedelta`.
        If None, no threshold is applied, NaN values are considered gaps.
    upper_td_threshold : str or timedelta, optional
        The maximum duration of a period for it to be considered valid.
        Can be passed as a string (e.g., '1d' for one day) or a `timedelta`.
        If None, no threshold is applied, NaN values are considered gaps.
    lower_threshold_inclusive : bool, optional
        Include the gaps of exactly lower_td_threshold duration
    upper_threshold_inclusive : bool, optional
        Include the gaps of exactly upper_td_threshold duration
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

    if isinstance(lower_td_threshold, str):
        lower_td_threshold = pd.to_timedelta(lower_td_threshold)
    elif lower_td_threshold is None:
        lower_td_threshold = pd.to_timedelta(0)

    if isinstance(upper_td_threshold, str):
        upper_td_threshold = pd.to_timedelta(upper_td_threshold)
    elif upper_td_threshold is None:
        upper_td_threshold = pd.Timedelta.max

    freq = get_freq_delta_or_min_time_interval(data)
    # If data index has no frequency, a frequency based on minimum
    # timedelta is set.
    df = data.asfreq(freq)

    df = df.isnull() if is_null else ~df.isnull()

    if return_combination:
        df["combination"] = df.any(axis=1)
        cols += ["combination"]

    def is_valid_block(group, lgt, hgt):
        new_block = pd.DatetimeIndex(group)
        block_duration = new_block.max() - new_block.min() + freq

        lower_check = (
            block_duration >= lgt if lower_threshold_inclusive else block_duration > lgt
        )
        upper_check = (
            block_duration <= hgt if upper_threshold_inclusive else block_duration < hgt
        )

        return lower_check and upper_check

    def finalize_block(current_group):
        # For indexes where frequency has been imposed,
        # Get back to the original data index
        current_group = [ts for ts in current_group if ts in data.index]
        new_block_index = pd.DatetimeIndex(current_group)
        new_block_index.freq = new_block_index.inferred_freq
        return new_block_index

    block_dict = {}
    for col in cols:
        groups = []
        current_group = []

        for timestamp in df.index:
            if df.loc[timestamp, col]:
                current_group.append(timestamp)
            else:
                if current_group and is_valid_block(
                    current_group, lower_td_threshold, upper_td_threshold
                ):
                    groups.append(finalize_block(current_group))
                current_group = []

        # Append the last group if it exists and is valid
        if current_group and is_valid_block(
            current_group, lower_td_threshold, upper_td_threshold
        ):
            groups.append(finalize_block(current_group))

        block_dict[col] = groups

    return block_dict


def get_freq_delta_or_min_time_interval(df: pd.Series | pd.DataFrame):
    check_datetime_index(df)
    freq = df.index.inferred_freq
    if freq:
        freq = pd.to_timedelta("1" + freq) if freq.isalpha() else pd.to_timedelta(freq)
    else:
        freq = df.index.to_frame().diff().min()[0]

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


def get_gaps_mask(
    data: pd.Series, operator: str, size: str | pd.Timedelta | dt.timedelta = None
):
    """
    Returns np boolean array of shape (len(data),) with True if an entry of data is
    part of a gap of size less han or equal to, LTE, less than LT, greater than or
     equal to GTE, greater than GT the specified size.
    :param data: Pandas Series with DatetimeIndex
    :param operator : str the operator for the test.
    Must be one of 'GTE', 'GT', 'LT', 'LTE'.
    Will raise an error otherwise
    :param size: str | timedelta : size threshold of the gap
    :return:
    """

    data = check_and_return_dt_index_df(data)
    size = pd.to_timedelta(size) if isinstance(size, str) else size

    if operator == "GTE":
        gaps = get_data_blocks(
            data, is_null=True, lower_td_threshold=size, return_combination=False
        )
    elif operator == "GT":
        gaps = get_data_blocks(
            data,
            is_null=True,
            lower_td_threshold=size,
            return_combination=False,
            lower_threshold_inclusive=False,
        )

    elif operator == "LTE":
        gaps = get_data_blocks(
            data, is_null=True, upper_td_threshold=size, return_combination=False
        )

    elif operator == "LT":
        gaps = get_data_blocks(
            data,
            is_null=True,
            upper_td_threshold=size,
            return_combination=False,
            upper_threshold_inclusive=False,
        )

    else:
        raise ValueError(
            f"invalid operator {operator}, choose one of 'LTE', 'LT', 'GTE', 'GT'"
        )

    gaps_idx = gaps[data.columns[0]]

    if gaps_idx:
        if len(gaps_idx) > 1:
            final_index = gaps_idx[0]
            for idx in gaps_idx[1:]:
                final_index = final_index.union(idx)
        else:
            final_index = gaps_idx[0]
    else:
        final_index = pd.DatetimeIndex([])

    return np.isin(data.index, final_index)
