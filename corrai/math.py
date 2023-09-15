import pandas as pd
import numpy as np

from scipy import integrate
from collections.abc import Callable

from corrai.utils import as_1_column_dataframe
from corrai.utils import check_datetime_index


def time_gradient(data):
    """
    Calculates the time gradient of a given time series `data`
    between two optional time bounds `begin` and `end`.

    Parameters:
    -----------
    data : pandas Series or DataFrame
        The time series to compute the gradient on.
        If a Series is provided, it will be converted to a DataFrame
        with a single column

    Returns:
    --------
    gradient : pandas DataFrame
        A DataFrame containing the gradient of the input time series
        for each column. The index will be a DatetimeIndex
        and the columns will be the same as the input.

    Raises:
    -------
    ValueError
        If `time_series` is not a pandas Series or DataFrame.
        If the index of `time_series` is not a pandas DateTimeIndex.

    Notes:
    ------
    This function applies the `time_data_control` function to ensure that
    the input `time_series` is formatted correctly
    for time series analysis. Then, it selects a subset of the data between
     `begin` and `end` if specified. Finally, the function computes
    the gradient of each column of the subset of the data, using the
    `np.gradient` function and the time difference between consecutive
    data points.
    """

    check_datetime_index(data)
    if isinstance(data, pd.Series):
        data = as_1_column_dataframe(data)

    ts_list = []
    for col in data:
        col_ts = data[col].dropna()

        chrono = col_ts.index - col_ts.index[0]
        chrono_sec = chrono.to_series().dt.total_seconds()

        ts_list.append(
            pd.Series(np.gradient(col_ts, chrono_sec), index=col_ts.index, name=col)
        )

    return pd.concat(ts_list, axis=1)


def time_integrate(
    data, begin=None, end=None, interpolate=True, interpolation_method="linear"
):
    """
    Perform time Integration of given time series `data` between two optional
    time bounds `begin` and `end`.

    Parameters:
    -----------
    data : pandas Series or DataFrame
        The time series to integrate. If a Series is provided, it will be
        converted to a DataFrame with a single column.
    begin : str or datetime-like, optional
        Beginning time of the selection. If None, defaults to the first
        index value of `data`.
    end : str or datetime-like, optional
        End time of the selection. If None, defaults to the last index value
        of `data`.
    interpolate : bool, optional
        Whether to interpolate missing values in the input `data` before
        integrating. If True, missing values will be filled using the specified
         `interpolation_method`. If False, missing values will be
         replaced with NaNs.
    interpolation_method : str, optional
        The interpolation method to use if `interpolate` is True. Can be one
        of 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
         or 'spline'.

    Returns:
    --------
    res_series : pandas Series
        A Series containing the result of integrating the input time series
        for each column. The index will be the same as the columns of
        the input `data`.

    Raises:
    -------
    ValueError
        If `data` is not a pandas Series or DataFrame.
        If the index of `data` is not a pandas DateTimeIndex.

    Notes:
    ------
    This function applies the `time_data_control` function to ensure that the
    input `data` is formatted correctly  for time series analysis. Then, it
    selects a subset of the data between `begin` and `end` if specified. If
    `interpolate` is True, missing values in the subset of the data will be
    filled using the specified interpolation method.
    The function then computes the integral of each column of the subset of
    the data, using the `integrate.trapz` function and the time difference
    between consecutive data points.
    """

    check_datetime_index(data)
    if isinstance(data, pd.Series):
        data = as_1_column_dataframe(data)

    if begin is None:
        begin = data.index[0]

    if end is None:
        end = data.index[-1]

    selected_ts = data.loc[begin:end, :]

    if interpolate:
        selected_ts = selected_ts.interpolate(method=interpolation_method)

    chrono = (selected_ts.index - selected_ts.index[0]).to_series()
    chrono = chrono.dt.total_seconds()

    res_series = pd.Series(dtype="float64")
    for col in data:
        res_series[col] = integrate.trapz(selected_ts[col], chrono)

    return res_series


def aggregate_time_series(
    result_df: pd.DataFrame,
    agg_method: Callable = np.mean,
    agg_method_kwarg: dict = {},
    reference_df: pd.DataFrame = None,
) -> pd.Series:
    """
    A function to perform data aggregation operations on a given DataFrame using a
    specified aggregation method. It also supports aggregation with respect to a
    reference DataFrame (eg. for error functions).

    Parameters:
    - result_df (pd.DataFrame): The DataFrame containing the data to be aggregated.
    - agg_method (Callable, optional): The aggregation method to be applied. Default is
        np.sum.
    - agg_method_kwarg (dict, optional): Additional keyword arguments to be passed to
        the aggregation method. Default is an empty dictionary.
    - reference_df (pd.DataFrame | None, optional): A reference DataFrame for error
        function aggregation. If provided, both result_df and reference_df should have
        the same shape. Default is None.

    Returns:
    - pd.Series: A pandas Series containing the aggregated values with column names as
        indices.

    Raises:
    - ValueError: If reference_df is provided and result_df and reference_df have
      inconsistent shapes.

    Example usage:
        >>> result_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> agg_series = aggregate_time_series(result_df)
        >>> print(agg_series)
    A    2
    B    5
    dtype: int64
    """

    check_datetime_index(result_df)

    if reference_df is not None:
        check_datetime_index(reference_df)
        if not result_df.shape == reference_df.shape:
            raise ValueError(
                "Cannot perform aggregation results_df and "
                "reference_df have inconsistent shapes"
            )
        return pd.Series(
            [
                agg_method(
                    result_df.iloc[:, i], reference_df.iloc[:, i], **agg_method_kwarg
                )
                for i in range(len(result_df.columns))
            ],
            index=result_df.columns,
        )

    else:
        return pd.Series(
            [
                agg_method(result_df.iloc[:, i], **agg_method_kwarg)
                for i in range(len(result_df.columns))
            ],
            index=result_df.columns,
        )
