import datetime as dt

import pandas as pd
import numpy as np

from fastprogress.fastprogress import progress_bar
from sklearn.metrics import mean_squared_error, mean_absolute_error

from corrai.base.utils import check_datetime_index
from corrai.metrics import nmbe, cv_rmse

METHODS = {
    "mean": np.mean,
    "sum": np.sum,
    "nmbe": nmbe,
    "cv_rmse": cv_rmse,
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
}


def aggregate_time_series(
    results: pd.Series,
    indicator: str,
    method: str = "mean",
    agg_method_kwarg: dict = None,
    reference_time_series: pd.Series = None,
    freq: str | pd.Timedelta | dt.timedelta = None,
    prefix: str = "aggregated",
) -> pd.DataFrame:
    """
    Aggregate time series data using a specified statistical or error metric.

    This function takes a Series of DataFrames (typically representing
    simulation results), extracts the specified `indicator` column, and aggregates
    the time series across simulations using the given method. If a reference
    time series is provided, metrics that require ground truth
    (e.g., mean_absolute_error) are supported.

    Parameters
    ----------
    results : pandas.Series of pandas.DataFrame
        A series where each element is a DataFrame indexed by datetime and contains
        time series data for one simulation run.
        Each DataFrame must include the `indicator` column.

    indicator : str
        The column name in each DataFrame to extract and aggregate.

    method : str, default="mean"
        The aggregation method to use. Supported methods include:
        - "mean"
        - "sum"
        - "nmbe"
        - "cv_rmse"
        - "mean_squared_error"
        - "mean_absolute_error"

    agg_method_kwarg : dict, optional
        Additional keyword arguments to pass to the aggregation function.

    reference_time_series : pandas.Series, optional
        Reference series (`y_true`) to compare each simulation against.
        Required for error-based methods such as "mean_absolute_error".

    prefix : str, default="aggregated"
        Prefix to use for naming the output Series.

    Returns
    -------
    pandas.Series
        A Series containing the aggregated metric per simulation, indexed by the same index as `results`.

    Raises
    ------
    ValueError
        If the shapes of `results` and `reference_time_series` are incompatible.
        If the datetime index is not valid or missing.

    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.base.math import aggregate_time_series
    >>> sim_res = pd.Series([
    ...     pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2009-01-01", freq="h", periods=2)),
    ...     pd.DataFrame({"a": [3, 4]}, index=pd.date_range("2009-01-01", freq="h", periods=2)),
    ... ], index=[101, 102])

    >>> ref = pd.Series([1, 1], index=pd.date_range("2009-01-01", freq="h", periods=2))

    >>> aggregate_time_series(sim_res, indicator="a")
    101    1.5
    102    3.5
    Name: aggregated_a, dtype: float64

    >>> aggregate_time_series(sim_res, indicator="a", method="mean_absolute_error", reference_time_series=ref)
    101    0.5
    102    2.5
    Name: aggregated_a, dtype: float64
    """

    agg_method_kwarg = {} if agg_method_kwarg is None else agg_method_kwarg
    method = METHODS[method]

    for df in results:
        check_datetime_index(df)
    agg_df = pd.concat([df[indicator].rename(i) for i, df in results.items()], axis=1)

    if reference_time_series is not None:
        check_datetime_index(reference_time_series)
        if not agg_df.shape[0] == reference_time_series.shape[0]:
            raise ValueError(
                "Cannot perform aggregation, Dataframes in results and "
                "reference_time_series have inconsistent shapes"
            )

    if freq is not None:
        grouped_results = agg_df.groupby(pd.Grouper(freq=freq))
        len_grouped = len(grouped_results)
        val = []
        if reference_time_series is not None:
            ref_grouped = reference_time_series.groupby(pd.Grouper(freq=freq))
            for (gr_tstamp, gr_res), (_, gr_ref) in progress_bar(
                zip(grouped_results, ref_grouped), total=len_grouped
            ):
                val.append(
                    pd.Series(
                        {
                            col: method(
                                y_true=gr_ref,
                                y_pred=gr_res[col],
                                **agg_method_kwarg,
                            )
                            for col in gr_res.columns
                        },
                        name=gr_tstamp,
                    )
                )
        else:
            for (gr_tstamp, gr_res) in progress_bar(grouped_results):
                ts_res = gr_res.apply(method, **agg_method_kwarg)
                ts_res.name = gr_tstamp
                val.append(ts_res)

        return pd.concat(val, axis=1)
    else:
        if reference_time_series is not None:
            res = pd.Series(
                {
                    col: method(
                        y_true=reference_time_series,
                        y_pred=agg_df[col],
                        **agg_method_kwarg,
                    )
                    for col in agg_df.columns
                }
            )
        else:
            res = agg_df.apply(method, **agg_method_kwarg)

        res.index = results.index
        res.name = f"{prefix}_{indicator}"
        return res.to_frame()
