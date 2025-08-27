import pandas as pd
import numpy as np

from collections.abc import Callable

from corrai.base.utils import check_datetime_index


def aggregate_time_series(
    result_df: pd.DataFrame,
    agg_method: Callable = np.mean,
    agg_method_kwarg: dict = None,
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
        >>> result_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> agg_series = aggregate_time_series(result_df)
        >>> print(agg_series)
    A    2
    B    5
    dtype: int64
    """

    if agg_method_kwarg is None:
        agg_method_kwarg = {}

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
