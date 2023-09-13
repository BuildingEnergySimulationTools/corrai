import numpy as np
import pandas as pd
from collections.abc import Callable
from corrai.utils import check_datetime_index


class AggregationMixin:
    """
    A mixin class for performing data aggregation operations on a DataFrame.

    This mixin provides a method `get_aggregated` to perform aggregation operations on
    a  given DataFrame using a specified aggregation method. It also supports
    aggregation with respect to a reference DataFrame for error functions.

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
        >>> mixin = AggregationMixin()
        >>> result_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> agg_series = mixin.get_aggregated(result_df)
        >>> print(agg_series)
    A    2
    B    5
    dtype: int64
    """

    def get_aggregated(
        self,
        result_df: pd.DataFrame,
        agg_method: Callable = np.mean,
        agg_method_kwarg: dict = {},
        reference_df: pd.DataFrame | None = None,
    ) -> pd.Series:
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
                        result_df[col], reference_df.iloc[:, i], **agg_method_kwarg
                    )
                    for i, col in enumerate(result_df.columns)
                ],
                index=result_df.columns,
            )

        else:
            return pd.Series(
                [
                    agg_method(result_df[col], **agg_method_kwarg)
                    for col in result_df.columns
                ],
                index=result_df.columns,
            )
