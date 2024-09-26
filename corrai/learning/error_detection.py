import datetime as dt

import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from statsmodels.tsa.seasonal import STL

from corrai.base.utils import check_datetime_index, as_1_column_dataframe


def timedelta_to_int(td: int | str | dt.timedelta, df):
    if isinstance(td, int):
        return td
    else:
        if isinstance(td, str):
            td = pd.to_timedelta(td)
        return int(td / df.index.freq)


def validate_odd_param(param_name, param_value):
    if isinstance(param_value, int) and param_value % 2 == 0:
        raise ValueError(
            f"{param_name}={param_value} is not valid, it must be an odd number"
        )


def process_stl_odd_args(param_name, param_value, X, stl_kwargs):
    if isinstance(param_value, int):
        # Is odd already check at init in case of int
        stl_kwargs[param_name] = param_value
    elif param_value is not None:
        processed_value = timedelta_to_int(param_value, X)
        if processed_value % 2 == 0:
            processed_value += 1  # Ensure the value is odd
        stl_kwargs[param_name] = processed_value


class STLEDetector(BaseEstimator, ClusterMixin):
    """
    A custom anomaly detection model based on statsmodel STL
    (Seasonal and Trend decomposition using Loess).

    The STL decomposition breaks down time series into three components: trend,
    seasonal, and residual. This class uses the residual component to detect anomalies
    based on the absolute threshold (absolute value of residual exceed threshold).

    See statsmodel doc for additional STL configuration.
    (https://www.statsmodels.org/stable/index.html)


    Parameters
    ----------
    period : int | str | dt.timedelta
        The period of the time series (e.g., daily, weekly, monthly, etc.).
        Can be an integer, string, or timedelta.
        This defines the seasonal periodicity for the STL decomposition.

    absolute_threshold : int | float
        The threshold value for residuals. Any residuals exceeding this threshold
        are considered anomalies.

    trend : int | str | dt.timedelta, optional
        The length of the trend smoother. Must be odd and larger than season
        Statsplot indicate it is usually around 150% of season.
        Strongly depends on your time series.

    seasonal : int | str | dt.timedelta, optional
        The seasonal component's smoothing parameter for STL. It defines how much
        the seasonal component is smoothed. If given as an integer,
        it must be an odd number. If None, a default value will be used.

    stl_kwargs : dict[str, float], optional
        Additional keyword arguments for the STL decomposition.
        These allow fine-tuning of the decomposition process.
        (https://www.statsmodels.org/stable/index.html)


    Attributes
    ----------

    labels_ : pd.DataFrame
        A DataFrame with binary labels (0 or 1), indicating whether an anomaly
        is detected (1) or not (0).

    stl_res : dict
        A dictionary that holds the fitted STL results for each feature in the dataset.

    Methods
    -------
    __sklearn_is_fitted__():
        Checks whether the model has been fitted and returns a boolean
        indicating the fitted status.

    fit(X: pd.Series | pd.DataFrame):
        Fits the STL model to the input time series data. Computes and stores
        residuals for each column in X.

    predict(X: pd.Series | pd.DataFrame):
        Fits the model and predicts anomalies by comparing the residuals with
        the absolute threshold. Returns a 0-1 Pandas DataFrame

    Raises
    ------
    ValueError
        If the seasonal parameter is an even number when passed as an integer.

    """

    def __init__(
        self,
        period: int | str | dt.timedelta,
        trend: int | str | dt.timedelta,
        absolute_threshold: int | float,
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, float] = None,
    ):
        self.period = period
        self.trend = trend
        self.absolute_threshold = absolute_threshold
        self.seasonal = seasonal
        validate_odd_param("seasonal", self.seasonal)
        validate_odd_param("trend", self.trend)

        self.stl_kwargs = stl_kwargs or {}
        self.labels_ = None
        self.stl_res = {}

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X: pd.Series | pd.DataFrame):
        check_datetime_index(X)
        if isinstance(X, pd.Series):
            X = as_1_column_dataframe(X)

        self.stl_kwargs["period"] = timedelta_to_int(self.period, X)
        process_stl_odd_args("seasonal", self.seasonal, X, self.stl_kwargs)
        process_stl_odd_args("trend", self.trend, X, self.stl_kwargs)

        for feat in X.columns:
            stl = STL(X[feat], **self.stl_kwargs)
            self.stl_res[feat] = stl.fit()

        self._is_fitted = True

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        self.fit(X)
        res_df = pd.concat([res.resid for res in self.stl_res.values()], axis=1)
        res_df.columns = X.columns
        self.labels_ = (abs(res_df) > self.absolute_threshold).astype(int)

        return self.labels_
