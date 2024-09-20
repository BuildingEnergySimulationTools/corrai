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

    seasonal : int | str | dt.timedelta, optional
        The seasonal component's smoothing parameter for STL. It defines how much
        the seasonal component is smoothed. If given as an integer,
        it must be an odd number. If None, a default value will be used.

    stl_additional_kwargs : dict[str, float], optional
        Additional keyword arguments for the STL decomposition.
        These allow fine-tuning of the decomposition process.
        (https://www.statsmodels.org/stable/index.html)


    Attributes
    ----------

    labels_ : pd.DataFrame
        A DataFrame with binary labels (0 or 1), indicating whether an anomaly
        is detected (1) or not (0).

    _stl_res : dict
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
        absolute_threshold: int | float,
        seasonal: int | str | dt.timedelta = None,
        stl_additional_kwargs: dict[str, float] = None,
    ):
        self.period = period
        self.absolute_threshold = absolute_threshold
        self.seasonal = seasonal
        if isinstance(self.seasonal, int):
            if self.seasonal % 2 == 0:
                raise ValueError(
                    f"seasonal={self.seasonal} is not valid, it must be "
                    f"an odd number"
                )
        self.stl_additional_kwargs = stl_additional_kwargs or {}
        self.labels_ = None
        self._stl_res = {}

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X: pd.Series | pd.DataFrame):
        check_datetime_index(X)
        if isinstance(X, pd.Series):
            X = as_1_column_dataframe(X)

        stl_args = [timedelta_to_int(self.period, X)]

        if isinstance(self.seasonal, int):
            stl_args.append(self.seasonal)
        elif self.seasonal is not None:
            seasonal = timedelta_to_int(self.seasonal, X)
            if seasonal % 2 == 0:
                seasonal += 1  # Ensure the seasonal value is odd
            stl_args.append(seasonal)

        for feat in X.columns:
            stl = STL(X[feat], *stl_args, **self.stl_additional_kwargs)
            self._stl_res[feat] = stl.fit()

        self._is_fitted = True

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        self.fit(X)
        res_df = pd.concat([res.resid for res in self._stl_res.values()], axis=1)
        res_df.columns = X.columns
        self.labels_ = (abs(res_df) > self.absolute_threshold).astype(int)

        return self.labels_
