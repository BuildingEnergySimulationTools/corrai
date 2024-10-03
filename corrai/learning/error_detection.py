import datetime as dt
import typing
import warnings

from abc import ABC

import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA

from corrai.base.utils import check_datetime_index, as_1_column_dataframe


def timedelta_to_int(td: int | str | dt.timedelta, df):
    if isinstance(td, int):
        return td
    else:
        if isinstance(td, str):
            td = pd.to_timedelta(td)
        return abs(int(td / df.index.freq))


def validate_odd_param(param_name, param_value):
    if isinstance(param_value, int) and param_value % 2 == 0:
        raise ValueError(
            f"{param_name}={param_value} is not valid, it must be an odd number"
        )


def process_stl_odd_args(param_name, X, stl_kwargs):
    param_value = stl_kwargs[param_name]
    if isinstance(param_value, int):
        # Is odd already check at init in case of int
        stl_kwargs[param_name] = param_value
    elif param_value is not None:
        processed_value = timedelta_to_int(param_value, X)
        if processed_value % 2 == 0:
            processed_value += 1  # Ensure the value is odd
        stl_kwargs[param_name] = processed_value


class STLBC(ABC, BaseEstimator):
    def __init__(
        self,
        period: int | str | dt.timedelta,
        trend: int | str | dt.timedelta,
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, typing.Any] = None,
    ):
        self.stl_kwargs = {} if stl_kwargs is None else stl_kwargs
        self.stl_kwargs["period"] = period
        validate_odd_param("trend", trend)
        self.stl_kwargs["trend"] = trend
        if seasonal is not None:
            validate_odd_param("seasonal", seasonal)
            self.stl_kwargs["seasonal"] = seasonal

    def _pre_fit(self, X: pd.Series | pd.DataFrame):
        check_datetime_index(X)
        if isinstance(X, pd.Series):
            X = as_1_column_dataframe(X)

        self.stl_kwargs["period"] = timedelta_to_int(self.stl_kwargs["period"], X)
        process_stl_odd_args("trend", X, self.stl_kwargs)
        if "seasonal" in self.stl_kwargs.keys():
            process_stl_odd_args("seasonal", X, self.stl_kwargs)


class STLEDetector(STLBC, ClusterMixin):
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
        super().__init__(period, trend, seasonal, stl_kwargs)
        self.absolute_threshold = absolute_threshold
        self.labels_ = None
        self.stl_res_ = {}

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        self._pre_fit(X)
        for feat in X.columns:
            self.stl_res_[feat] = STL(X[feat], **self.stl_kwargs).fit()

        self._is_fitted = True

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        self.fit(X)
        res_df = pd.concat([res.resid for res in self.stl_res_.values()], axis=1)
        res_df.columns = X.columns
        self.labels_ = (abs(res_df) > self.absolute_threshold).astype(int)

        return self.labels_


class SkSTLForecast(STLBC, RegressorMixin):
    def __init__(
        self,
        period: int | str | dt.timedelta,
        trend: int | str | dt.timedelta,
        ar_model=None,
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, float] = None,
        ar_kwargs: dict = None,
        backcast: bool = False,
    ):
        super().__init__(period, trend, seasonal, stl_kwargs)
        self.backcast = backcast
        self.ar_model = ARIMA if ar_model is None else ar_model
        self.ar_kwargs = {} if ar_kwargs is None else ar_kwargs

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        self._pre_fit(X)
        self.training_freq_ = (
            X.index.freq if X.index.freq is not None else X.index.inferred_freq
        )
        if self.backcast:
            X = X[::-1]
        self.train_dat_end_ = X.index[-1]
        self.forecaster_ = {}

        for feat in X:
            self.forecaster_[feat] = STLForecast(
                endog=X[feat].to_numpy(),
                model=self.ar_model,
                model_kwargs=self.ar_kwargs,
                **self.stl_kwargs,
            ).fit()

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(
            self,
            attributes=[
                "forecaster_",
                "train_dat_end_",
                "training_freq_",
            ],
        )

        if X.index.freq != self.training_freq_:
            raise ValueError(
                f"Required prediction freq {X.index.freq} "
                f"differs from training_freq_ {self.training_freq_}"
            )

        if (self.backcast and X.index[-1] >= self.train_dat_end_) or (
            not self.backcast and X.index[0] <= self.train_dat_end_
        ):
            direction = "future" if self.backcast else "past"
            raise ValueError(
                f"Cannot forecast on {direction} values or training data. "
                f"{'Backcast' if self.backcast else 'Forecast'} can only happen "
                f"{'before' if self.backcast else 'after'} {self.train_dat_end_}"
            )

        output_index = X.index[::-1] if self.backcast else X.index

        if set(self.forecaster_.keys()) != set(X.columns):
            warnings.warn(
                "Columns in X differs from columns in the training DataSet. "
                "Forecast will be performed for the trained data",
                UserWarning,
            )

        casting_steps = int(
            len(output_index)
            + abs(output_index[0] - self.train_dat_end_) / self.training_freq_
            - 1
        )
        steps_to_jump = casting_steps - len(output_index)
        inferred_df = pd.DataFrame(index=output_index)
        for feat in self.forecaster_.keys():
            cast = self.forecaster_[feat].forecast(casting_steps)
            inferred_df[feat] = cast[steps_to_jump:]

        return inferred_df
