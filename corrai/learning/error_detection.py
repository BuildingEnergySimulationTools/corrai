import datetime as dt
import typing
import warnings

from abc import ABC

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA

from corrai.base.utils import check_datetime_index, check_and_return_dt_index_df

MODEL_MAP = {"ARIMA": ARIMA}

MODEL_DEFAULT_CONF = {"ARIMA": {"order": (1, 1, 0), "trend": "t"}}


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
        period: int | str | dt.timedelta = "24h",
        trend: int | str | dt.timedelta = "15d",
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, typing.Any] = None,
    ):
        self.stl_kwargs = stl_kwargs
        self.period = period
        self.trend = trend
        self.seasonal = seasonal

    def _pre_fit(self, X: pd.Series | pd.DataFrame):
        self.stl_kwargs = {} if self.stl_kwargs is None else self.stl_kwargs

        check_datetime_index(X)
        if isinstance(X, pd.Series):
            X = X.to_frame()
        check_array(X)

        self.stl_kwargs["period"] = timedelta_to_int(self.period, X)
        validate_odd_param("trend", self.trend)
        self.stl_kwargs["trend"] = self.trend
        process_stl_odd_args("trend", X, self.stl_kwargs)
        if self.seasonal is not None:
            self.stl_kwargs["seasonal"] = self.seasonal
            process_stl_odd_args("seasonal", X, self.stl_kwargs)


class STLEDetector(ClassifierMixin, STLBC):
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
        period: int | str | dt.timedelta = "24h",
        trend: int | str | dt.timedelta = "15d",
        absolute_threshold: int | float = 100,
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, float] = None,
    ):
        super().__init__(period, trend, seasonal, stl_kwargs)
        self.absolute_threshold = absolute_threshold

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        self._pre_fit(X)
        self.stl_fit_res_ = {}
        for feat in X.columns:
            self.stl_fit_res_[feat] = STL(X[feat], **self.stl_kwargs).fit()

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["stl_fit_res_"])
        check_datetime_index(X)
        if isinstance(X, pd.Series):
            X = X.to_frame()
        check_array(X)

        res_df = pd.concat([res.resid for res in self.stl_fit_res_.values()], axis=1)
        res_df.columns = X.columns
        return (abs(res_df) > self.absolute_threshold).astype(int)


class SkSTLForecast(RegressorMixin, STLBC):
    """
    A model designed for time series forecasting or backcasting
    (predicting past values).
    It applies seasonal-trend decomposition (STL) to the training data to capture both
    trend and seasonal patterns. The model then uses ARIMA or a custom autoregressive
    model to predict these components, as well as the overall observed variable.

    Parameters
    ----------
    period : int, str, or datetime.timedelta
        The period of the time series (e.g., daily, weekly, monthly, etc.).
        Can be an integer, string, or timedelta.
        This defines the seasonal periodicity for the STL decomposition.

    trend : int, str, or datetime.timedelta
        The length of the trend smoother. If an int is specified, it must be odd and
        larger than season. Statsplot indicate it is usually around 150% of season.
        Strongly depends on your time series.

    ar_model : object, optional
        A string corresponding to the name of the Autoregressive model to be used
        to predict STL trend an periodic component.
        The name must be chosen among MODEL_MAP keys()
        If not provided, ARIMA will be used as the default model.

    seasonal : int, str, or datetime.timedelta, optional
        The seasonal component's smoothing parameter for STL. It defines how much
        the seasonal component is smoothed. If given as an integer,
        it must be an odd number. If None, a default value will be used.

    stl_kwargs : dict[str, float], optional
        Additional keyword arguments for the STL decomposition.
        These allow fine-tuning of the decomposition process.
        (https://www.statsmodels.org/stable/index.html)

    ar_kwargs : dict, optional
        Keyword arguments to be passed to the autoregressive model
        (e.g., order for ARIMA).
    backcast : bool, optional
        If True, the model will be trained to backcast (predict the past), otherwise,
        it will perform standard forward forecasting.

    Attributes
    ----------
    forecaster_ : dict
        Dictionary containing the fitted forecaster for each feature in the time series.
    train_dat_end_ : pandas.Timestamp
        Timestamp of the last data point used in training.
    training_freq_ : pandas.tseries.offsets.BaseOffset
        Frequency of the training data, either provided explicitly or inferred.

    """

    def __init__(
        self,
        period: int | str | dt.timedelta = "24h",
        trend: int | str | dt.timedelta = "15d",
        ar_model: str = "ARIMA",
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, float] = None,
        ar_kwargs: str | dict = None,
        backcast: bool = False,
    ):
        super().__init__(period, trend, seasonal, stl_kwargs)
        self.backcast = backcast
        self.ar_model = ar_model
        self.ar_kwargs = ar_kwargs

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        ar_model = MODEL_MAP[self.ar_model]
        if self.ar_kwargs is None:
            ar_kwargs = MODEL_DEFAULT_CONF[self.ar_model]
        else:
            ar_kwargs = self.ar_kwargs

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
                model=ar_model,
                model_kwargs=ar_kwargs,
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
        check_datetime_index(X)
        X = X.to_frame() if isinstance(X, pd.Series) else X
        check_array(X)

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

        return inferred_df.sort_index()
