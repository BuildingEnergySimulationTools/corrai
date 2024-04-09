import datetime as dt
import itertools

import numpy as np
import pandas as pd
from plotly import colors as colors, graph_objects as go
from scipy.signal import argrelextrema
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corrai.transformers import PdSkTransformer
from corrai.base.utils import (
    as_1_column_dataframe,
    check_datetime_index,
    float_to_hour,
    _reshape_1d,
)


def get_hours_switch(X, diff_filter_threshold=0, switch="positive"):
    """
    From a time series determine the number of hour since the beginning
    of the day when the signal rises (switch='positive') or decreases
    (switch='negative') or both (switch='both').
    numpy argrelextrema and diff_filter_threshold are is used to filter
    small variations

    :param X: pandas Series or one column DataFrame with DatetimeIndex
    :param diff_filter_threshold: float or integer
    :param switch: 'positive', 'negative', 'both
    :return: list of hour string formatted "%H:%M"
    """

    X = as_1_column_dataframe(X)
    check_datetime_index(X)

    df = X.dropna().copy().diff()
    data_col_name = X.columns[0]
    if switch == "positive":
        df.loc[df[data_col_name] < 0, data_col_name] = 0
    elif switch == "negative":
        df.loc[df[data_col_name] > 0, data_col_name] = 0
        df = abs(df)
    elif switch == "both":
        df = abs(df)
    else:
        raise ValueError(f"Unknown value {switch} as switch argument")

    df["start_day"] = pd.DatetimeIndex(list(map(lambda x: x.date(), df.index)))
    if df.index.tz:
        df["start_day"] = pd.DatetimeIndex(list(df["start_day"])).tz_localize(
            tz=df.index.tz
        )
    df["hour_since_beg_day"] = (df.index - df.start_day).dt.total_seconds() / 60 / 60

    max_index = argrelextrema(df[data_col_name].to_numpy(), np.greater)[0]
    max_values = df[data_col_name].iloc[max_index]
    filt_max_values_index = max_values.loc[max_values > diff_filter_threshold].index

    df.loc[~df.index.isin(filt_max_values_index), "hour_since_beg_day"] = np.nan

    return float_to_hour(list(df.hour_since_beg_day.dropna()))


class KdeSetPointIdentificator(BaseEstimator, ClusterMixin):
    """
    KDE-based set point detection algorithm. The algorithm fits a Kernel Density
    Estimate to the provided data and detects local maxima as set points. The
    set points are then used as the basis for the clustering of new data points.
    Clustering of new points is performed by labeling each new point with the
    index of the nearest set point.

    Parameters
    ----------
    bandwidth : float, default=0.1
        The bandwidth parameter for the KDE estimator.

    domain_tol : float, default=0.2
        The tolerance parameter used for determining the range of values
        to consider when estimating the probability density function.

    domain_n_sample : int, default=1000
        The number of samples to use when estimating the probability density
        function.

    lik_filter : float, default=1
        The minimum likelihood threshold required for a set point to be
        included in the model.

    cluster_tol : float, default=0.05
        The tolerance parameter used for assigning data points to clusters.

    Attributes
    ----------
    kde : sklearn.neighbors.KernelDensity
        The KDE estimator used to estimate the probability density function.

    domain : ndarray of shape (n_samples,)
        The array of samples used to estimate the probability density function.

    set_points : ndarray of shape (n_set_points,)
        The set points detected by the algorithm.

    set_points_likelihood : ndarray of shape (n_set_points,)
        The likelihoods associated with each set point.

    Methods
    -------
    fit(X, y=None)
        Fit the model to the given data.

    predict(X)
        Use the fitted model to cluster new data points.
    """

    def __init__(
        self,
        bandwidth=0.1,
        domain_tol=0.2,
        domain_n_sample=1000,
        lik_filter=1,
        cluster_tol=0.05,
    ):
        self.bandwidth = bandwidth
        self.domain_tol = domain_tol
        self.domain_n_sample = domain_n_sample
        self.lik_filter = lik_filter
        self.cluster_tol = cluster_tol
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.domain = None
        self.set_points = None
        self.set_points_likelihood = None
        self.labels_ = None

    def fit(self, X, y=None):
        X = as_1_column_dataframe(X)

        tolerance = np.mean(np.abs(X), axis=0) * self.domain_tol

        self.domain = np.linspace(
            np.min(X, axis=0) - tolerance,
            np.max(X, axis=0) + tolerance,
            self.domain_n_sample,
        )

        self.kde.fit(X.to_numpy())

        like_domain = np.exp(self.kde.score_samples(self.domain))
        max_index = argrelextrema(like_domain, np.greater)
        self.set_points = self.domain[max_index[0]]

        # Pass if perfectly flat (all values equal in the domain)
        if self.set_points.shape[0] > 0:
            set_points_likelihood = np.exp(self.kde.score_samples(self.set_points))
            like_mask = set_points_likelihood > self.lik_filter

            self.set_points = self.set_points[like_mask].flatten()
            self.set_points_likelihood = set_points_likelihood[like_mask]
            self.labels_ = self.predict(X)

        return self

    def predict(self, X):
        X = as_1_column_dataframe(X)
        X = X.to_numpy()
        X = _reshape_1d(X)

        x_cluster = np.empty(X.shape[0])
        x_cluster[:] = np.nan
        for nb, sp in enumerate(self.set_points):
            mask = np.logical_and(
                (X > (sp - self.cluster_tol)), (X < (sp + self.cluster_tol))
            )
            x_cluster[mask] = nb

        return np.nan_to_num(x_cluster, nan=-1)


def set_point_identifier(
    X: pd.DataFrame | pd.Series, estimator: KdeSetPointIdentificator
):
    """
    Identifies set points in a time series data using kernel density estimation.
    Uses CorrAI KdeSetPointIdentificator combined with a transformer to scale the data.
    If no scaler is provided, the function uses scikit learn StandardScaler

    Parameters
    ----------
    X : pandas.DataFrame
        The input data containing time series data to identify set points in. The data
        must have a DateTimeIndex.
    estimator : object, optional
        A corrai KdeSetPointIdentificator. Default is None, which will use a
        `KdeSetPointIdentificator` object with default parameters values.
    sk_scaler : object, optional
        A scikit-learn scaler object that can transform data.
        Default is None, which will use a `StandardScaler` object wrapped in a
        corrai `PdSkTransformer` object.
    cols : list, optional
        The column names to identify set points in. Default is None, which will
        identify set points in all columns.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the identified set points for each specified column,
        with MultiIndex row labels indicating the period and the set point number.
        Returns None if no set points were identified.

    Raises
    ------
    ValueError
        If the input data is not a DataFrame and does not have a DateTimeIndex.

    Notes
    -----
    The set point identification is performed by fitting the kernel density estimator
    to each column of the input data, and then finding the peaks of the density
    estimate, which shall correspond to the set points.
    """
    if isinstance(X, pd.Series):
        X = as_1_column_dataframe(X)
    check_datetime_index(X)

    model = make_pipeline(PdSkTransformer(StandardScaler()), estimator)

    pd_scaler = model.named_steps["pdsktransformer"]
    kde = model.named_steps["kdesetpointidentificator"]

    duration = X.index[-1] - X.index[-0]

    period = pd.Period(
        X.index[0].strftime("%Y-%m-%d %H-%M-%S"),
        day=duration.days,
        second=duration.seconds,
    )

    multi_series_list = []
    for col in X.columns:
        model.fit(X[[col]])
        try:
            set_points = pd_scaler.inverse_transform(kde.set_points)
        except ValueError:
            set_points = None

        if set_points is not None:
            index_tuple = [
                (period, f"set_point_{i}") for i in range(set_points.shape[0])
            ]
            multiindex = pd.MultiIndex.from_tuples(index_tuple)
            set_points_series = pd.Series(
                set_points.to_numpy().flatten(), index=multiindex, name=col
            )
            multi_series_list.append(set_points_series)

    if multi_series_list:
        return pd.concat(multi_series_list, axis=1)
    else:
        return None


def moving_window_set_point_identifier(
    X: pd.DataFrame | pd.Series,
    window_size: dt.timedelta,
    slide_size: dt.timedelta,
    estimator: KdeSetPointIdentificator,
):
    """
    Identify set points in a time series dataset using a moving window approach
    with kernel density estimation.

    Parameters
    ----------
    X : pandas.DataFrame or pandas.Series
        Input data containing time series data for set point identification.
        Must have a DateTimeIndex.

    window_size : datetime.timedelta
        Size of the moving window for set point identification.

    slide_size : datetime.timedelta
        Size of the sliding step between consecutive windows.

    estimator : KdeSetPointIdentificator
        An instance of the CorrAI KdeSetPointIdentificator class used for set point
        identification.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame with identified set points for each specified column within the m
        oving windows. MultiIndex row labels indicate the period and set point number.
        Returns None if no set points are identified in any window.

    Raises
    ------
    ValueError
        If the input data is not a DataFrame, a Series or lacks a DateTimeIndex.

    Notes
    -----
    Set point identification is performed using the provided kernel density estimator
    within a moving window approach. The function iterates through the time series
    data with windows of a specified size and identifies set points in each window.
    """
    if isinstance(X, pd.Series):
        X = as_1_column_dataframe(X)
    check_datetime_index(X)
    start_date = X.index[0]
    end_date = X.index[-1]

    groups_res_list = []
    while start_date <= end_date - window_size:
        selected_data = X.loc[start_date : start_date + window_size, :]
        groups_res_list.append(
            set_point_identifier(X=selected_data, estimator=estimator)
        )

        start_date += slide_size

    if not all(v is None for v in groups_res_list):
        return pd.concat(groups_res_list)
    else:
        return None


def plot_kde_set_point(
    X: pd.DataFrame | pd.Series,
    estimator: KdeSetPointIdentificator = None,
    sk_scaler: bool = None,
    title="Clustered Timeseries",
    y_label="[-]",
):
    """
    Plots a scatter plot of the input data with different colors representing the
    clusters of the data points identified by the `KdeSetPointIdentificator` estimator.

    Parameters
    ----------
        X : pandas.DataFrame | pandas.Series
            The input data with shape (n_samples, 1).
        estimator : corrai.learning.KdeSetPointIdentificator, optional
            An instance of KdeSetPointIdentificator to use for clustering the data.
            Defaults to `KdeSetPointIdentificator` with default values.
        sk_scaler : object, optional
            An instance of the scaler to use for preprocessing the
            data. Defaults to `StandardScaler`.
        title : str, optional
            The title of the plot. Defaults to "Clustered Timeseries".
        y_label : str
            The label of the y-axis. Defaults to "[-]".
    """
    X = as_1_column_dataframe(X)

    if sk_scaler is None:
        pd_scaler = PdSkTransformer(StandardScaler())
    else:
        pd_scaler = PdSkTransformer(sk_scaler)

    if estimator is None:
        estimator = KdeSetPointIdentificator()

    model = make_pipeline(pd_scaler, estimator)
    cluster_col = model.fit_predict(X)

    color_dict = {
        key: value
        for key, value in zip(
            sorted(set(cluster_col)), itertools.cycle(colors.DEFAULT_PLOTLY_COLORS)
        )
    }

    color_list = [color_dict[val] for val in cluster_col]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=X.index,
            y=X.squeeze(),
            mode="markers",
            marker=dict(color=color_list),
            name=X.squeeze().name,
        )
    )

    fig.update_layout(
        dict(
            title=title,
            yaxis_title=y_label,
        )
    )

    annotation_text = (
        f"<span style='color:{list(color_dict.values())[0]}'>"
        f"Not regulated / transient</span><br>"
    )
    for i, color in enumerate(list(color_dict.values())[1:]):
        annotation_text += f"<span style='color:{color}'>" f"Set point {i}</span><br>"
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        text=annotation_text,
        showarrow=False,
        align="right",
        xanchor="right",
        yanchor="top",
        font=dict(size=14),
    )

    fig.show()


def plot_time_series_kde(
    X: pd.DataFrame | pd.Series,
    title: str = "Likelihood and data",
    x_label: str = "",
    scaled: bool = True,
    bandwidth: float = 0.1,
    xbins: int = 100,
):
    """
    Plots the likelihood function and histogram of the input data as estimated by
    kernel density estimation.

    Parameters
    ----------
        X (pandas.DataFrame): The input data with shape (n_samples, n_features).
        title (str): The title of the plot. Defaults to "Likelihood and data".
        x_label (str): The label of the x-axis. Defaults to "".
        scaled (bool): Whether to scale the input data using `StandardScaler`.
            Defaults to True.
        bandwidth (float): The bandwidth parameter for the kernel density estimator.
            Defaults to 0.1.
        xbins (int): The number of bins to use for the histogram. Defaults to 100.
    """

    X = as_1_column_dataframe(X)

    X = X.dropna()

    scaler = StandardScaler()
    if scaled:
        X = scaler.fit_transform(X)
    else:
        X = X.to_numpy()

    domain = np.linspace(
        np.min(X, axis=0) - (np.mean(np.abs(X), axis=0) * 0.15),
        np.max(X, axis=0) + (np.mean(np.abs(X), axis=0) * 0.15),
        1000,
    )

    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X)

    log_like_domain = np.exp(kde.score_samples(domain))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=domain.flatten(),
            y=log_like_domain.flatten(),
            mode="lines",
            name="likelihood x original",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=X.squeeze(),
            histnorm="probability density",
            name="measured data",
            nbinsx=xbins,
        )
    )
    fig.update_layout(
        dict(
            title=title,
            xaxis_title=x_label,
            yaxis_title="[-]",
        )
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
    )

    fig.show()
