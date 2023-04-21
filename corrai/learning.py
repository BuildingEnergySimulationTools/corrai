import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema

import itertools
import datetime as dt

import plotly.colors as colors
import plotly.graph_objects as go

import corrai.custom_transformers as ct
from sklearn.pipeline import make_pipeline


def _reshape_2d_df(sample):
    if isinstance(sample, pd.DataFrame):
        return sample
    if isinstance(sample, pd.Series):
        return sample.to_frame()
    elif isinstance(sample, np.ndarray) and sample.ndim == 1:
        x = np.reshape(sample, (-1, 1))
        return pd.DataFrame(x)


def _reshape_1d(sample):
    if isinstance(sample, pd.DataFrame):
        return sample.squeeze()
    elif isinstance(sample, np.ndarray):
        return sample.flatten()


def _2d_n_1_dataframer(X):
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

    if not isinstance(X, (list, np.ndarray, pd.DataFrame)):
        raise ValueError(
            f"X must be one of {list, np.array, pd.DataFrame}, " f"got {type(X)}"
        )

    if isinstance(X, list):
        X = pd.DataFrame(np.array(X))

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if X.shape[1] > 1:
        raise ValueError(
            f"X has {X.shape[1]} columns "
            f"KdeSetPointIdentificator only fits 1 times series at a time"
        )

    return X


def get_hours_switch(timeseries, diff_filter_threshold=0, switch="positive"):
    """
    From a time series determine the number of hour since the beginning
    of the day when the signal rises (switch='positive') or decreases
    (switch='negative') or both (switch='both').
    diff_filter_threshold is used to filter small variations

    :param timeseries: pandas Series with DatetimeIndex
    :param diff_filter_threshold: float or integer
    :param switch: 'positive', 'negative', 'both
    :return: pandas Series
    """
    if not isinstance(timeseries, pd.Series):
        raise ValueError("timeseries must be a Pandas DataFrame object")
    if not isinstance(timeseries.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a Pandas DatetimeIndex")

    df = timeseries.dropna().copy().to_frame().diff()
    data_col_name = timeseries.name
    if switch == "positive":
        df = df[df > 0]
    elif switch == "negative":
        df = abs(df[df < 0])
    elif switch == "both":
        df = abs(df)
    else:
        raise ValueError(f"Unknown value {switch} fo switch argument")

    df["start_day"] = pd.DatetimeIndex(list(map(lambda x: x.date(), df.index)))
    if df.index.tz:
        df["start_day"] = pd.DatetimeIndex(list(df["start_day"])).tz_localize(
            tz=df.index.tz
        )
    df["hour_since_beg_day"] = (df.index - df.start_day).dt.total_seconds() / 60 / 60
    df.loc[~(df[data_col_name] > diff_filter_threshold), "hour_since_beg_day"] = np.nan

    return df.hour_since_beg_day.dropna()


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
        X = _2d_n_1_dataframer(X)

        tolerance = np.mean(np.abs(X), axis=0) * self.domain_tol

        self.domain = np.linspace(
            np.min(X, axis=0) - tolerance,
            np.max(X, axis=0) + tolerance,
            self.domain_n_sample,
        )

        self.kde.fit(X.to_numpy())

        func_sample = self.kde.score_samples(self.domain)
        max_index = argrelextrema(func_sample, np.greater)
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
        X = _2d_n_1_dataframer(X)
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


def set_point_identifier(X, estimator=None, sk_scaler=None, cols=None):
    """
    Identifies set points in a time series data using kernel density estimation.

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
    to each column of the input data, and then finding the peaks of the density estimate,
    which shall correspond to the set points.
    """
    X = _reshape_2d_df(X)

    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X index must be a DateTimeIndex")

    if cols is None:
        cols = X.columns

    if sk_scaler is None:
        pd_scaler = ct.PdSkTransformer(StandardScaler())
    else:
        pd_scaler = ct.PdSkTransformer(sk_scaler)

    if estimator is None:
        estimator = KdeSetPointIdentificator()

    model = make_pipeline(pd_scaler, estimator)
    pd_scaler = model.named_steps["pdsktransformer"]
    kde = model.named_steps["kdesetpointidentificator"]

    duration = X.index[-1] - X.index[-0]

    period = pd.Period(
        X.index[0].strftime("%Y-%m-%d"),
        day=duration.days,
        second=duration.seconds,
    )

    multi_series_list = []
    for col in cols:
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
    X, window_size, slide_size, estimator=None, cols=None
):
    if estimator is None:
        estimator = KdeSetPointIdentificator()
    if not isinstance(estimator, KdeSetPointIdentificator):
        raise ValueError(
            f"estimator must be a corrai KdeSetPointIdentificator object "
            f"got {type(estimator)}"
        )
    if not isinstance(window_size, dt.timedelta):
        raise ValueError("window_size must be a datetime.timedelta object")
    if not isinstance(slide_size, dt.timedelta):
        raise ValueError("window_size must be a datetime.timedelta object")
    if not isinstance(X, pd.DataFrame):
        raise ValueError("data must be a Pandas DataFrame")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("data index be a Pandas DateTimeIndex")

    if cols is None:
        cols = X.columns

    start_date = X.index[0]
    end_date = X.index[-1]

    groups_res_list = []

    while start_date <= end_date - window_size:
        selected_data = X.loc[start_date : start_date + window_size, cols]
        groups_res_list.append(
            set_point_identifier(X=selected_data, estimator=estimator)
        )

        start_date += slide_size

    if not all(v is None for v in groups_res_list):
        return pd.concat(groups_res_list)
    else:
        return None


def plot_kde_set_point(
    X, estimator, title="Clustered Timeseries", y_label="[-]", fit=False
):
    """
    Plots a scatter plot of a time series with markers colored by cluster,
    along with an annotation in the upper right corner showing the different
    cluster colors.

     Args:
         estimator: A clustering estimator that has a `fit_predict` or
         `predict` method. This estimator is used to assign cluster labels to
          the data.

         X: A pandas Series or DataFrame with a time index and numeric values.
          If X is a DataFrame, each column is treated as a separate time series
          and plotted separately.

         title: (optional) The title of the plot.

         y_label: (optional) The label of the y-axis.

         fit: (optional) Whether to call the `fit_predict` method of the
          estimator instead of the `predict` method.

     Returns:
         None (the plot is displayed using `fig.show()`).
    """
    X = _2d_n_1_dataframer(X)

    if fit:
        cluster_col = estimator.fit_predict(X)
    else:
        cluster_col = estimator.predict(X)

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


def plot_ts_kde(
    X, title="Likelihood and data", x_label="", scaled=True, bandwidth=0.1, xbins=100
):
    X = _2d_n_1_dataframer(X)

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
