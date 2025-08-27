import datetime as dt
import itertools

import numpy as np
import pandas as pd
from plotly import colors as colors, graph_objects as go
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

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

    # Add second diff, because in some case (system), measured decrease
    # has a maximum. eg for some reason AHU measured flowrate
    # cannot decrease by more than Xm3/h per minutes
    df = df.diff().fillna(0)
    df.loc[df[data_col_name] < 0] = 0
    df["start_day"] = pd.DatetimeIndex(list(map(lambda x: x.date(), df.index)))
    if df.index.tz:
        df["start_day"] = pd.DatetimeIndex(list(df["start_day"])).tz_localize(
            tz=df.index.tz
        )
    df["hour_since_beg_day"] = (df.index - df.start_day).dt.total_seconds() / 60 / 60

    max_index = find_peaks(df[data_col_name].to_numpy())[0]
    max_values = df[data_col_name].iloc[max_index]
    filt_max_values_index = max_values.loc[max_values > diff_filter_threshold].index

    df.loc[~df.index.isin(filt_max_values_index), "hour_since_beg_day"] = np.nan

    return float_to_hour(list(df.hour_since_beg_day.dropna()))


class KdeSetPoint(BaseEstimator, ClusterMixin):
    """
    A KDE-based set point detection and clustering transformer.

    This estimator uses Kernel Density Estimation (KDE) to identify statistically
    significant "set points" in a univariate dataset. These set points correspond
    to local maxima in the estimated probability density function. After detecting
    the set points, the algorithm assigns labels to new data points based on their
    proximity to the nearest set point.

    Parameters
    ----------
    bandwidth : float, default=0.1
        Bandwidth parameter for the kernel used in KDE. Controls the smoothness
        of the estimated density.

    domain_tol : float, default=0.2
        Relative tolerance for extending the domain over which the KDE is evaluated.
        The domain is computed by expanding the min/max range of the data by this
        factor times the mean absolute value.

    domain_n_sample : int, default=1000
        Number of evenly spaced samples over the domain used to evaluate the KDE.

    lik_filter : float, default=1
        Minimum likelihood required for a peak to be accepted as a set point.

    cluster_tol : float, default=0.05
        Tolerance used when assigning labels. A point is assigned to the nearest
        set point if the distance is below this value. Otherwise, it's labeled -1.

    Attributes
    ----------
    kde : sklearn.neighbors.KernelDensity
        The KDE estimator used to fit the probability density function.

    domain : ndarray of shape (n_samples,)
        The domain over which the KDE is evaluated.

    set_points_ : ndarray of shape (n_set_points,)
        The set points identified as local maxima of the density function.

    set_points_likelihood_ : ndarray of shape (n_set_points,)
        The likelihood values of the accepted set points.

    labels_ : ndarray of shape (n_samples,)
        The cluster labels assigned to the input data during `fit`.

    Methods
    -------
    fit(X, y=None)
        Fits the KDE model to the input data and identifies set points.

    predict(X)
        Assigns each input sample to the nearest set point (cluster).

    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.learning.cluster import KdeSetPoint
    >>> from corrai.learning.cluster import plot_kde_hist, plot_kde_predict

    >>> data = pd.DataFrame(
    ...     {"data": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]}
    ... )
    >>> kde_setpoint = KdeSetPoint(bandwidth=0.1, lik_filter=0.14)

    >>> kde_setpoint.fit_predict(data)

    >>> print(kde_setpoint.set_points_)
    [-1.0010010e-04  9.9885441e-01  2.0001001e+00]

    >>> print(kde_setpoint.labels_)
    [0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 2. 2. 2. 2.]

    Notes
    -----
    - This transformer only supports 1D data (i.e., a single column).
    - Clustering is performed based on the Euclidean distance to detected set points.
    - Points that do not fall within `cluster_tol` of any set point are assigned -1.
    - Useful for unsupervised clustering of time series values, especially for set point detection
      in building operation data (e.g., thermostat settings, power modes).
    """

    def __init__(
        self,
        bandwidth: float = 0.1,
        domain_tol: float = 0.2,
        domain_n_sample: int = 1000,
        lik_filter: float = 1.0,
        cluster_tol: float = 0.05,
    ):
        self.bandwidth = bandwidth
        self.domain_tol = domain_tol
        self.domain_n_sample = domain_n_sample
        self.lik_filter = lik_filter
        self.cluster_tol = cluster_tol
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.domain = None

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
        max_index = find_peaks(like_domain)
        self.set_points_ = self.domain[max_index[0]]

        # Pass if perfectly flat (all values equal in the domain)
        if self.set_points_.shape[0] > 0:
            set_points_likelihood = np.exp(self.kde.score_samples(self.set_points_))
            like_mask = set_points_likelihood > self.lik_filter

            self.set_points_ = self.set_points_[like_mask].flatten()
            self.set_points_likelihood_ = set_points_likelihood[like_mask]
            self.labels_ = self.predict(X)

        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["set_points_", "set_points_likelihood_"])

        X = as_1_column_dataframe(X)
        X = X.to_numpy()
        X = _reshape_1d(X)

        x_cluster = np.empty(X.shape[0])
        x_cluster[:] = np.nan
        for nb, sp in enumerate(self.set_points_):
            mask = np.logical_and(
                (X > (sp - self.cluster_tol)), (X < (sp + self.cluster_tol))
            )
            x_cluster[mask] = nb

        return np.nan_to_num(x_cluster, nan=-1)


def set_point_identifier(X: pd.DataFrame | pd.Series, estimator: KdeSetPoint):
    """
    Identifies set points in a time series data using kernel density estimation.
    Uses CorrAI KdeSetPointIdentificator combined with a StandardScaler to scale
    the data

    Parameters
    ----------
    X : pandas.DataFrame
        The input data containing time series data to identify set points in. The data
        must have a DateTimeIndex.
    estimator : object, optional
        A corrai KdeSetPointIdentificator. Default is None, which will use a
        `KdeSetPointIdentificator` object with default parameters values.

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

    model = make_pipeline(StandardScaler().set_output(transform="pandas"), estimator)

    pd_scaler = model.named_steps["standardscaler"]
    kde = model.named_steps["kdesetpoint"]

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
            set_points = pd_scaler.inverse_transform([kde.set_points_])
        except ValueError:
            set_points = None

        if set_points is not None:
            index_tuple = [
                (period, f"set_point_{i}") for i in range(set_points.shape[1])
            ]
            multiindex = pd.MultiIndex.from_tuples(index_tuple)
            set_points_series = pd.Series(
                set_points.flatten(), index=multiindex, name=col
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
    estimator: KdeSetPoint,
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

    estimator : KdeSetPoint
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


def plot_kde_predict(
    X: pd.DataFrame | pd.Series,
    bandwidth: float = 0.1,
    domain_tol: float = 0.2,
    domain_n_sample: int = 1000,
    lik_filter: float = 1.0,
    cluster_tol: float = 0.05,
    estimator: KdeSetPoint = None,
    title="Clustered Timeseries",
    y_label="[-]",
):
    """
    Plot a clustered time series using KDE-based setpoint identification.

    This function visualizes the time series data along with the output of a KDE-based
    setpoint identification algorithm. Each cluster (setpoint or transient region)
    is represented with a different color in the scatter plot.

    Parameters
    ----------
    X : pandas.DataFrame or pandas.Series
        Time series input data of shape (n_samples, 1). Must have a DateTimeIndex.
    bandwidth : float, default=0.1
        Bandwidth of the kernel used in the KDE estimation.
    domain_tol : float, default=0.2
        Tolerance to merge nearby domain samples when identifying setpoints.
    domain_n_sample : int, default=1000
        Number of points to evaluate in the KDE domain.
    lik_filter : float, default=1.0
        Minimum likelihood threshold for considering a domain sample as a valid setpoint.
    cluster_tol : float, default=0.05
        Tolerance for merging close likelihood peaks into a single setpoint.
    estimator : KdeSetPoint, optional
        Pre-configured instance of the KDE-based estimator. If None, a new instance is created
        with the specified parameters.
    title : str, default="Clustered Timeseries"
        Title of the generated plot.
    y_label : str, default="[-]"
        Label for the y-axis of the plot.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object showing the clustered time series.

    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.learning.cluster import plot_kde_predict

    >>> # Create an example time series with some steady-state regions
    >>> data = pd.DataFrame(
    ...     {"data": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]},
    ...     index=pd.date_range("2009", periods=18, freq="h"),
    ... )

    >>> # Plot clustered data using default KDE parameters
    >>> fig = plot_kde_predict(data, bandwidth=0.1, lik_filter=0.14)
    >>> fig.show()

    Notes
    -----
    - The input series must contain only one column and a DateTimeIndex.
    - The function uses Plotly for visualization and colors each cluster distinctly.
    - The first cluster (usually the transient or unclassified region) is shown first in the legend.
    - The `KdeSetPoint` estimator is part of the corrai.learning module and encapsulates
      the KDE and clustering logic.

    See Also
    --------
    corrai.learning.KdeSetPoint : The KDE-based estimator used to classify setpoints.
    """
    X = as_1_column_dataframe(X)

    if estimator is None:
        estimator = KdeSetPoint(
            bandwidth, domain_tol, domain_n_sample, lik_filter, cluster_tol
        )

    cluster_col = estimator.fit_predict(X)

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

    return fig


def plot_kde_hist(
    X: pd.DataFrame | pd.Series,
    bandwidth: float = 0.1,
    domain_tol: float = 0.2,
    domain_n_sample: int = 1000,
    xbins: int = 100,
    title="Clustered Timeseries",
    x_label="",
):
    """
    Plot a histogram and KDE likelihood curve of a univariate time series.

    This function overlays a histogram of the input time series with the likelihood
    function estimated using kernel density estimation (KDE). It is useful for
    visualizing the distribution of the data, detecting likely setpoints, selecting
    parameters for KdeSetpoint.

    Parameters
    ----------
    X : pandas.DataFrame or pandas.Series
        Input time series data of shape (n_samples, 1). Must have a DateTimeIndex.

    bandwidth : float, default=0.1
        Bandwidth of the kernel used in the KDE estimation.

    domain_tol : float, default=0.2
        Tolerance used to extend the KDE domain beyond the min/max of the data.

    domain_n_sample : int, default=1000
        Number of points in the domain over which the KDE likelihood is evaluated.

    xbins : int, default=100
        Number of bins to use for the histogram.

    title : str, default="Clustered Timeseries"
        Title for the plot.

    x_label : str, default=""
        Label for the x-axis.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object showing the histogram and KDE likelihood curve.

    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.learning.cluster import plot_kde_hist

    >>> data = pd.DataFrame(
    ...     {"data": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]},
    ...     index=pd.date_range("2009", periods=18, freq="h"),
    ... )

    >>> fig = plot_kde_hist(data)
    >>> fig.show()

    Notes
    -----
    - This function uses `sklearn.neighbors.KernelDensity` to estimate the density
      function.
    - The KDE domain is extended around the min and max of the data by `domain_tol`
      times the average absolute value of the series.
    - The histogram is normalized to form a probability density, making it directly
      comparable to the KDE likelihood.
    - This plot is helpful to visually identify potential steady states (setpoints)
      and to configure the class KdeSetpoint.
    """

    X = as_1_column_dataframe(X)

    tolerance = np.mean(np.abs(X), axis=0) * domain_tol

    domain = np.linspace(
        np.min(X, axis=0) - tolerance,
        np.max(X, axis=0) + tolerance,
        domain_n_sample,
    )

    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X.to_numpy())

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

    return fig
