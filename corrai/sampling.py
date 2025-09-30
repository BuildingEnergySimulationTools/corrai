from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Union, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    max_error,
)

import plotly.figure_factory as ff

from scipy.stats.qmc import LatinHypercube
from SALib.sample import morris as morris_sampler
from SALib.sample import sobol as sobol_sampler
from SALib.sample import fast_sampler, latin

from corrai.base.utils import check_indicators_configs
from corrai.base.metrics import nmbe, cv_rmse
from corrai.base.parameter import Parameter
from corrai.base.model import Model
from corrai.base.math import aggregate_time_series
from corrai.base.simulate import run_simulations

SCORE_MAP = {
    "r2": r2_score,
    "nmbe": nmbe,
    "cv_rmse": cv_rmse,
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "max": max_error,
}


def plot_pcp(
    parameter_values: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    bounds: list[tuple[float, float]] | None = None,
    color_by: str | None = None,
    title: str | None = "Parallel Coordinates — Samples",
    html_file_path: str | None = None,
) -> go.Figure:
    """
    Creates a Parallel Coordinates Plot (PCP) for parameter samples and aggregated
    indicators. Each vertical axis corresponds to a parameter or an aggregated
    indicator, and each polyline represents one simulation.
    """

    if parameter_values.shape[0] != aggregated_results.shape[0]:
        raise ValueError(
            "Shape mismatch between parameter_values and aggregated_results"
        )

    df = pd.concat([parameter_values, aggregated_results], axis=1)

    if color_by is None:
        if not aggregated_results.empty:
            color_by = aggregated_results.columns[0]
        else:
            color_by = parameter_values.columns[0]

    dimensions = []
    for j, pname in enumerate(parameter_values.columns):
        dim = {"label": pname, "values": df[pname].to_numpy()}
        if bounds is not None:
            lb, ub = bounds[j]
            dim["range"] = [lb, ub]
        dimensions.append(dim)

    for col in aggregated_results.columns:
        col_vals = df[col].to_numpy()
        if np.all(np.isnan(col_vals)):
            dim = {"label": col, "values": col_vals}
        else:
            vmin = float(np.nanmin(col_vals))
            vmax = float(np.nanmax(col_vals))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                dim = {"label": col, "values": col_vals, "range": [vmin, vmax]}
            else:
                dim = {"label": col, "values": col_vals}
        dimensions.append(dim)

    line_kwargs = {}
    if color_by is not None and color_by in df.columns:
        line_kwargs = dict(color=df[color_by], colorscale="Viridis", showscale=True)

    fig = go.Figure(data=go.Parcoords(dimensions=dimensions, line=line_kwargs))
    fig.update_layout(title=title)

    if html_file_path:
        fig.write_html(html_file_path)

    return fig


@dataclass
class Sample:
    """
    Container for simulation samples and results.

    Each `Sample` instance stores parameter values and the corresponding
    simulation results. It supports indexing, aggregation, plotting, and
    integration with sampling strategies.

    Handle both dynamic and static models.

    Parameters
    ----------
    parameters : list of Parameter
        List of model parameters used to generate the samples.

    Attributes
    ----------
    parameters : list of Parameter
        Parameters associated with this sample.
    is_dynamic : Bool default True
        Specify if stored results are timeeries in a DataFrame for dynamic models
        or a Series of float for static models
    values : ndarray of shape (n_samples, n_parameters)
        Numerical values of the sampled parameters.
    results : Series of DataFrames
        Simulation results for each sample. Each element is typically a
        pandas DataFrame indexed by time, containing model outputs.
    """

    parameters: list[Parameter]
    is_dynamic: bool = True
    values: pd.DataFrame = field(init=False)
    results: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))

    def __post_init__(self):
        self.values = pd.DataFrame(columns=[par.name for par in self.parameters])

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice, list, np.ndarray)):
            return {"values": self.values.loc[idx, :], "results": self.results.loc[idx]}
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __setitem__(self, idx, item: dict):
        if "values" in item:
            self.values.loc[idx, :] = item["values"]
        if "results" in item:
            if isinstance(idx, int):
                self.results.at[idx] = item["results"]
            elif isinstance(idx, slice):
                self.results.loc[idx] = pd.Series(
                    item["results"], index=self.results.loc[idx].index
                )
            else:
                self.results.iloc[idx] = pd.Series(
                    item["results"], index=self.results.index[idx]
                )

        self._validate()

    def _validate(self):
        assert len(self.results) == len(
            self.values
        ), f"Mismatch: {len(self.values)} values vs {len(self.results)} results"

        if not self.values.index.equals(self.results.index):
            raise ValueError("Mismatch between values and results indices")

    def get_pending_index(self) -> np.ndarray:
        """
        Identify which samples have not yet been simulated.

        Returns
        -------
        ndarray of bool
            Boolean mask of length `len(self)`, where True
            indicates a sample without results.
        """
        return self.results.apply(lambda df: df.empty).values

    def get_parameters_intervals(self):
        """
        Return parameter intervals.

        Returns
        -------
        ndarray of shape (n_parameters, 2)
            Lower and upper bounds for each parameter.

        Raises
        ------
        NotImplementedError
            If any parameter has type 'Integer'.
        ValueError
            If parameters are not of type 'Real'.
        """
        if all(param.ptype == "Real" for param in self.parameters):
            return np.array([param.interval for param in self.parameters])
        elif any(param.ptype == "Integer" for param in self.parameters):
            raise NotImplementedError(
                "get_param_interval is not yet implemented for integer parameters"
            )
        else:
            raise ValueError("All parameter must have an ptype = 'Real'")

    def get_list_parameter_value_pairs(
        self, idx: int | list[int] | np.ndarray | slice = None
    ):
        """
        Map parameter objects to their sampled values.

        Parameters
        ----------
        idx : int, list of int, ndarray, or slice, optional
            Indices of samples to retrieve. Defaults to all.

        Returns
        -------
        list of list of (Parameter, value)
            Nested list where each inner list corresponds to a sample.
        """
        selected_values = self[idx]["values"]

        if isinstance(selected_values, pd.Series):
            selected_values = selected_values.to_frame().T

        return [
            [(par, val) for par, val in zip(self.parameters, row)]
            for row in selected_values.values
        ]

    def get_dimension_less_values(
        self, idx: int | list[int] | np.ndarray | slice = slice(None)
    ):
        """
        Normalize parameter values to [0, 1].

        Parameters
        ----------
        idx : int, list, ndarray, or slice, optional
            Indices of samples to normalize. Defaults to all.

        Returns
        -------
        ndarray of shape (n_selected, n_parameters)
            Dimensionless parameter values, scaled using
            their defined intervals.
        """
        values = self[idx]["values"]
        intervals = self.get_parameters_intervals()
        return (values - intervals[:, 0]) / (intervals[:, 1] - intervals[:, 0])

    def add_samples(
        self, values: np.ndarray, results: list[Union[pd.DataFrame, pd.Series]] = None
    ):
        """
        Add new samples and optionally their results.

        Parameters
        ----------
        values : ndarray of shape (n_samples, n_parameters)
            Sampled parameter values to add.
        results : list of DataFrame, optional
            Simulation results corresponding to `values`.
            If None, empty DataFrames are stored.

        Raises
        ------
        AssertionError
            If `results` length does not match `values` length.
        """
        n_samples, n_params = values.shape
        assert n_params == len(self.parameters), "Mismatch in number of parameters"

        new_df = pd.DataFrame(values, columns=self.values.columns)
        if self.values.empty:
            self.values = new_df
        else:
            self.values = pd.concat([self.values, new_df], ignore_index=True)

        if results is None:
            new_results = pd.Series([pd.DataFrame()] * n_samples, dtype=object)
        else:
            assert len(results) == n_samples, "Mismatch between values and results"
            new_results = pd.Series(results, dtype=object)

        self.results = pd.concat([self.results, new_results], ignore_index=True)

    def get_aggregated_time_series(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        prefix: str = "aggregated",
    ) -> pd.DataFrame:
        """
        Aggregate sample results using a specified statistical or error metric.

        This method extracts the specified `indicator` column, and aggregates
        the time series across simulations using the given method. If a reference
        time series is provided, metrics that require ground truth
        (e.g., mean_absolute_error) are supported.

        If `freq` is provided, the aggregation is done over time bins, producing a
        table of simulation runs versus time periods.

        Only works for dynamic models

        Parameters
        ----------
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
            Must have the same datetime index and length as the individual simulation
            results.

        freq : str or pandas.Timedelta or datetime.timedelta, optional
            If provided, aggregate the time series within bins of this frequency
            (e.g., "d" for daily, "h" for hourly).
            The result will be a DataFrame where each row corresponds to a simulation and
            each column to a time bin.

        prefix : str, default="aggregated"
            Prefix to use for naming the output column when `freq` is not specified.

        Returns
        -------
        pandas.DataFrame
            If `freq` is not provided, returns a one-column DataFrame containing the
            aggregated metric per simulation, indexed by the same index as `results`.

            If `freq` is provided, returns a DataFrame indexed by simulation IDs
            (same as `results.index`), with columns representing each aggregated time bin.

        Raises
        ------
        ValueError
            If the shapes of `results` and `reference_time_series` are incompatible.
            If the datetime index is not valid or missing.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np

        >>> from corrai.base.parameter import Parameter
        >>> from corrai.sampling import Sample

        >>> sample = Sample(
        ...     parameters=[
        ...         Parameter("a", interval=(1, 10)),
        ...         Parameter("b", interval=(1, 10)),
        ...     ]
        ... )

        >>> t = pd.date_range("2009-01-01", freq="h", periods=2)
        >>> res_1 = pd.DataFrame({"a": [1, 2]}, index=t)
        >>> res_2 = pd.DataFrame({"a": [3, 4]}, index=t)

        >>> sample.add_samples(np.array([[1, 2], [3, 4]]), [res_1, res_2])

        >>> # No frequency aggregation: one aggregated value per simulation
        >>> sample.get_aggregated_time_series("a")
           aggregated_a
        0           1.5
        1           3.5

        >>> # With frequency aggregation: one value per time bin per simulation
        >>> ref = pd.Series(
        ...     [1, 1], index=pd.date_range("2009-01-01", freq="h", periods=2)
        ... )

        >>> sample.get_aggregated_time_series(
        ...     indicator="a",
        ...     method="mean_absolute_error",
        ...     reference_time_series=ref,
        ...     freq="h",
        ...)

           2009-01-01 00:00:00  2009-01-01 01:00:00
        0                  0.0                  1.0
        1                  2.0                  3.0

        """
        if not self.is_dynamic:
            raise ValueError("Cannot perform time aggregation on static sample")

        return aggregate_time_series(
            self.results,
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq,
            prefix,
        )

    def get_static_results_as_df(self):
        if self.is_dynamic:
            raise ValueError("Cannot map results to a DataFrame with a dynamic Sample")
        return pd.DataFrame(self.results.to_list(), index=self.results.index)

    def get_score_df(
        self,
        indicator: str,
        reference_time_series: pd.Series,
        scoring_methods: list[str | Callable] = None,
        resample_rule: str | pd.Timedelta | dt.timedelta = None,
        agg_method: str = "mean",
    ) -> pd.DataFrame:
        """
        Compute scoring metrics for a given indicator across all sample results.

        This method evaluates the performance of dynamic model predictions by comparing
        them against a reference time series. It supports multiple scoring metrics
        (R², NMBE, CV(RMSE), MAE, RMSE, max error) and optional resampling of data.

        Parameters
        ----------
        indicator : str
            Name of the indicator/variable to evaluate from the simulation results.
            Must be a valid columns in the sample results DataFrame.
        reference_time_series : pd.Series
            Ground truth or measured time series data to compare against.
        scoring_methods : list of str or callable, optional
            List of scoring methods to apply. Can be:

            - String values from ``SCORE_MAP``: ``"r2"``, ``"nmbe"``, ``"cv_rmse"``,
              ``"mae"``, ``"rmse"``, ``"max"``
            - Custom callable functions with signature ``func(y_true, y_pred) -> float``

            If None, all methods are used.
            Default is None.
        resample_rule : str, pd.Timedelta or dt.timedelta, optional
            Resampling frequency for aggregating the time series data before scoring.
            Examples: ``"D"`` (daily), ``"h"`` (hourly), ``"ME"`` (month end).
            If None, no resampling is performed.
            Default is None.
        agg_method : str, optional
            Aggregation method to use when resampling. Common values include:
            ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"``.
            Default is ``"mean"``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing scoring metrics for each sample.

            - Index: sample identifiers from ``self.results``
            - Columns: metric names (e.g., ``"r2_score"``, ``"nmbe"``, ``"cv_rmse"``)
            - Values: computed metric values (float)

            The DataFrame's index name is set to the resampling rule or the inferred
            frequency of the reference time series.

        Raises
        ------
        NotImplementedError
            If the model is not dynamic (``self.is_dynamic == False``).

        Notes
        -----
        The scoring metrics available in ``SCORE_MAP`` are:

        - ``r2``: R² score (coefficient of determination)
        - ``nmbe``: Normalized Mean Bias Error
        - ``cv_rmse``: Coefficient of Variation of Root Mean Squared Error
        - ``mae``: Mean Absolute Error
        - ``rmse``: Root Mean Squared Error
        - ``max``: Maximum absolute error

        When resampling is applied, both the predicted and reference time series are
        resampled using the same rule and aggregation method to ensure alignment.

        Examples
        --------
        Basic usage with default metrics:

        >>> import pandas as pd
        >>> import numpy as np
        >>> # Assuming 'sample' is an instance of Sample class with results
        >>> reference = pd.Series(
        ...     np.random.randn(100),
        ...     index=pd.date_range("2023-01-01", periods=100, freq="h"),
        ... )
        >>> scores = sample.get_score_df(
        ...     indicator="temperature", reference_time_series=reference
        ... )
        >>> print(scores)
                    r2_score      nmbe   cv_rmse       mae      rmse       max
        0    0.85234  0.012345  0.234567  1.234567  1.567890  3.456789
        1    0.82156  0.023456  0.345678  1.345678  1.678901  3.567890
        ...

        Using specific metrics and daily resampling:

        >>> scores = sample.get_score_df(
        ...     indicator="Energy",
        ...     reference_time_series=reference,
        ...     scoring_methods=["r2", "rmse", "mae"],
        ...     resample_rule="D",
        ...     agg_method="sum",
        ... )
        >>> print(scores)
                  r2_score      rmse       mae
        D
        0  0.91234  12.34567  10.12345
        1  0.89123  13.45678  11.23456
        ...

        See Also
        --------
        sklearn.metrics.r2_score : R² metric implementation
        sklearn.metrics.mean_absolute_error : MAE metric implementation
        sklearn.metrics.root_mean_squared_error : RMSE metric implementation
        """

        if not self.is_dynamic:
            raise NotImplementedError(
                "get_score_df is not implemented for non dynamic models"
            )

        scores = pd.DataFrame()
        scoring_methods = (
            list(SCORE_MAP.keys()) if scoring_methods is None else scoring_methods
        )

        method_func = [
            SCORE_MAP[method] if isinstance(method, str) else method
            for method in scoring_methods
        ]

        for idx, sample_res in self.results.items():
            data = sample_res[indicator]
            if resample_rule:
                data = data.resample(resample_rule).agg(agg_method)
                reference_time_series = reference_time_series.resample(
                    resample_rule
                ).agg(agg_method)

            for method in method_func:
                scores.loc[idx, method.__name__] = method(reference_time_series, data)

        scores.index.name = (
            resample_rule
            if resample_rule
            else reference_time_series.index.freq
            if reference_time_series.index.freq
            else reference_time_series.index.inferred_freq
        )
        return scores

    def plot_hist(
        self,
        indicator: str,
        method: str = "mean",
        unit: str = "",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        bins: int = 30,
        colors: str = "orange",
        reference_value: int | float = None,
        reference_label: str = "Reference",
        show_rug: bool = False,
        title: str = None,
    ):
        """
        Plot histogram of aggregated results.

        Parameters
        ----------
        indicator : str
            Name of the indicator column to plot.
        method : str, default="mean"
            Aggregation method.
        unit : str, optional
            Unit of the indicator.
        agg_method_kwarg : dict, optional
            Additional kwargs for aggregation.
        reference_time_series : Series, optional
            Reference time series.
        bins : int, default=30
            Histogram number of bins.
        colors : str, default="orange"
            Color of the histogram.
        reference_value: int, float, optional
            Add a vertical dashed red line at reference value.
            May be used for comparison with an expected value
        reference_label: str, optional
            Label name for reference value line to be displayed in the legend.
            Default is "Reference"
        show_rug : bool, default=False
            If True, display rug plot below histogram.
        title : str, optional
            Custom title.

        Returns
        -------
        go.Figure
            Plotly histogram figure.
        """
        if self.is_dynamic:
            res = self.get_aggregated_time_series(
                indicator,
                method,
                agg_method_kwarg,
                reference_time_series,
                freq=None,
                prefix=method,
            )
            hist_label = f"{method} {indicator}"

        else:
            res = self.get_static_results_as_df()[[indicator]]
            hist_label = indicator

        fig = ff.create_distplot(
            [res.squeeze().to_numpy()],
            [hist_label],
            bin_size=(res.max() - res.min()) / bins,
            colors=[colors],
            show_rug=show_rug,
        )

        if reference_value is not None:
            counts, _ = np.histogram(res, bins=bins, density=True)

            fig.add_trace(
                go.Scatter(
                    x=[reference_value, reference_value],
                    y=[0, max(counts)],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name=reference_label,
                )
            )

            # Make sure it spans the full y-axis range
            fig.update_yaxes(range=[0, None])  # auto from 0 to max
        title = f"Sample distribution of {hist_label}" if title is None else title
        fig.update_layout(
            title=title,
            xaxis_title=f"{hist_label} {unit}",
        )
        return fig

    def plot_sample(
        self,
        indicator: str | None,
        reference_timeseries: pd.Series | None = None,
        title: str | None = None,
        y_label: str | None = None,
        x_label: str | None = None,
        alpha: float = 0.5,
        show_legends: bool = False,
        round_ndigits: int = 2,
        quantile_band: float = 0.75,
        type_graph: str = "area",
    ) -> go.Figure:
        """
        Plot simulation results with different visualization modes.

        This function allows visualization of multiple simulation samples,
        either as a scatter plot of all samples or as an aggregated area
        with min–max envelope, median, and quantile bands.

        Only works for dynamic models

        Parameters
        ----------
        indicator : str, optional
            Column name to extract if inner elements are DataFrames
            with multiple columns.
        reference_timeseries : pandas.Series, optional
            A reference time series to plot alongside simulations
            (e.g., measured data).
        title : str, optional
            Plot title.
        y_label : str, optional
            Label for the y-axis.
        x_label : str, optional
            Label for the x-axis.
        alpha : float, default=0.5
            Opacity for scatter markers when ``type_graph="scatter"``.
        show_legends : bool, default=False
            Whether to display legends for each individual sample trace
            when ``type_graph="scatter"``.
        round_ndigits : int, default=2
            Number of digits for rounding parameter values in legend strings.
        quantile_band : float, default=0.75
            Upper quantile to display when ``type_graph="area"``.
            Both ``(1 - quantile_band)`` and ``quantile_band`` are drawn
            as dotted lines, e.g. ``0.75`` → 25% and 75%.
        type_graph : {"area", "scatter"}, default="area"
            Visualization mode:
            - ``"scatter"`` : plot all samples individually as scatter markers.
            - ``"area"`` : plot aggregated area with min–max envelope,
              median line, and quantile bands.

        Examples
        --------
        >>> fig = plot_sample(results, reference_timeseries=ref)
        >>> fig.show()

        >>> fig = plot_sample(results, reference_timeseries=ref, type_graph="scatter")
        >>> fig.show()
        """
        if not self.is_dynamic:
            raise ValueError("Cannot plot_sample a static sample")

        if self.results.empty:
            raise ValueError("`results` is empty. Simulate samples first.")

        def _legend_for(i: int) -> str:
            if not show_legends:
                return "Simulations"
            parameter_names = [par.name for par in self.parameters]
            vals = self.values.loc[i, :].values
            return ", ".join(
                f"{n}: {round(v, round_ndigits)}" for n, v in zip(parameter_names, vals)
            )

        series_list = []
        for sample in self.results:
            if sample is None or sample.empty:
                continue
            series_list.append(sample[indicator])

        if not series_list and reference_timeseries is None:
            raise ValueError("No simulated data available to plot.")

        fig = go.Figure()
        if type_graph == "scatter":
            for i, s in enumerate(series_list):
                fig.add_trace(
                    go.Scattergl(
                        name=_legend_for(i),
                        mode="markers",
                        x=s.index,
                        y=s.to_numpy(),
                        marker=dict(color=f"rgba(135,135,135,{alpha})"),
                        showlegend=show_legends,
                    )
                )
            if reference_timeseries is not None:
                fig.add_trace(
                    go.Scattergl(
                        name="Reference",
                        mode="lines",
                        x=reference_timeseries.index,
                        y=reference_timeseries.to_numpy(),
                        line=dict(color="red", width=2),
                        showlegend=True,
                    )
                )

        elif type_graph == "area":
            df_all = pd.concat(series_list, axis=1) if series_list else None

            if df_all is not None:
                lower = df_all.min(axis=1)
                upper = df_all.max(axis=1)

                fig.add_trace(
                    go.Scatter(
                        x=upper.index,
                        y=upper.values,
                        line=dict(width=0),
                        mode="lines",
                        name="max",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=lower.index,
                        y=lower.values,
                        line=dict(width=0),
                        mode="lines",
                        fill="tonexty",
                        name="Area min - max",
                        fillcolor="rgba(255,165,0,0.4)",
                        showlegend=True,
                    )
                )

                median = df_all.median(axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=median.index,
                        y=median.values,
                        mode="lines",
                        line=dict(color="black"),
                        name="Median",
                        showlegend=True,
                    )
                )

                q_low = 1 - quantile_band
                q_high = quantile_band
                q1 = df_all.quantile(q_low, axis=1)
                q2 = df_all.quantile(q_high, axis=1)

                fig.add_trace(
                    go.Scatter(
                        x=q1.index,
                        y=q1.values,
                        mode="lines",
                        line=dict(color="black", dash="dot"),
                        name="Quantiles",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=q2.index,
                        y=q2.values,
                        mode="lines",
                        line=dict(color="black", dash="dot"),
                        name="Quantiles",
                        showlegend=True,
                    )
                )

            if reference_timeseries is not None:
                fig.add_trace(
                    go.Scatter(
                        name="Reference",
                        mode="lines",
                        x=reference_timeseries.index,
                        y=reference_timeseries.to_numpy(),
                        line=dict(color="red"),
                        showlegend=True,
                    )
                )

        else:
            raise ValueError("`type_graph` must be either 'area' or 'scatter'.")

        if title is None:
            title = f"Sample plot of {indicator} indicator"
        else:
            title = "Sample plot"

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=True,
            legend_traceorder="normal",
        )
        return fig

    def plot_pcp(
        self,
        indicators_configs: list[str]
        | list[tuple[str, str | Callable] | tuple[str, str | Callable, pd.Series]],
        color_by: str | None = None,
        title: str | None = "Parallel Coordinates — Samples",
        html_file_path: str | None = None,
    ) -> go.Figure:
        """
        This method produces an interactive PCP visualization that allows comparison
        of model parameters against aggregated indicators from simulation results.
        It supports both dynamic and static models.

        For dynamic models, the specified indicators are aggregated across time using
        the provided functions (e.g., "mean", "sum", error metrics). For static models,
        the indicators are taken directly from the stored results.

        Parameters
        ----------
        indicators_configs : list of str or list of tuple
            Configuration of indicators to include in the plot.

            - For dynamic models, each element must be a tuple of the form:
              ``(indicator_name, method)`` or
              ``(indicator_name, method, reference_series)``.

              Here:
                * `indicator_name` : str
                  Column name in the simulation results to aggregate.
                * `method` : str or Callable
                  Aggregation function or metric to apply.
                * `reference_series` : pandas.Series, optional
                  Reference time series required for error-based methods
                  (e.g., mean absolute error).

            - For static models, a simple list of indicator names (str) is sufficient.

        color_by : str, optional
            Name of a parameter or result column to use for coloring the PCP lines.
            If None, all lines are plotted in the same color.

        title : str, default="Parallel Coordinates — Samples"
            Title of the plot.

        html_file_path : str, optional
            If provided, saves the interactive plot as an HTML file at the specified
            path.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated parallel coordinates figure. The figure can be displayed
            interactively in a Jupyter notebook, web browser, or exported to HTML.

        Raises
        ------
        ValueError
            If the `indicators_configs` are incompatible with the model type
            (dynamic vs static).

        See Also
        --------
        get_aggregated_time_series :
            For details on supported aggregation methods and how indicator values
            are computed for dynamic models.
        """

        check_indicators_configs(self.is_dynamic, indicators_configs)

        if self.is_dynamic:
            results = pd.DataFrame()
            for config in indicators_configs:
                col, func, *extra = config
                results[f"{func}_{col}"] = self.get_aggregated_time_series(
                    col, func, reference_time_series=None if not extra else extra[0]
                )
        else:
            results = self.get_static_results_as_df()[indicators_configs]

        return plot_pcp(
            parameter_values=self.values,
            aggregated_results=results,
            bounds=self.get_parameters_intervals().tolist(),
            color_by=color_by,
            title=title,
            html_file_path=html_file_path,
        )


class Sampler(ABC):
    """
    Abstract base class for parameter samplers.

    A `Sampler` generates parameter sets according to a chosen
    sampling method and runs simulations of a given model.

    Parameters
    ----------
    parameters : list of Parameter
        List of parameters to be sampled.
    model : Model
        The model to simulate for each sample.
    simulation_options : dict, optional
        Options passed to the simulation (e.g., time range, timestep).

    Attributes
    ----------
    sample : Sample
        Container holding parameter values and simulation results.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
    ):
        self.simulation_options = (
            {} if simulation_options is None else simulation_options
        )
        self.model = model
        self.sample = Sample(parameters, is_dynamic=model.is_dynamic)

    @property
    def parameters(self):
        return self.sample.parameters

    @property
    def values(self):
        return self.sample.values

    @property
    def results(self):
        return self.sample.results

    @abstractmethod
    def add_sample(self, *args, **kwargs) -> np.ndarray:
        """
        Generate new samples and optionally run simulations.

        Must be implemented in subclasses.

        Returns
        -------
        ndarray
            The newly generated sample values.
        """
        pass

    def _post_draw_sample(
        self,
        new_sample,
        simulate=True,
        n_cpu: int = 1,
        sample_is_dimless: bool = False,
        simulation_kwargs: dict = None,
    ):
        if sample_is_dimless:
            intervals = self.sample.get_parameters_intervals()
            lower_bounds = intervals[:, 0]
            upper_bounds = intervals[:, 1]
            new_values = lower_bounds + new_sample * (upper_bounds - lower_bounds)
        else:
            new_values = new_sample

        self.sample.add_samples(new_values)

        if simulate:
            sample_starts = len(self.sample) - new_values.shape[0]
            sample_ends = len(self.sample)
            self.simulate_at(
                slice(sample_starts, sample_ends), n_cpu, simulation_kwargs
            )

    def append_sample_from_param_dict(
        self, param_dict: dict[str, int | float | str], simulation_kwargs=None
    ):
        """
        Add a new sample from a parameter dictionary and simulate it.

        This method appends a single sample to the existing sample set by providing
        parameter values as a dictionary. The keys must correspond to the ``name``
        property of the ``Parameter`` objects in the ``parameters``. After adding
        the sample, it automatically runs a simulation for the newly added parameters.

        Parameters
        ----------
        param_dict : dict of {str: int, float, or str}
            Dictionary mapping parameter names to their values. Keys must match the
            ``name`` attribute of ``Parameter`` objects in ``self.parameters``.
            All parameters must be present in the dictionary.

        simulation_kwargs : dict, optional
            Additional keyword arguments to pass to the simulation method.
            These can include simulation-specific options such as solver settings,
            output options, or other model-specific parameters.
            Default is None.

        Returns
        -------
        None
            The method modifies the sample in place by adding a new row to
            ``self.sample`` and stores the simulation results in ``self.results``.

        Raises
        ------
        AssertionError
            If any parameter name in ``param_dict`` is not found in
            ``self.values.columns``, indicating missing or misspelled parameter names.

        Notes
        -----
        - The newly added sample is assigned the index ``self.values.index[-1]``,
          which corresponds to the last index after appending.
        - This method is useful for manual parameter exploration, adding specific
          test cases, or iteratively building samples based on optimization results.

        Examples
        --------
        Add a single sample with specific parameter values:

        >>> from corrai.base.parameter import Parameter
        >>> from corrai.base.model import IshigamiDynamic
        >>> from corrai.sampling import LHSSampler
        >>>
        >>> # Define parameters
        >>> params = [
        ...     Parameter("par_x1", (-3.14, 3.14), model_property="x1"),
        ...     Parameter("par_x2", (-3.14, 3.14), model_property="x2"),
        ...     Parameter("par_x3", (-3.14, 3.14), model_property="x3"),
        ... ]
        >>>
        >>> # Create sample object
        >>> simulation_opts = {
        ...     "start": "2023-01-01 00:00:00",
        ...     "end": "2023-01-01 23:00:00",
        ...     "timestep": "h",
        ... }
        >>> sample = LHSSampler(
        ...     parameter_list=params,
        ...     model=IshigamiDynamic(),
        ...     simulation_options=simulation_opts,
        ... )
        >>>
        >>> # Add a specific sample
        >>> new_params = {"par_x1": 1.5, "par_x2": -0.5, "par_x3": 2.0}
        >>> sample.append_sample_from_param_dict(new_params)
        >>> print(sample.values.tail(1))
              par_x1  par_x2  par_x3
        0        1.5    -0.5     2.0

        See Also
        --------
        add_samples : Add multiple samples at once using a numpy array
        simulate_at : Simulate a specific sample by its index
        Parameter : Class defining model parameters with bounds and properties
        """

        assert all(
            val in self.values.columns for val in param_dict.keys()
        ), "Missing parameters in columns"

        self.sample.add_samples(
            np.array([[param_dict[val] for val in self.values.columns]])
        )
        self.simulate_at(idx=self.values.index[-1], simulation_kwargs=simulation_kwargs)

    def simulate_at(
        self,
        idx: int | list[int] | np.ndarray | slice = None,
        n_cpu: int = 1,
        simulation_kwargs=None,
    ):
        list_param_value_pairs = self.sample.get_list_parameter_value_pairs(idx)
        res = run_simulations(
            self.model,
            list_param_value_pairs,
            self.simulation_options,
            n_cpu,
            simulation_kwargs,
        )
        if isinstance(idx, int):
            self.sample[idx] = {"results": res[0]}
        else:
            self.sample[idx] = {"results": [r for r in res]}

    def simulate_pending(self, n_cpu: int = 1, simulation_kwargs: dict = None):
        unsimulated_idx = self.sample.get_pending_index()
        self.simulate_at(unsimulated_idx, n_cpu, simulation_kwargs)

    @wraps(Sample.plot_sample)
    def plot_sample(
        self,
        indicator: str | None,
        reference_timeseries: pd.Series | None = None,
        title: str | None = None,
        y_label: str | None = None,
        x_label: str | None = None,
        alpha: float = 0.5,
        show_legends: bool = False,
        round_ndigits: int = 2,
        quantile_band: float = 0.75,
        type_graph: str = "area",
    ) -> go.Figure:
        return self.sample.plot_sample(
            indicator=indicator,
            reference_timeseries=reference_timeseries,
            title=title,
            y_label=y_label,
            x_label=x_label,
            alpha=alpha,
            show_legends=show_legends,
            round_ndigits=round_ndigits,
            quantile_band=quantile_band,
            type_graph=type_graph,
        )

    @wraps(Sample.get_aggregated_time_series)
    def get_sample_aggregated_time_series(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        prefix: str = "aggregated",
    ):
        return self.sample.get_aggregated_time_series(
            indicator, method, agg_method_kwarg, reference_time_series, freq, prefix
        )

    @wraps(Sample.plot_pcp)
    def plot_pcp(
        self,
        indicators_configs: list[str]
        | list[tuple[str, str | Callable] | tuple[str, str | Callable, pd.Series]],
        color_by: str | None = None,
        title: str | None = "Parallel Coordinates — Samples",
        html_file_path: str | None = None,
    ) -> go.Figure:
        return self.sample.plot_pcp(
            indicators_configs=indicators_configs,
            color_by=color_by,
            title=title,
            html_file_path=html_file_path,
        )


class RealSampler(Sampler, ABC):
    """
    Abstract base class for samplers that only support real-valued parameters.

    Provides utilities for interoperability with SALib.

    Parameters
    ----------
    parameters : list of Parameter
        Parameters to sample. All must have `ptype='Real'`.
    model : Model
        Model to simulate.
    simulation_options : dict, optional
        Simulation options to pass to the model.

    Raises
    ------
    ValueError
        If any parameter is not of type 'Real'.

    Methods
    -------
    get_salib_problem()
        Returns a SALib-compatible problem definition.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
    ):
        super().__init__(parameters, model, simulation_options)

        bad_params = [
            (par.name, par.ptype) for par in parameters if par.ptype != "Real"
        ]
        if bad_params:
            raise ValueError(
                f"All parameters must have a ptype 'Real'. Found {bad_params}"
            )

    def get_salib_problem(self):
        return {
            "num_vars": len(self.parameters),
            "names": [p.name for p in self.parameters],
            "bounds": [p.interval for p in self.parameters],
        }


class MorrisSampler(RealSampler):
    """
    Elementary Effects (Morris) sampler.

    Uses SALib's Morris method to generate trajectories and samples.

    Parameters
    ----------
    parameters : list of Parameter
        Real-valued parameters to sample.
    model : Model
        Model to simulate.
    simulation_options : dict, optional
        Simulation options for the model.

    Methods
    -------
    add_sample(N, num_levels=4, simulate=True, n_cpu=1, **kwargs)
        Generate samples using the Morris method.
    """

    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)
        self._dimless_values = None

    def add_sample(
        self,
        N: int,
        num_levels: int = 4,
        simulate: bool = True,
        n_cpu: int = 1,
        **morris_kwargs,
    ):
        morris_sample = morris_sampler.sample(
            self.get_salib_problem(), N, num_levels, **morris_kwargs
        )
        self._post_draw_sample(morris_sample, simulate, n_cpu)


class FASTSampler(RealSampler):
    """
    FAST sampler.

    Uses the Fourier Amplitude Sensitivity Test (FAST) to generate samples.

    Parameters
    ----------
    parameters : list of Parameter
        Real-valued parameters.
    model : Model
        Model to simulate.
    simulation_options : dict, optional
        Options for simulation.

    Methods
    -------
    add_sample(N, M=4, simulate=True, n_cpu=1, **kwargs)
        Generate samples using FAST.
    """

    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)

    def add_sample(
        self,
        N: int,
        M: int = 4,
        simulate: bool = True,
        n_cpu: int = 1,
        **fast_kwargs,
    ):
        fast_sample = fast_sampler.sample(self.get_salib_problem(), N, M, **fast_kwargs)
        self._post_draw_sample(fast_sample, simulate, n_cpu)


class RBDFASTSampler(RealSampler):
    """
    RBD-FAST sampler.

    Generates samples for the Random Balance Designs Fourier Amplitude
    Sensitivity Test (RBD-FAST).

    Parameters
    ----------
    parameters : list of Parameter
        Real-valued parameters.
    model : Model
        Model to simulate.
    simulation_options : dict, optional
        Options for simulation.

    Methods
    -------
    add_sample(N, simulate=True, n_cpu=1, **kwargs)
        Generate samples using RBD-FAST.
    """

    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)

    def add_sample(
        self,
        N: int,
        simulate: bool = True,
        n_cpu: int = 1,
        **rbdfast_kwargs,
    ):
        rbdfast_sample = latin.sample(self.get_salib_problem(), N, **rbdfast_kwargs)
        self._post_draw_sample(rbdfast_sample, simulate, n_cpu)


class LHSSampler(RealSampler):
    """
    Latin Hypercube sampler.

    Uses `scipy.stats.qmc.LatinHypercube` to generate stratified
    samples in the unit hypercube.

    Parameters
    ----------
    parameters : list of Parameter
        Real-valued parameters.
    model : Model
        Model to simulate.
    simulation_options : dict, optional
        Options for simulation.

    Methods
    -------
    add_sample(n, rng=None, simulate=True, **kwargs)
        Generate `n` samples using LHS.
    """

    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)

    def add_sample(
        self, n: int, rng: int = None, simulate=True, n_cpu: int = 1, **lhs_kwargs
    ):
        lhs = LatinHypercube(d=len(self.parameters), rng=rng, **lhs_kwargs)
        new_dimless_sample = lhs.random(n=n)
        self._post_draw_sample(
            new_dimless_sample, simulate, n_cpu, sample_is_dimless=True
        )


class SobolSampler(RealSampler):
    """
    Sobol sequence sampler.

    Generates low-discrepancy quasi-random samples using SALib's
    Sobol generator.

    Parameters
    ----------
    parameters : list of Parameter
        Real-valued parameters.
    model : Model
        Model to simulate.
    simulation_options : dict, optional
        Options for simulation.

    Methods
    -------
    add_sample(N, simulate=True, n_cpu=1, scramble=True,
               calc_second_order=True, **kwargs)
        Generate Sobol samples.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
    ):
        super().__init__(parameters, model, simulation_options)

    def add_sample(
        self,
        N: int,
        simulate: bool = True,
        n_cpu: int = 1,
        scramble: bool = True,
        *,
        calc_second_order: bool = True,
        **sobol_kwargs,
    ):
        new_sample = sobol_sampler.sample(
            problem=self.get_salib_problem(),
            scramble=scramble,
            N=N,
            calc_second_order=calc_second_order,
            **sobol_kwargs,
        )
        self._post_draw_sample(new_sample, simulate, n_cpu, sample_is_dimless=False)
