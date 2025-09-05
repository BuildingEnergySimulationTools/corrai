from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt

import plotly.figure_factory as ff

from scipy.stats.qmc import LatinHypercube
from SALib.sample import morris as morris_sampler
from SALib.sample import sobol as sobol_sampler
from SALib.sample import fast_sampler, latin

from corrai.base.parameter import Parameter
from corrai.base.model import Model
from corrai.base.math import aggregate_time_series
from corrai.base.simulate import run_simulations


def plot_pcp(
    parameter_values: np.ndarray,
    parameter_names: list[str],
    aggregated_results: pd.DataFrame,
    *,
    bounds: list[tuple[float, float]] | None = None,
    color_by: str | None = None,
    title: str | None = "Parallel Coordinates — Samples",
    html_file_path: str | None = None,
) -> go.Figure:
    """
    Creates a Parallel Coordinates Plot (PCP) for parameter samples and aggregated indicators.
    Each vertical axis corresponds to a parameter or an aggregated indicator,
    and each polyline represents one simulation.
    """

    if parameter_values.shape[0] != len(aggregated_results):
        raise ValueError("Mismatch between number of samples and aggregated results.")
    if len(parameter_names) != parameter_values.shape[1]:
        raise ValueError(
            "`parameter_names` length must match parameter_values.shape[1]."
        )

    df = pd.DataFrame(
        parameter_values, columns=parameter_names, index=aggregated_results.index
    )
    df = pd.concat([df, aggregated_results], axis=1)

    if color_by is None:
        if not aggregated_results.empty:
            color_by = aggregated_results.columns[0]
        else:
            color_by = parameter_names[0]

    dimensions = []
    for j, pname in enumerate(parameter_names):
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

    Parameters
    ----------
    parameters : list of Parameter
        List of model parameters used to generate the samples.

    Attributes
    ----------
    parameters : list of Parameter
        Parameters associated with this sample.
    values : ndarray of shape (n_samples, n_parameters)
        Numerical values of the sampled parameters.
    results : Series of DataFrames
        Simulation results for each sample. Each element is typically a
        pandas DataFrame indexed by time, containing model outputs.
    """

    parameters: list[Parameter]
    values: np.ndarray = field(init=False)
    results: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))

    def __post_init__(self):
        self.values = np.empty((0, len(self.parameters)))

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice, list, np.ndarray)):
            return {"values": self.values[idx], "results": self.results[idx]}
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __setitem__(self, idx, item: dict):
        if "values" in item:
            self.values[idx] = item["values"]
        if "results" in item:
            if isinstance(idx, int):
                self.results.at[idx] = item["results"]
            else:
                self.results.iloc[idx] = pd.Series(
                    item["results"], index=self.results.index[idx]
                )

        self._validate()

    def _validate(self):
        assert len(self.results) == len(
            self.values
        ), f"Mismatch: {len(self.values)} values vs {len(self.results)} results"

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

        if selected_values.ndim == 1:
            selected_values = selected_values[np.newaxis, :]

        return [
            [(par, val) for par, val in zip(self.parameters, row)]
            for row in selected_values
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

    def add_samples(self, values: np.ndarray, results: list[pd.DataFrame] = None):
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

        self.values = np.vstack([self.values, values])

        if results is None:
            new_results = pd.Series([pd.DataFrame()] * n_samples, dtype=object)
        else:
            assert len(results) == n_samples, "Mismatch between values and results"
            new_results = pd.Series(results, dtype=object)

        self.results = pd.concat([self.results, new_results], ignore_index=True)

    @wraps(aggregate_time_series)
    def get_aggregated_time_series(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        prefix: str = "aggregated",
    ) -> pd.DataFrame:
        return aggregate_time_series(
            self.results,
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq,
            prefix,
        )

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
        res = self.get_aggregated_time_series(
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq=None,
            prefix=method,
        )

        fig = ff.create_distplot(
            [res.squeeze().to_numpy()],
            [f"{method}_{indicator}"],
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
        title = (
            f"Sample distribution of {method} {indicator}" if title is None else title
        )
        fig.update_layout(
            title=title,
            xaxis_title=f"{method} {indicator} {unit}",
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

        if self.results.empty:
            raise ValueError("`results` is empty. Simulate samples first.")

        def _legend_for(i: int) -> str:
            if not show_legends:
                return "Simulations"
            parameter_names = [par.name for par in self.parameters]
            vals = self.values[i, :]
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
        self.sample = Sample(parameters)

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

    def plot_pcp(
        self,
        indicator: str | None = None,
        method: str | list[str] = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        prefix: str | None = None,
        bounds: list[tuple[float, float]] | None = None,
        color_by: str | None = None,
        title: str | None = "Parallel Coordinates — Samples",
        html_file_path: str | None = None,
    ) -> go.Figure:
        if indicator is None:
            aggregated = pd.DataFrame(index=range(len(self.values)))
        else:
            methods = [method] if isinstance(method, str) else method
            dfs = []
            for m in methods:
                this_prefix = prefix if prefix is not None else m
                agg = aggregate_time_series(
                    results=self.results,
                    indicator=indicator,
                    method=m,
                    agg_method_kwarg=agg_method_kwarg,
                    reference_time_series=reference_time_series,
                    freq=freq,
                    prefix=this_prefix,
                )
                if agg is not None and not agg.empty:
                    dfs.append(agg)
            aggregated = (
                pd.concat(dfs, axis=1)
                if dfs
                else pd.DataFrame(index=range(len(self.values)))
            )

        return plot_pcp(
            parameter_values=self.values,
            parameter_names=[p.name for p in self.parameters],
            aggregated_results=aggregated,
            bounds=bounds,
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

    def add_sample(self, n: int, rng: int = None, simulate=True, **lhs_kwargs):
        lhs = LatinHypercube(d=len(self.parameters), rng=rng, **lhs_kwargs)
        new_dimless_sample = lhs.random(n=n)
        self._post_draw_sample(new_dimless_sample, simulate, sample_is_dimless=True)


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
