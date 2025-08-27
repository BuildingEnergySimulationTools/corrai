from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt

import plotly.figure_factory as ff

from scipy.stats.qmc import LatinHypercube
from SALib.sample import morris as morris_sampler
from SALib.sample import sobol as sobol_sampler
from SALib.sample import fast_sampler, latin
from SALib.sample import saltelli

from corrai.base.parameter import Parameter
from corrai.base.model import Model
from corrai.base.math import aggregate_time_series
from corrai.base.simulate import run_simulations


@dataclass
class Sample:
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
                self.results.iloc[idx] = item["results"]
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
        return self.results.apply(lambda df: df.empty).values

    def get_parameters_intervals(self):
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
        values = self[idx]["values"]
        intervals = self.get_parameters_intervals()
        return (values - intervals[:, 0]) / (intervals[:, 1] - intervals[:, 0])

    def add_samples(self, values: np.ndarray, results: list[pd.DataFrame] = None):
        n_samples, n_params = values.shape
        assert n_params == len(self.parameters), "Mismatch in number of parameters"

        self.values = np.vstack([self.values, values])

        if results is None:
            new_results = pd.Series([pd.DataFrame()] * n_samples, dtype=object)
        else:
            assert len(results) == n_samples, "Mismatch between values and results"
            new_results = pd.Series(results, dtype=object)

        self.results = pd.concat([self.results, new_results], ignore_index=True)

    def get_aggregate_time_series(
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
        bin_size: float = 1.0,
        colors: str = "orange",
        show_rug: bool = False,
        title: str = None,
    ):
        res = self.get_aggregate_time_series(
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
            bin_size=bin_size,
            colors=[colors],
            show_rug=show_rug,
        )

        title = (
            f"Sample distribution of {method} {indicator}" if title is None else title
        )
        fig.update_layout(
            title=title,
            xaxis_title=f"{method} {indicator} {unit}",
        )
        return fig

    def plot(
        self,
        indicator: str | None = None,
        reference_timeseries: pd.Series | None = None,
        title: str | None = None,
        y_label: str | None = None,
        x_label: str | None = None,
        alpha: float = 0.5,
        show_legends: bool = False,
        round_ndigits: int = 2,
    ) -> go.Figure:
        if self.results is None:
            raise ValueError("No results available to plot." " Run a simulation first.")

        return plot_sample(
            results=self.results,
            indicator=indicator,
            reference_timeseries=reference_timeseries,
            title=title,
            y_label=y_label,
            x_label=x_label,
            alpha=alpha,
            show_legends=show_legends,
            parameter_values=self.values,
            parameter_names=[p.name for p in self.parameters],
            round_ndigits=round_ndigits,
        )


class Sampler(ABC):
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

    def plot_sample(
        self,
        indicator: str | None = None,
        reference_timeseries: pd.Series | None = None,
        title: str | None = None,
        y_label: str | None = None,
        x_label: str | None = None,
        alpha: float = 0.5,
        show_legends: bool = False,
        parameter_values: np.ndarray | None = None,
        parameter_names: list[str] | None = None,
        round_ndigits: int = 2,
    ) -> go.Figure:
        if parameter_values is None:
            parameter_values = self.values
        if parameter_names is None:
            parameter_names = [p.name for p in self.parameters]

        return plot_sample(
            self.results,
            indicator=indicator,
            reference_timeseries=reference_timeseries,
            title=title,
            y_label=y_label,
            x_label=x_label,
            alpha=alpha,
            show_legends=show_legends,
            parameter_values=parameter_values,
            parameter_names=parameter_names,
            round_ndigits=round_ndigits,
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
    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
    ):
        super().__init__(parameters, model, simulation_options)

        if not all(param.ptype == "Real" for param in parameters):
            raise ValueError(
                f"All parameters must have a ptype 'Real'"
                f"Found {
                [(par.name, par.ptype) for par in parameters if par.ptype != "Real"]}"
            )

    def get_salib_problem(self):
        return {
            "num_vars": len(self.parameters),
            "names": [p.name for p in self.parameters],
            "bounds": [p.interval for p in self.parameters],
        }


class MorrisSampler(RealSampler):
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
    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)

    def add_sample(self, n: int, rng: int = None, simulate=True, **lhs_kwargs):
        lhs = LatinHypercube(d=len(self.parameters), rng=rng, **lhs_kwargs)
        new_dimless_sample = lhs.random(n=n)
        self._post_draw_sample(new_dimless_sample, simulate, sample_is_dimless=True)


class SobolSampler(RealSampler):
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


class SaltelliSampler(RealSampler):
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
        *,
        calc_second_order: bool = True,
        **saltelli_kwargs,
    ):
        new_sample = saltelli.sample(
            problem=self.get_salib_problem(),
            N=N,
            calc_second_order=calc_second_order,
            **saltelli_kwargs,
        )
        self._post_draw_sample(new_sample, simulate, n_cpu, sample_is_dimless=False)


def plot_sample(
    results: pd.Series,
    indicator: str | None = None,
    reference_timeseries: pd.Series | None = None,
    title: str | None = None,
    y_label: str | None = None,
    x_label: str | None = None,
    alpha: float = 0.5,
    show_legends: bool = False,
    parameter_values: np.ndarray | None = None,
    parameter_names: list[str] | None = None,
    round_ndigits: int = 2,
) -> go.Figure:
    """
    Plot all available (non-empty) simulation results contained in a Series, optionally
    annotating each trace legend with the corresponding parameter values.

    Parameters
    ----------
    results : pd.Series
        A pandas Series where each element is either a pandas Series or a pandas DataFrame
        (typically 1 column per indicator). Empty elements are ignored.
    indicator : str, optional
        Column name to use if inner elements are DataFrames with multiple columns.
        If None and the DataFrame has exactly one column, that column is used.
    reference_timeseries : pd.Series, optional
        A reference time series to plot alongside simulations.
    title, y_label, x_label : str, optional
        Plot title and axis labels.
    alpha : float, default 0.5
        Opacity for the markers.
    show_legends : bool, default False
        Whether to show a legend per sample trace.
    parameter_values : np.ndarray, optional
        Array of shape (n_samples, n_params) with the parameter values used per sample.
        Only used to build legend strings when `show_legends=True`.
    parameter_names : list[str], optional
        Names of the parameters (same order as in `parameter_values`).
    round_ndigits : int, default 2
        Number of digits for rounding parameter values in legend strings.

    Returns
    -------
    go.Figure
    """

    if not isinstance(results, pd.Series):
        raise ValueError("`results` must be a pandas Series.")
    if results.empty:
        raise ValueError("`results` is empty. Simulate samples first.")

    ref_name = getattr(reference_timeseries, "name", None)

    def _to_series(obj, indicator_, ref_name_):
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                return None
            if indicator_ is not None:
                return obj[indicator_]
            if ref_name_ is not None and ref_name_ in obj.columns:
                return obj[ref_name_]
            if obj.shape[1] == 1:
                return obj.iloc[:, 0]
            raise ValueError(
                "Provide `indicator`: multiple columns in the sample DataFrame."
            )
        return None

    def _legend_for(i: int) -> str:
        if not show_legends:
            return "Simulations"
        if parameter_values is None or parameter_names is None:
            return f"sample {i}"
        vals = parameter_values[i]
        return ", ".join(
            f"{n}: {round(v, round_ndigits)}" for n, v in zip(parameter_names, vals)
        )

    fig = go.Figure()
    plotted = 0

    for i, sample in enumerate(results):
        s = _to_series(sample, indicator, ref_name)
        if s is None or s.empty:
            continue
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
        plotted += 1

    if plotted == 0 and reference_timeseries is None:
        raise ValueError("No simulated data available to plot.")

    if reference_timeseries is not None:
        fig.add_trace(
            go.Scattergl(
                name="Reference",
                mode="lines",
                x=reference_timeseries.index,
                y=reference_timeseries.to_numpy(),
                marker=dict(color="red"),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=show_legends or (reference_timeseries is not None),
    )
    return fig


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
