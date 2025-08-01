import enum
from abc import ABC, abstractmethod
from typing import Any, Callable

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from SALib.analyze import fast, morris, sobol, rbd_fast
from SALib.sample import sobol as sobol_sampler
from SALib.sample import fast_sampler, latin
from SALib.sample import morris as morris_sampler

from corrai.base.parameter import Parameter
from corrai.base.sampling import SobolSampler, MorrisSampler
from corrai.base.model import Model
from corrai.base.simulate import run_simulations
from corrai.base.math import aggregate_time_series
from corrai.multi_optimize import plot_parcoord


class Method(enum.Enum):
    FAST = "FAST"
    MORRIS = "MORRIS"
    SOBOL = "SOBOL"
    RBD_FAST = "RBD_FAST"


METHOD_SAMPLER_DICT = {
    Method.FAST: {
        "method": fast,
        "sampling": fast_sampler,
    },
    Method.MORRIS: {"method": morris, "sampling": morris_sampler},
    Method.SOBOL: {
        "method": sobol,
        "sampling": sobol_sampler,
    },
    Method.RBD_FAST: {
        "method": rbd_fast,
        "sampling": latin,
    },
}


class Sanalysis(ABC):
    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
        x_needed: bool = False,
    ):
        self.sampler = self._set_sampler(parameters, model, simulation_options)
        self.analyser = self._set_analyser()
        self.x_needed = x_needed

    @property
    def parameters(self):
        return self.sampler.parameters

    @property
    def values(self):
        return self.sampler.values

    @property
    def results(self):
        return self.sampler.results

    @abstractmethod
    def _set_sampler(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        """Return a method-specific sampler."""
        pass

    @abstractmethod
    def _set_analyser(self):
        pass

    def add_sample(
        self,
        simulate: bool = True,
        n_cpu: int = 1,
        **sample_kwargs,
    ):
        self.sampler.add_sample(simulate=simulate, n_cpu=n_cpu, **sample_kwargs)

    def analyze(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str = None,
        **analyse_kwargs,
    ):
        agg_result = self.sampler.sample.get_aggregate_time_series(
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq,
            prefix=method,
        )

        if self.x_needed:
            analyse_kwargs["X"] = self.sampler.sample.get_dimension_less_values()

        analyse_kwargs["problem"] = self.sampler.get_salib_problem()

        if freq is None:
            analyse_kwargs["Y"] = agg_result.to_numpy().flatten()
            res = self.analyser.analyze(**analyse_kwargs)
            return pd.Series({f"{method}_{indicator}": res})

        else:
            result_dict = {}
            for tstamp in agg_result:
                analyse_kwargs["Y"] = agg_result[tstamp].to_numpy().flatten()
                res = self.analyser.analyze(**analyse_kwargs)
                result_dict[tstamp] = res
            return pd.Series(result_dict)

    def salib_plot_bar(
        self,
        indicator: str,
        sensitivity_metric: str,
        sensitivity_method_name: str,
        method: str = "mean",
        unit: str = "",
        reference_time_series: pd.Series = None,
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwarg,
    ):
        title = (
            f"{sensitivity_method_name} {sensitivity_metric} {method} {indicator}"
            if title is None
            else title
        )
        res = self.analyze(
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq=None,
            **analyse_kwarg,
        )[f"{method}_{indicator}"]

        return plot_bars(
            pd.Series(
                data=res[sensitivity_metric],
                index=res["names"],
                name=f"{sensitivity_metric} {unit}",
            ),
            title=title,
        )

    def salib_plot_dynamic_metric(
        self,
        indicator: str,
        sensitivity_metric: str,
        sensitivity_method_name: str,
        freq: str | pd.Timedelta | dt.timedelta = None,
        method: str = "mean",
        unit: str = "",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        title: str = None,
        **analyse_kwarg,
    ):
        title = (
            f"{sensitivity_method_name} dynamic {sensitivity_metric} {method} {indicator}"
            if title is None
            else title
        )

        if freq is None:
            freq = pd.infer_freq(self.sampler.results[0].index)
            if freq is None:
                raise ValueError(
                    "freq is not specified and cannot be inferred"
                    "from results. Specify freq for analyse"
                )

        res = self.analyze(
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq,
            **analyse_kwarg,
        )

        metrics = pd.DataFrame(
            data=[val[sensitivity_metric] for val in res],
            columns=res.iloc[0]["names"],
            index=res.index,
        )

        return plot_dynamic_metric(
            metrics, sensitivity_metric, unit, title, stacked=False
        )


class SobolSanalysis(Sanalysis):
    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)

    def _set_sampler(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        return SobolSampler(parameters, model, simulation_options)

    def _set_analyser(self):
        return sobol

    # noinspection PyMethodOverriding
    def add_sample(
        self,
        N: int,
        simulate: bool = True,
        n_cpu: int = 1,
        *,
        calc_second_order: bool = True,
        scramble: bool = True,
        **sample_kwargs,
    ):
        super().add_sample(
            N=N,
            simulate=simulate,
            n_cpu=n_cpu,
            calc_second_order=calc_second_order,
            scramble=scramble,
            **sample_kwargs,
        )

    def analyze(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        calc_second_order: bool = True,
        **analyse_kwargs,
    ):
        return super().analyze(
            indicator=indicator,
            method=method,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            freq=freq,
            calc_second_order=calc_second_order,
            **analyse_kwargs,
        )


class MorrisSanalysis(Sanalysis):
    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options, x_needed=True)
        self._analysis_cache = {}

    def _set_sampler(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        return MorrisSampler(parameters, model, simulation_options)

    def _set_analyser(self):
        return morris

    # noinspection PyMethodOverriding
    def add_sample(
        self,
        N: int,
        simulate: bool = True,
        n_cpu: int = 1,
        num_levels: int = 4,
        **sample_kwargs,
    ):
        super().add_sample(
            N=N,
            simulate=simulate,
            n_cpu=n_cpu,
            num_levels=num_levels,
            **sample_kwargs,
        )

    def analyze(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        **analyse_kwargs: object,
    ) -> pd.Series:
        res = super().analyze(
            indicator=indicator,
            method=method,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            freq=freq,
            **analyse_kwargs,
        )

        for idx in res.index:
            res[idx]["euclidian_distance"] = np.sqrt(
                res[idx]["mu_star"] ** 2 + res[idx]["sigma"] ** 2
            )

        return res

    def plot_scatter(
        self,
        indicator: str = "res",
        method: str = "mean",
        title: str = "Morris Sensitivity Analysis",
        unit: str = "",
        scaler: float = 100,
        autosize: bool = True,
        **analyse_kwargs,
    ):
        cache_key = (indicator, method, "None")
        if cache_key in self._analysis_cache:
            result = self._analysis_cache[cache_key][f"{method}_{indicator}"]
        else:
            result = self.analyze(indicator=indicator, method=method, **analyse_kwargs)[
                f"{method}_{indicator}"
            ]
            self._analysis_cache[cache_key] = {f"{method}_{indicator}": result}

        return plot_morris_scatter(
            result,
            title=title,
            unit=unit,
            scaler=scaler,
            autosize=autosize,
        )

    def plot_bar(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "euclidian_distance",
        method: str = "mean",
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
    ):
        return super().salib_plot_bar(
            indicator,
            sensitivity_metric,
            "Morris",
            method,
            unit,
            agg_method_kwarg,
            title,
        )

    def plot_dynamic_metric(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "euclidian_distance",
        freq: str | pd.Timedelta | dt.timedelta = None,
        method: str = "mean",
        unit: str = "",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        title: str = None,
    ):
        return super().salib_plot_dynamic_metric(
            indicator,
            sensitivity_metric,
            "Morris",
            freq,
            method,
            unit,
            agg_method_kwarg,
            reference_time_series,
            title,
        )

def plot_dynamic_metric(
    metrics: pd.DataFrame,
    metric_name: str = "",
    unit: str = "",
    title: str = None,
    stacked: bool = False,
):
    fig = go.Figure()
    for param in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics.index,
                y=metrics[param],
                name=param,
                mode="lines",
                stackgroup="one" if stacked else None,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=f"{metric_name} {unit}",
    )

    return fig


def plot_bars(
    sensitivity_results: pd.Series, title: str = None, error: pd.Series = None
):
    error = {} if error is None else dict(type="data", array=error.values)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sensitivity_results.index,
            y=sensitivity_results.values,
            error_y=error,
            name=sensitivity_results.name,
            marker_color="orange",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Parameters",
        yaxis_title=f"{sensitivity_results.name}",
    )

    return fig


def plot_morris_scatter(
    morris_result,
    title: str = "Morris Sensitivity Analysis",
    unit: str = "",
    scaler: float = 100,
    autosize: bool = True,
) -> go.Figure:
    """
    Plot a Morris sensitivity analysis scatter plot using μ* and σ.

    Parameters
    ----------
    morris_result : MorrisResult or pd.DataFrame
        Result from a Morris analysis (SALib or internal format).
    title : str, optional
        Plot title.
    unit : str, optional
        Unit for axis labels.
    scaler : float, optional
        Scaling factor for marker size.
    autosize : bool, optional
        Whether to autoscale y-axis (True: based on σ, False: based on μ*).
    """
    if hasattr(morris_result, "to_df"):
        morris_df = morris_result.to_df()
    elif isinstance(morris_result, pd.DataFrame):
        morris_df = morris_result
    else:
        raise ValueError("Expected `MorrisResult` or `pd.DataFrame`.")

    morris_df["distance"] = np.sqrt(morris_df.mu_star**2 + morris_df.sigma**2)
    morris_df["dimless_distance"] = morris_df["distance"] / morris_df["distance"].max()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=morris_df["mu_star"],
            y=morris_df["sigma"],
            mode="markers+text",
            name="Morris index",
            text=list(morris_df.index),
            textposition="top center",
            marker=dict(
                size=morris_df["dimless_distance"] * scaler,
                color=np.arange(len(morris_df)),
            ),
            error_x=dict(
                type="data",
                array=morris_df["mu_star_conf"],
                color="#696969",
                visible=True,
            ),
        )
    )

    x_max = morris_df["mu_star"].max() * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, x_max],
            y=[0, 0.1 * x_max],
            name="linear",
            mode="lines",
            line=dict(dash="dash", color="grey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, x_max],
            y=[0, 0.5 * x_max],
            name="monotonic",
            mode="lines",
            line=dict(dash="dot", color="grey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, x_max],
            y=[0, x_max],
            name="non-linear",
            mode="lines",
            line=dict(dash="dashdot", color="grey"),
        )
    )

    y_max = morris_df["sigma"].max() * 1.5 if autosize else x_max
    fig.update_layout(
        title=title,
        xaxis_title=f"Absolute mean of elementary effects μ* [{unit}]",
        yaxis_title=f"Standard deviation of elementary effects σ [{unit}]",
        yaxis_range=[-0.1 * y_max, y_max],
    )

    return fig


class ObjectiveFunction:
    """
    A class to represent configure an objective function for model calibration
    and optimization.
    A specific method is designed for scipy compatibility.

    Parameters
    ----------
    model : Model
        The model to be calibrated.
    simulation_options : dict
        Dictionary containing simulation options.
    param_list : list of dict
        List of dictionaries specifying the parameters to be calibrated,
        including bounds and initial values.
    indicators_config : dict[str, tuple[Callable, pd.Series | pd.DataFrame | None] | Callable]
        Dictionary where keys are indicator names corresponding to Model simulation
         output names, and values are either:
        - A aggregation function to compute the indicator (ex: np.mean, np.sum).
        - A tuple (function, reference data) if comparison when reference is needed.
        (ex. sklearn.metrics.mean_squared_error, corrai.metrics.mae)
    scipy_obj_indicator : str, optional
        The indicator to be used as the objective function for scipy optimization.
        Defaults to the first key in `indicators_config`.

    Attributes
    ----------
    model : Model
        The simulation model being calibrated.
    simulation_options : dict
        Options for running the simulation.
    param_list : list of dict
        List of parameter dictionaries, each containing:
        - `Parameter.NAME`: Name of the parameter.
        - `Parameter.INTERVAL`: Bounds for the parameter.
        - `Parameter.INIT_VALUE`: Initial value (optional).
    indicators_config : dict
        Dictionary mapping indicator names to their computation functions and optional reference data.
    scipy_obj_indicator : str
        The indicator used as the objective function in `scipy.optimize`.

    Properties
    ----------
    bounds : list[tuple[float, float]]
        List of parameter bounds extracted from `param_list`.
    init_values : list[float] or None
        List of initial values for parameters if provided; otherwise, None.

    Methods
    -------
    function(param_dict, kwargs: dict = None) -> dict[str, float]
        Runs the model simulation and computes indicator values.

        Parameters
        ----------
        param_dict : dict
            Dictionary containing parameter names and their values.
        kwargs : dict, optional
            Additional arguments for the model simulation.

        Returns
        -------
        dict[str, float]
            Dictionary containing computed indicator values.

    scipy_obj_function(x, kwargs: dict = None) -> float
        Computes the objective function value for scipy optimization.

        Parameters
        ----------
        x : float or list of float
            List of parameter values or a single parameter value.
        kwargs : dict, optional
            Additional arguments for the model simulation.

        Returns
        -------
        float
            The calculated value of the objective function based on `scipy_obj_indicator`.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> model = SomeModel()
    >>> simulation_options = {"duration": 100}
    >>> param_list = [{Parameter.NAME: "param1", Parameter.INTERVAL: [0, 1], Parameter.INIT_VALUE: 0.5}]
    >>> indicators_config = {"res": (mean_squared_error, reference_series)}
    >>> obj_func = ObjectiveFunction(model, simulation_options, param_list, indicators_config)
    >>> obj_func.function({"param1": 0.8})
    {'indicator1': 0.123}
    >>> obj_func.scipy_obj_function([0.8])
    0.123
    """

    def __init__(
        self,
        model: Model,
        simulation_options: dict,
        param_list: list[dict],
        indicators_config: dict[
            str, tuple[Callable, pd.Series | pd.DataFrame | None] | Callable
        ],
        scipy_obj_indicator: str = None,
    ):
        self.model = model
        self.param_list = param_list
        self.indicators_config = indicators_config
        self.simulation_options = simulation_options
        self.scipy_obj_indicator = (
            list(indicators_config.keys())[0]
            if scipy_obj_indicator is None
            else scipy_obj_indicator
        )

    @property
    def bounds(self):
        return [param[Parameter.INTERVAL] for param in self.param_list]

    @property
    def init_values(self):
        if all(Parameter.INIT_VALUE in param for param in self.param_list):
            return [param[Parameter.INIT_VALUE] for param in self.param_list]
        else:
            return None

    def function(self, param_dict, kwargs: dict = None):
        """
        Calculate the objective function for given parameter values.

        Parameters
        ----------
        x_dict : dict
            Dictionary containing parameter names and their values.

        Returns
        -------
        pd.Series
            A series containing the calculated values for each indicator.
        """
        kwargs = {} if kwargs is None else kwargs
        res = self.model.simulate(param_dict, self.simulation_options, **kwargs)

        function_results = {}
        for ind, (func, ref) in self.indicators_config.items():
            function_results[ind] = (
                func(res[ind]) if ref is None else func(res[ind], ref)
            )
        return function_results

    def scipy_obj_function(self, x, kwargs: dict = None):
        """
        Wrapper for scipy.optimize that calculates the
        objective function for given parameter values.

        Parameters
        ----------
        x : float or list of float
            List of parameter values or a single parameter value.

        Returns
        -------
        float
            The calculated value of the objective function.
        """
        x = [x] if isinstance(x, (float, int)) else x
        if len(x) != len(self.param_list):
            raise ValueError("Length of x does not match length of param_list")

        param_dict = {
            self.param_list[i][Parameter.NAME]: x[i]
            for i in range(len(self.param_list))
        }
        result = self.function(param_dict, kwargs)
        return result[self.scipy_obj_indicator]
