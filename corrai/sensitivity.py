from abc import ABC, abstractmethod

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from SALib.analyze import morris, sobol, fast, rbd_fast
from corrai.base.parameter import Parameter
from corrai.base.sampling import (
    SaltelliSampler,
    SobolSampler,
    MorrisSampler,
    FASTSampler,
    RBDFASTSampler,
)
from corrai.base.model import Model


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
        """
        Plot all available simulation runs against an optional reference time series.

        This method wraps :meth:`Sampler.plot_sample` and plots the simulations
        associated with this sensitivity analysis instance. Each run is displayed
        as a scatter trace, optionally annotated with parameter values.

        Parameters
        ----------
        indicator : str, optional
            Column name to select if simulation outputs are DataFrames with multiple
            columns. If None and a DataFrame has a single column, that column is used.
        reference_timeseries : pd.Series, optional
            A time series to plot as ground truth or reference, shown as a red line.
        title : str, optional
            Plot title.
        y_label : str, optional
            Label for the y-axis.
        x_label : str, optional
            Label for the x-axis.
        alpha : float, default=0.5
            Opacity for the sample markers.
        show_legends : bool, default=False
            Whether to display a legend entry for each sample.
        parameter_values : np.ndarray, optional
            Custom parameter values for legend annotation. If None, values from
            this Sanalysis instance are used.
        parameter_names : list of str, optional
            Custom parameter names. If None, names from this Sanalysis instance are used.
        round_ndigits : int, default=2
            Number of decimal digits for rounding parameter values in legends.

        Returns
        -------
        go.Figure
            A Plotly Figure containing the sample simulations and optional reference.
        """
        return self.sampler.plot_sample(
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
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        prefix: str | None = None,
        bounds: list[tuple[float, float]] | None = None,
        color_by: str | None = None,
        title: str | None = "Parallel Coordinates - Samples",
        html_file_path: str | None = None,
    ) -> go.Figure:
        """
        Create a Parallel Coordinates Plot (PCP) of parameters and aggregated results.

        Each vertical axis corresponds to a parameter or an aggregated indicator,
        and each polyline represents one simulation. Useful for visualizing the
        relationship between sampled parameters and performance metrics.

        This method wraps :meth:`Sampler.plot_pcp`.

        Parameters
        ----------
        indicator : str, optional
            Indicator name to extract from simulation results before aggregation.
            If None, only parameters are shown.
        method : str, default="mean"
            Aggregation method to apply. Supported values include:
            - "mean"
            - "sum"
            - "nmbe"
            - "cv_rmse"
            - "mean_squared_error"
            - "mean_absolute_error"
        agg_method_kwarg : dict, optional
            Extra keyword arguments passed to the aggregation function.
        reference_time_series : pd.Series, optional
            Required for error-based methods (e.g., "cv_rmse"). Must have the same
            index and length as each simulation.
        freq : str or pd.Timedelta or datetime.timedelta, optional
            If provided, aggregation is performed per time bin.
        prefix : str, optional
            Custom prefix for naming aggregated columns. Defaults to the method name.
        bounds : list of tuple(float, float), optional
            Parameter bounds for each parameter axis.
        color_by : str, optional
            Column name (parameter or aggregate) used to color polylines.
        title : str, optional
            Figure title.
        html_file_path : str, optional
            If provided, saves the plot as an interactive HTML file.

        Returns
        -------
        go.Figure
            A Plotly Figure with the parallel coordinates visualization.
        """

        return self.sampler.plot_pcp(
            indicator=indicator,
            method=method,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            freq=freq,
            prefix=prefix,
            bounds=bounds,
            color_by=color_by,
            title=title,
            html_file_path=html_file_path,
        )

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
                index=[par.name for par in self.sampler.sample.parameters],
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
        stacked: bool = False,
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
            columns=[par.name for par in self.sampler.sample.parameters],
            index=res.index,
        )

        return plot_dynamic_metric(metrics, sensitivity_metric, unit, title, stacked)

    # def plot_pcp(
    #     self,
    #     aggregations: dict | None = None,  # <= optionnel
    #     *,
    #     bounds: list[tuple[float, float]] | None = None,
    #     color_by: str | None = None,
    #     title: str | None = "Parallel Coordinates — Samples",
    #     html_file_path: str | None = None,
    # ):
    #     """
    #     Parallel Coordinates Plot basé sur les échantillons et résultats présents dans l'analyse.
    #
    #     Parameters
    #     ----------
    #     aggregations : dict
    #         {indicator: callable | [callable] | {label: callable}}
    #         Ex. {"res": [np.sum, np.mean]}  -> colonnes "res:sum", "res:mean".
    #     bounds : list[(float, float)] | None
    #         Bornes (min, max) par paramètre (même ordre que les paramètres). Si None, autoscale.
    #     color_by : str | None
    #         Nom d'une dimension (paramètre ou indicateur agrégé) pour colorer les lignes.
    #     title : str | None
    #         Titre.
    #     html_file_path : str | None
    #         Si fourni, export HTML.
    #     """
    #     results = self.sampler.sample.results
    #     parameter_values = self.sampler.sample.values
    #     parameter_names = [p.name for p in self.sampler.sample.parameters]
    #
    #     return _plot_pcp(
    #         results=results,
    #         parameter_values=parameter_values,
    #         parameter_names=parameter_names,
    #         aggregations=aggregations,
    #         bounds=bounds,
    #         color_by=color_by,
    #         title=title,
    #         html_file_path=html_file_path,
    #     )

    def salib_plot_matrix(
        self,
        indicator: str,
        sensitivity_method_name: str,
        method: str = "mean",
        unit: str = "",
        reference_time_series: pd.Series = None,
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwarg,
    ):
        title = (
            f"{sensitivity_method_name} {method} {indicator} "
            f"- 2nd order interactions"
            if title is None
            else title
        )

        result = self.analyze(
            indicator,
            method,
            agg_method_kwarg,
            reference_time_series,
            freq=None,
            **analyse_kwarg,
        )[f"{method}_{indicator}"]

        parameter_names = [p.name for p in self.sampler.sample.parameters]
        return plot_s2_matrix(result, parameter_names, title=title)


class SobolSanalysis(Sanalysis):
    """
    Sobol sensitivity analysis class using Saltelli sampling.

    This class extends :class:`Sanalysis` and provides variance-based global
    sensitivity analysis following the Sobol method. Sampling of the parameter
    space is performed using the Saltelli scheme, which ensures efficient
    estimation of first-order, second-order, and total-order Sobol indices.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
        sampler: str = "saltelli",
    ):
        self._sampler_choice = sampler
        super().__init__(parameters, model, simulation_options)

    def _set_sampler(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        if self._sampler_choice == "saltelli":
            return SaltelliSampler(parameters, model, simulation_options)
        elif self._sampler_choice == "sobol":
            return SobolSampler(parameters, model, simulation_options)
        else:
            raise ValueError("sampler must be 'saltelli' or 'sobol'")

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
        **sample_kwargs,
    ):
        super().add_sample(
            N=N,
            simulate=simulate,
            n_cpu=n_cpu,
            calc_second_order=calc_second_order,
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

    def plot_bar(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "ST",
        method: str = "mean",
        reference_time_series: pd.Series = None,
        calc_second_order: bool = True,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwargs,
    ):
        return super().salib_plot_bar(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="Sobol",
            method=method,
            unit=unit,
            reference_time_series=reference_time_series,
            agg_method_kwarg=agg_method_kwarg,
            title=title,
            calc_second_order=calc_second_order,
            **analyse_kwargs,
        )

    def plot_dynamic_metric(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "ST",
        freq: str | pd.Timedelta | dt.timedelta = None,
        method: str = "mean",
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
        calc_second_order: bool = True,
        title: str = None,
    ):
        return super().salib_plot_dynamic_metric(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="Sobol",
            freq=freq,
            method=method,
            unit=unit,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            calc_second_order=calc_second_order,
            stacked=True,
            title=title,
        )

    def plot_s2_matrix(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "S2",
        method: str = "mean",
        reference_time_series: pd.Series = None,
        calc_second_order: bool = True,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwargs,
    ):
        return super().salib_plot_matrix(
            indicator=indicator,
            sensitivity_method_name="Sobol",
            method=method,
            unit=unit,
            reference_time_series=reference_time_series,
            agg_method_kwarg=agg_method_kwarg,
            title=title,
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
        reference_time_series: pd.Series = None,
        agg_method_kwarg: dict = None,
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
            result = self.analyze(
                indicator=indicator,
                method=method,
                reference_time_series=reference_time_series,
                agg_method_kwarg=agg_method_kwarg,
                **analyse_kwargs,
            )[f"{method}_{indicator}"]
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
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwargs,
    ):
        return super().salib_plot_bar(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="Morris",
            method=method,
            unit=unit,
            reference_time_series=reference_time_series,
            agg_method_kwarg=agg_method_kwarg,
            title=title,
            **analyse_kwargs,
        )

    def plot_dynamic_metric(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "euclidian_distance",
        freq: str | pd.Timedelta | dt.timedelta = None,
        method: str = "mean",
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
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


class FASTSanalysis(Sanalysis):
    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options, x_needed=False)
        self._analysis_cache = {}

    def _set_sampler(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        return FASTSampler(parameters, model, simulation_options)

    def _set_analyser(self):
        return fast

    def add_sample(
        self,
        N: int,
        M: int = 4,
        simulate: bool = True,
        n_cpu: int = 1,
        **sample_kwargs,
    ):
        super().add_sample(
            N=N,
            simulate=simulate,
            n_cpu=n_cpu,
            M=M,
            **sample_kwargs,
        )

    def analyze(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        **analyse_kwargs,
    ):
        return super().analyze(
            indicator=indicator,
            method=method,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            freq=freq,
            **analyse_kwargs,
        )

    def plot_bar(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "ST",
        method: str = "mean",
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwargs,
    ):
        return super().salib_plot_bar(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="FAST",
            method=method,
            unit=unit,
            reference_time_series=reference_time_series,
            agg_method_kwarg=agg_method_kwarg,
            title=title,
            **analyse_kwargs,
        )

    def plot_dynamic_metric(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "ST",
        freq: str | pd.Timedelta | dt.timedelta = None,
        method: str = "mean",
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
    ):
        return super().salib_plot_dynamic_metric(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="FAST",
            freq=freq,
            method=method,
            unit=unit,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            stacked=True,
            title=title,
        )


class RBDFASTSanalysis(Sanalysis):
    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options, x_needed=True)

    def _set_sampler(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        return RBDFASTSampler(parameters, model, simulation_options)

    def _set_analyser(self):
        return rbd_fast

    def add_sample(
        self,
        N: int,
        simulate: bool = True,
        n_cpu: int = 1,
        **sample_kwargs,
    ):
        super().add_sample(
            N=N,
            simulate=simulate,
            n_cpu=n_cpu,
            **sample_kwargs,
        )

    def analyze(
        self,
        indicator: str,
        method: str = "mean",
        agg_method_kwarg: dict = None,
        reference_time_series: pd.Series = None,
        freq: str | pd.Timedelta | dt.timedelta = None,
        **analyse_kwargs,
    ):
        return super().analyze(
            indicator=indicator,
            method=method,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            freq=freq,
            **analyse_kwargs,
        )

    def plot_bar(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "S1",
        method: str = "mean",
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
        **analyse_kwargs,
    ):
        return super().salib_plot_bar(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="RBD_FAST",
            method=method,
            unit=unit,
            reference_time_series=reference_time_series,
            agg_method_kwarg=agg_method_kwarg,
            title=title,
            **analyse_kwargs,
        )

    def plot_dynamic_metric(
        self,
        indicator: str = "res",
        sensitivity_metric: str = "S1",
        freq: str | pd.Timedelta | dt.timedelta = None,
        method: str = "mean",
        reference_time_series: pd.Series = None,
        unit: str = "",
        agg_method_kwarg: dict = None,
        title: str = None,
    ):
        return super().salib_plot_dynamic_metric(
            indicator=indicator,
            sensitivity_metric=sensitivity_metric,
            sensitivity_method_name="RBD_FAST",
            freq=freq,
            method=method,
            unit=unit,
            agg_method_kwarg=agg_method_kwarg,
            reference_time_series=reference_time_series,
            stacked=True,
            title=title,
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


def plot_s2_matrix(
    result: dict,
    param_names: list[str],
    title: str = "Sobol 2nd-order interactions (S2)",
    colorscale: str = "Reds",
):
    df_S2 = pd.DataFrame(result["S2"], index=param_names, columns=param_names)

    fig = go.Figure(
        data=go.Heatmap(
            z=df_S2.values,
            x=df_S2.columns,
            y=df_S2.index,
            colorscale=colorscale,
            zmin=0,
            zmax=df_S2.values.max(),
            colorbar=dict(title="S2"),
            text=df_S2.round(3).astype(str),
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Parameter",
        yaxis_title="Parameter",
    )

    return fig
