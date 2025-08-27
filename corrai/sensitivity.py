import enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from SALib.analyze import fast, morris, sobol, rbd_fast
from SALib.sample import sobol as sobol_sampler
from SALib.sample import fast_sampler, latin
from SALib.sample import morris as morris_sampler

from corrai.base.parameter import Parameter
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


class SAnalysis:
    """
    This class is designed to perform sensitivity analysis on a given model using
    various sensitivity methods,  Global Sensitivity Analysis (GSA) methods such as
    Sobol and FAST.

    Parameters:
    - parameters_list (list of dict): A list of dictionaries describing the model
        parameters to be analyzed. Each dictionary should contain 'name', 'interval',
        and 'type' keys to define the parameter name, its range (interval),
        and its data type (Real, integer).
    - method (str): The sensitivity analysis method to be used.
        Supported methods are specified in METHOD_SAMPLER_DICT.keys().

    Attributes:
    - method (str): The selected sensitivity analysis method.
    - parameters_list (list of dict): The list of model parameters for analysis.
    - _salib_problem (dict): Problem definition for the sensitivity analysis library
        SAlib.
    - sample (DataFrame): The generated parameter samples.
    - sensitivity_results (dict): The sensitivity analysis results.
    - sample_results (list): List of results obtained from model evaluations.

    Methods:
    - set_parameters_list(parameters_list): Sets the parameters list and updates
        the _salib_problem.
    - draw_sample(n, sampling_kwargs=None): Draws a sample of parameter values for
        sensitivity analysis.
    - evaluate(model, simulation_options, n_cpu=1): Evaluates the model for the
        generated parameter samples.
    - analyze(indicator, agg_method=np.mean, agg_method_kwarg={}, reference_df=None,
        sensitivity_method_kwargs={}): Performs sensitivity analysis on the model
        outputs using the selected method and indicator.

    """

    def __init__(self, parameters_list: list[dict[Parameter, Any]], method: Method):
        self.method = method
        self.parameters_list = parameters_list
        self._salib_problem = None
        self.set_parameters_list(parameters_list)
        self.sample = None
        self.sensitivity_results = None
        self.sample_results = []
        self.static = True  # Flag to indicate if the analysis is static or dynamic

    def set_parameters_list(self, parameters_list: list[dict[Parameter, Any]]):
        """
        Set the list of model parameters and update the _salib_problem definition.

        Parameters:
        - parameters_list (list of dict): A list of dictionaries describing the model
            parameters to be analyzed. Each dictionary should contain 'name',
             'interval', and 'type' keys.
        """
        self._salib_problem = {
            "num_vars": len(parameters_list),
            "names": [p[Parameter.NAME] for p in parameters_list],
            "bounds": list(map(lambda p: p[Parameter.INTERVAL], parameters_list)),
        }

    def draw_sample(self, n: int, sampling_kwargs: dict = None):
        """
        Samples the parameters for sensitivity analysis.

        Parameters:
        - n (int): The number of samples to generate.
        - sampling_kwargs (dict, optional): Additional keyword arguments for the
            sampling method.
        """
        # Erase previous result to prevent false indicator
        self.sample_results = []

        if sampling_kwargs is None:
            sampling_kwargs = {}

        sampler = METHOD_SAMPLER_DICT[self.method]["sampling"]
        sample_temp = sampler.sample(
            N=n, problem=self._salib_problem, **sampling_kwargs
        )

        for index, param in enumerate(self.parameters_list):
            vtype = param[Parameter.TYPE]
            if vtype == "Integer":
                sample_temp[:, index] = np.round(sample_temp[:, index])
                sample_temp = np.unique(sample_temp, axis=0)

        self.sample = pd.DataFrame(sample_temp, columns=self._salib_problem["names"])

    def evaluate(
        self,
        model,
        simulation_options: dict = None,
        n_cpu: int = 1,
        simulate_kwargs: dict = None,
    ):
        """
        Evaluate the model for the generated samples.

        Parameters:
        - model: The model to be evaluated.
        - simulation_options (dict): Options for running the model simulations.
        - n_cpu (int, optional): Number of CPU cores to use for parallel evaluation.
        """
        if self.sample is None:
            raise ValueError(
                "Cannot perform evaluation, no sample was found. "
                "draw sample using draw_sample() method"
            )

        self.sample_results = run_simulations(
            model=model,
            parameter_samples=self.sample,
            simulation_options=simulation_options,
            n_cpu=n_cpu,
            simulate_kwargs=simulate_kwargs,
        )

    def analyze(
        self,
        indicator: str,
        agg_method=np.mean,
        agg_method_kwarg: dict = None,
        reference_df: pd.DataFrame = None,
        sensitivity_method_kwargs: dict = None,
        freq: str = None,
        absolute: bool = False,
    ):
        """
        Perform sensitivity analysis on the model outputs using the selected method
        and indicator. Supports both static and dynamic analysis.

        Parameters:
        ----------
        indicator : str
            The name of the model output to analyze (must match a column name in the
            simulation results).

        agg_method : Callable, optional
            Aggregation method to reduce time series to a single value per simulation.
            Default is numpy.mean.

        agg_method_kwarg : dict, optional
            Additional keyword arguments passed to the aggregation method.

        reference_df : pd.DataFrame, optional
            Optional reference data (e.g., measured values) to use with error-based
            aggregation methods (e.g., RMSE, NMBE). If provided, must align in time
            with the simulation output.

        sensitivity_method_kwargs : dict, optional
            Additional keyword arguments passed to the sensitivity analysis method
            from SALib.

        freq : str, optional
            If provided (e.g., "6h", "1D"), enables dynamic sensitivity analysis by
            aggregating results over time intervals. Uses pandas frequency strings.

        absolute : bool, optional
            If True, multiplies sensitivity indices by the variance of the aggregated
            output at each timestep. This gives absolute contributions to output
            variability, not normalized indices.

        Returns:
        -------
        None
            Stores results in `self.sensitivity_results` (static)
            or `self.sensitivity_dynamic_results` (dynamic).
        """
        self.static = freq is None  # static becomes True if freq is given
        self.sensitivity_dynamic_results = {}  # here so that it's emptied when reran for new frequencies

        if sensitivity_method_kwargs is None:
            sensitivity_method_kwargs = {}

        if agg_method_kwarg is None:
            agg_method_kwarg = {}

        if not self.sample_results:
            raise ValueError(
                "No simulation results were found. Use evaluate() method to run "
                "the model."
            )

        if agg_method_kwarg is None:
            agg_method_kwarg = {}

        if indicator not in self.sample_results[0][2].columns:
            raise ValueError("Specified indicator not in computed outputs")

        analyser = METHOD_SAMPLER_DICT[self.method]["method"]

        if freq is not None:
            aggregated_list = []
            for result in self.sample_results:
                series = result[2][indicator]
                grouped = series.groupby(pd.Grouper(freq=freq))

                if reference_df is not None:
                    ref_grouped = reference_df.groupby(pd.Grouper(freq=freq))
                    vals = [
                        agg_method(g, r, **agg_method_kwarg)
                        for (_, g), (_, r) in zip(grouped, ref_grouped)
                    ]
                else:
                    vals = [agg_method(g, **agg_method_kwarg) for _, g in grouped]

                aggregated_list.append(pd.Series(vals, index=[i for i, _ in grouped]))

            index = aggregated_list[0].index
            numpy_res = np.array([s.values for s in aggregated_list]).T

            for t_idx, values in zip(index, numpy_res):
                if self.method.value in ["SOBOL", "FAST"]:
                    res = analyser.analyze(
                        problem=self._salib_problem,
                        Y=values,
                        **sensitivity_method_kwargs,
                    )
                else:
                    res = analyser.analyze(
                        problem=self._salib_problem,
                        X=self.sample.to_numpy(),
                        Y=values,
                        **sensitivity_method_kwargs,
                    )

                res["names"] = self._salib_problem["names"]
                if absolute:
                    res["_absolute"] = True
                self.sensitivity_dynamic_results[t_idx] = res

            # Option "absolute"
            if absolute:
                var_t = np.var(numpy_res, axis=1)

                for key, res, var in zip(
                    self.sensitivity_dynamic_results.keys(),
                    self.sensitivity_dynamic_results.values(),
                    var_t,
                ):
                    if self.method.value in ["SOBOL", "FAST"]:
                        for k in list(res.keys()):
                            if k == "names":
                                continue
                            try:
                                res[k] = np.array(res[k], dtype=float) * var
                            except (TypeError, ValueError):
                                continue  # ignore non-numeric fields
                    else:
                        if "names" in res:
                            del res["names"]
                        for k in list(res.keys()):
                            try:
                                res[k] = np.array(res[k], dtype=float) * var
                            except (TypeError, ValueError):
                                continue

        else:
            results_2d = pd.concat(
                [s_res[2][indicator] for s_res in self.sample_results], axis=1
            )
            if reference_df is not None:
                reference_df_duplicated = pd.concat(
                    [reference_df] * len(results_2d.columns), axis=1
                )
            else:
                reference_df_duplicated = None
            y_array = np.array(
                aggregate_time_series(
                    results_2d, agg_method, agg_method_kwarg, reference_df_duplicated
                )
            )

            if self.method.value in ["SOBOL", "FAST"]:
                self.sensitivity_results = analyser.analyze(
                    problem=self._salib_problem, Y=y_array, **sensitivity_method_kwargs
                )
            elif self.method.value in ["MORRIS", "RBD_FAST"]:
                self.sensitivity_results = analyser.analyze(
                    problem=self._salib_problem,
                    X=self.sample.to_numpy(),
                    Y=y_array,
                    **sensitivity_method_kwargs,
                )

    def calculate_sensitivity_indicators(self):
        """
        Returns sensitivity indicators based on the method used.

        Returns:
        - dict: A dictionary containing the calculated sensitivity summary.

        If the method is FAST or RBD_FAST:
            - sum_st (float): Sum of the sensitivity indices (ST or S1).
            - mean_conf (float): Mean confidence level of the sensitivity indices.

        If the method is SOBOL:
            - sum_st (float): Sum of the first-order sensitivity indices (ST).
            - mean_conf (float): Mean confidence level of the first-order
            sensitivity indices (ST_conf).
            - sum_s1 (float): Sum of the first-order sensitivity indices (S1).
            - mean_conf1 (float): Mean confidence level of the first-order
            sensitivity indices (S1_conf).
            - sum_s2 (float): Sum of the second-order sensitivity indices (S2).
            - mean_conf2 (float): Mean confidence level of the second-order
            sensitivity indices (S2_conf).

        If the method is MORRIS:
            - euclidian distance (array-like): Euclidian distance
             between mu_star and sigma.
            Mu_star represents the average elementary effects of the outputs
            with respect to each input factor, while sigma measures the total
            variability induced by each input factor.
            - normalized euclidian distance (array-like): Normalized euclidian
             distance between mu_star and sigma. The normalized euclidian distance
             is the euclidian distance divided by its maximum value, providing a
             relative measure of the variability induced by each input factor.

        """

        def average_dynamic_index(index_name):
            df = pd.DataFrame(
                {
                    t: res[index_name]
                    for t, res in self.sensitivity_dynamic_results.items()
                    if index_name in res
                }
            ).T
            return df.mean()

        if not self.static:
            if not self.sensitivity_dynamic_results:
                raise ValueError("No dynamic sensitivity results found.")

            if self.method in [Method.SOBOL, Method.FAST]:
                st_mean = average_dynamic_index("ST")
                s1_mean = (
                    average_dynamic_index("S1")
                    if "S1" in next(iter(self.sensitivity_dynamic_results.values()))
                    else None
                )
                return {
                    "ST": st_mean,
                    "S1": s1_mean,
                }

            elif self.method == Method.RBD_FAST:
                s1_mean = average_dynamic_index("S1")
                return {"S1": s1_mean}

            elif self.method == Method.MORRIS:
                raise NotImplementedError(
                    "Dynamic MORRIS indicators not supported yet."
                )

            else:
                raise ValueError("Unknown method for dynamic sensitivity.")

        if self.sensitivity_results is None:
            raise ValueError("No static sensitivity results available.")

        if self.method == Method.FAST:
            sum_st = sum(self.sensitivity_results["ST"])
            conf = self.sensitivity_results["ST_conf"]
            mean_conf = sum(conf) / len(conf)

            return {"sum_st": sum_st, "mean_conf": mean_conf}

        elif self.method == Method.RBD_FAST:
            sum_st = sum(self.sensitivity_results["S1"])
            conf = self.sensitivity_results["S1_conf"]
            mean_conf = sum(conf) / len(conf)

            return {"sum_st": sum_st, "mean_conf": mean_conf}

        elif self.method == Method.SOBOL:
            sum_st = self.sensitivity_results.to_df()[0].sum().loc["ST"]
            mean_conf = self.sensitivity_results.to_df()[0].mean().loc["ST_conf"]
            sum_s1 = self.sensitivity_results.to_df()[1].sum().loc["S1"]
            mean_conf1 = self.sensitivity_results.to_df()[1].mean().loc["S1_conf"]
            sum_s2 = self.sensitivity_results.to_df()[2].sum().loc["S2"]
            mean_conf2 = self.sensitivity_results.to_df()[2].mean().loc["S2_conf"]

            return {
                "sum_st": sum_st,
                "mean_conf": mean_conf,
                "sum_s1": sum_s1,
                "mean_conf1": mean_conf1,
                "sum_s2": sum_s2,
                "mean_conf2": mean_conf2,
            }

        elif self.method == Method.MORRIS:
            morris_res = self.sensitivity_results
            morris_res["euclidian distance"] = np.sqrt(
                morris_res["mu_star"] ** 2 + morris_res["sigma"] ** 2
            )
            morris_res["normalized euclidian distance"] = (
                morris_res["euclidian distance"]
                / morris_res["euclidian distance"].max()
            )

            return {
                "euclidian distance": morris_res["euclidian distance"].data,
                "normalized euclidian distance": morris_res[
                    "normalized euclidian distance"
                ].data,
            }

        else:
            raise ValueError("Invalid method")


def plot_sobol_st_bar(salib_res, normalize_dynamic=False):
    """
    Plot Sobol total sensitivity indices (ST) as a bar chart or a dynamic line chart.

    This function automatically detects whether the input is a static or dynamic result,
    and adjusts the plot accordingly. For dynamic results, it also detects if the indices
    are absolute (i.e., multiplied by output variance) and adapts the y-axis label.

    Parameters
    ----------
    salib_res : SALib.analyze._results.SobolResults | dict
        The result object returned by a SALib Sobol analysis.
        - For static analysis: a SobolResults object (e.g., `sanalysis.sensitivity_results`)
        - For dynamic analysis: a dictionary of time-indexed results (e.g., `sanalysis.sensitivity_dynamic_results`),
          where each value must contain keys "ST" and "names", and optionally "_absolute" to indicate absolute mode.
    normalize_dynamic : bool, optional
        If True, normalizes dynamic results to percentages and plots a cumulative graph (0 to 1).
    """

    if isinstance(salib_res, dict) and isinstance(next(iter(salib_res.values())), dict):
        try:
            df_to_plot = pd.DataFrame(
                {
                    t: pd.Series(res["ST"], index=res["names"])
                    for t, res in salib_res.items()
                }
            ).T
        except KeyError:
            raise ValueError(
                "ST index not found in dynamic results. Ensure 'ST' and 'names' are present."
            )
        absolute = "_absolute" in next(iter(salib_res.values()))

        if normalize_dynamic:
            df_to_plot = df_to_plot.div(df_to_plot.sum(axis=1), axis=0)

        df_to_plot.index.name = "Time"
        df_to_plot.columns.name = "Parameter"

        fig = go.Figure()
        for param in df_to_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_to_plot.index,
                    y=df_to_plot[param],
                    name=param,
                    mode="lines",
                    stackgroup="one" if normalize_dynamic else None,
                )
            )

        yaxis_title = (
            "Cumulative percentage [0-1]"
            if normalize_dynamic
            else (
                "Absolute Sobol contribution"
                if absolute
                else "Sobol total index value [0-1]"
            )
        )

        fig.update_layout(
            title="Sobol ST indices (dynamic)",
            xaxis_title="Time",
            yaxis_title=yaxis_title,
        )
        return fig

    sobol_ind = salib_res.to_df()[0]
    sobol_ind.sort_values(by="ST", ascending=True, inplace=True)

    absolute = sobol_ind.ST.max() > 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sobol_ind.index,
            y=sobol_ind.ST,
            name="Sobol Total Indices",
            marker_color="orange",
            error_y=dict(type="data", array=sobol_ind.ST_conf.to_numpy()),
            yaxis="y1",
        )
    )

    fig.update_layout(
        title="Sobol Total indices",
        xaxis_title="Parameters",
        yaxis_title="Absolute Sobol contribution"
        if absolute
        else "Sobol total index value [0-1]",
    )

    return fig


def plot_morris_st_bar(salib_res, distance_metric="normalized"):
    if distance_metric not in ["absolute", "normalized"]:
        raise ValueError("Distance metric must be either 'absolute'" " or 'normalized'")

    salib_res = salib_res.to_df()
    if distance_metric == "absolute":
        dist = "euclidian distance"
    else:
        dist = "normalized euclidian distance"
    salib_res.sort_values(by=dist, ascending=True, inplace=True)

    if distance_metric == "absolute":
        distance_values = salib_res["euclidian distance"]
        distance_conf = None
        title = "Morris Sensitivity Analysis - euclidian Distance"
    else:
        distance_values = salib_res["normalized euclidian distance"]
        distance_conf = None
        title = "Morris Sensitivity Analysis - Normalized euclidian Distance"

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=salib_res.index,
            y=distance_values,
            name=distance_metric.capitalize(),
            marker_color="orange",
            error_y=dict(
                type="data",
                array=distance_conf.to_numpy() if distance_conf is not None else None,
            ),
            yaxis="y1",
        )
    )

    figure.update_layout(
        title=title,
        xaxis_title="Parameters",
        yaxis_title=f"{distance_metric.capitalize()} (d)",
    )

    return figure


def plot_morris_scatter(
    salib_res,
    title: str = None,
    unit: str = "",
    scaler: int = 100,
    autosize: bool = True,
):
    """
    This function generates a scatter plot for Morris sensitivity analysis results.
    It displays the mean of elementary effects (μ*) on the x-axis and the standard
    deviation of elementary effects (σ) on the y-axis.
    Marker sizes and colors represent the 'distance' to the origin.

    Parameters:
    - salib_res (pandas DataFrame): DataFrame containing sensitivity analysis results.
    - title (str, optional): Title for the plot. If not provided, a default title
    is used.
    - unit (str, optional): Unit for the axes labels.
    - scaler (int, optional): A scaling factor for marker sizes in the plot.
    - autosize (bool, optional): Whether to automatically adjust the y-axis range.

    """
    morris_res = salib_res.to_df()
    morris_res["distance"] = np.sqrt(morris_res.mu_star**2 + morris_res.sigma**2)
    morris_res["dimless_distance"] = morris_res.distance / morris_res.distance.max()

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=morris_res.mu_star,
            y=morris_res.sigma,
            name="Morris index",
            mode="markers+text",
            text=list(morris_res.index),
            textposition="top center",
            marker=dict(
                size=morris_res.dimless_distance * scaler,
                color=np.arange(morris_res.shape[0]),
            ),
            error_x=dict(
                type="data",  # value of error bar given in data coordinates
                array=morris_res.mu_star_conf,
                color="#696969",
                visible=True,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([0, morris_res.mu_star.max() * 1.1]),
            y=np.array([0, 0.1 * morris_res.mu_star.max() * 1.1]),
            name="linear_lim",
            mode="lines",
            line=dict(color="grey", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([0, morris_res.mu_star.max() * 1.1]),
            y=np.array([0, 0.5 * morris_res.mu_star.max() * 1.1]),
            name="Monotonic limit",
            mode="lines",
            line=dict(color="grey", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([0, morris_res.mu_star.max() * 1.1]),
            y=np.array([0, 1 * morris_res.mu_star.max() * 1.1]),
            name="Non linear limit",
            mode="lines",
            line=dict(color="grey", dash="dashdot"),
        )
    )

    # Edit the layout
    if title is not None:
        title = title
    else:
        title = "Morris Sensitivity Analysis"

    if autosize:
        y_lim = [-morris_res.sigma.max() * 0.1, morris_res.sigma.max() * 1.5]
    else:
        y_lim = [-morris_res.sigma.max() * 0.1, morris_res.mu_star.max() * 1.1]

    x_label = f"Absolute mean of elementary effects μ* [{unit}]"
    y_label = f"Standard deviation of elementary effects σ [{unit}]"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_lim,
    )

    return fig


def plot_sample(
    sample_results,
    indicator=None,
    ref=None,
    title=None,
    y_label=None,
    x_label=None,
    alpha=0.5,
    loc=None,
    show_legends=False,
    html_file_path=None,
):
    """
    Plot sample results for a given indicator.

    Parameters:
    - sample_results (list): List of tuples, each containing parameters,
    simulation options, and results.
    - indicator (str): The model output indicator to plot.
    - ref (pd.Series or pd.DataFrame, optional): Reference data for comparison.
    - title (str, optional): Title for the plot.
    - y_label (str, optional): Label for the y-axis.
    - x_label (str, optional): Label for the x-axis.
    - alpha (float, optional): Opacity of the markers.
    - loc (tuple, optional): Tuple specifying the time range to plot,
        e.g., (start_time, end_time).
    - show_legends (bool, optional): Whether to display legends with parameter values.
    - html_file_path (str, optional): If provided, save the plot as an HTML file.
    """
    if indicator is None:
        raise ValueError("Please specify at least the model output name as 'indicator'")

    fig = go.Figure()

    for _, result in enumerate(sample_results):
        parameters, simulation_options, simulation_results = result

        to_plot_indicator = simulation_results[indicator]

        if loc is not None:
            to_plot_indicator = to_plot_indicator.loc[loc[0] : loc[1]]

        rounded_parameters = {key: round(value, 2) for key, value in parameters.items()}
        parameter_names = [
            param.split(".")[-1] if "." in param else param
            for param in rounded_parameters.keys()
        ]
        legend_str = ", ".join(
            [
                f"{name}: {value}"
                for name, value in zip(parameter_names, rounded_parameters.values())
            ]
        )

        fig.add_trace(
            go.Scattergl(
                name=legend_str if show_legends else "Simulations",
                mode="markers",
                x=to_plot_indicator.index,
                y=np.array(to_plot_indicator),
                marker=dict(
                    color=f"rgba(135, 135, 135, {alpha})",
                ),
            )
        )

    if ref is not None:
        if loc is not None:
            to_plot_ref = ref.loc[loc[0] : loc[1]]
        else:
            to_plot_ref = ref

        fig.add_trace(
            go.Scattergl(
                name="Reference",
                mode="lines",
                x=to_plot_ref.index,
                y=np.array(to_plot_ref),
                marker=dict(
                    color="red",
                ),
            )
        )

    if title is not None:
        fig.update_layout(title=title)

    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)

    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)

    fig.update_layout(showlegend=show_legends)

    if html_file_path:
        pio.write_html(fig, html_file_path)
    return fig


def plot_pcp(
    sample_results,
    parameters,
    indicators,
    aggregation_method=np.mean,
    bounds=False,
    html_file_path=None,
    plot_unselected=True,
):
    """
    Plots a parallel coordinate plot for sensitivity analysis results.

    Parameters
    ----------
    sample_results : list
        A list of results from sensitivity analysis simulations. Each element
        in the list should be a tuple containing three components:
        1. A dictionary representing the parameter values used in the simulation.
        2. A dictionary representing simulation options.
        3. A DataFrame containing the results of the simulation.

    parameters : list
        A list of dictionaries, where each dictionary represents a parameter and
        contains its name, interval, and type.

    indicators : list
        A list of strings representing the indicators (columns) in the simulation
        results DataFrame to be plotted.

    aggregation_method : function, optional
        The aggregation method used to summarize indicator values across multiple
        simulations. Default is numpy.mean.

    bounds : bool, optional
        If True, includes the bounds of the parameters in the plot. Default is False.

    html_file_path : str, optional
        If provided, save the plot as an HTML file.


    Returns
    -------
    None
        The function displays the parallel coordinate plot using Plotly.

    Notes
    -----
    The parallel coordinate plot visualizes the relationships between parameters
    and indicators across multiple simulations. Each line in the plot represents a
    simulation, and the position of each line along the axes corresponds to the
    parameter values. The color of the lines can be determined by an indicator.
    """

    data_dict = {
        param[Parameter.NAME]: np.array(
            [res[0][param[Parameter.NAME]] for res in sample_results]
        )
        for param in parameters
    }

    if isinstance(indicators, str):
        indicators = [indicators]

    for indicator in indicators:
        data_dict[indicator] = np.array(
            [aggregation_method(res[2][indicator]) for res in sample_results]
        )

    colorby = indicators[0] if indicators else None

    plot_parcoord(
        data_dict=data_dict,
        bounds=bounds,
        parameters=parameters,
        colorby=colorby,
        obj_res=np.array([res[2][indicators] for res in sample_results]),
        html_file_path=html_file_path,
        plot_unselected=plot_unselected,
    )


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
    >>> param_list = [
    ...     {
    ...         Parameter.NAME: "param1",
    ...         Parameter.INTERVAL: [0, 1],
    ...         Parameter.INIT_VALUE: 0.5,
    ...     }
    ... ]
    >>> indicators_config = {"res": (mean_squared_error, reference_series)}
    >>> obj_func = ObjectiveFunction(
    ...     model, simulation_options, param_list, indicators_config
    ... )
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
