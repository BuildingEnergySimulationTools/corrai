import webbrowser

from dash import Dash, dcc, Input, Output, Patch, html
import dash_bootstrap_components as dbc
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from corrai.base.utils import aggregate_sample_results


def parallel_coordinates_plot(df, color_column, plot_unselected=True, title=None):
    """
    Creates a parallel coordinates plot with specified color coding and unselected line styling.

    Parameters:
    - df (pd.DataFrame): The data frame containing the data to plot.
    - color_column (str): The column name to use for line coloring.
    - plot_unselected (bool): Whether to display unselected lines with reduced opacity.
    - title (str): The title of the plot.

    Returns:
    - fig (go.Figure): The Plotly figure object.
    """
    color_data = df[color_column]
    color_palette = px.colors.diverging.Tealrose
    color_min = color_data.min()
    color_max = color_data.max()

    # Define unselected line properties
    unselected = dict(line=dict(color="grey", opacity=0.5 if plot_unselected else 0))

    # Create the Parcoords figure
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_data,
                colorscale=color_palette,
                showscale=True,
                cmin=color_min,
                cmax=color_max,
            ),
            dimensions=[
                {
                    "range": [df[col].min(), df[col].max()],
                    "label": col,
                    "values": df[col],
                }
                for col in df.columns
            ],
            unselected=unselected,
        )
    )

    # Set the title if provided
    if title:
        fig.update_layout(title=title)

    return fig


def sample_ts_plot(
    data,
    selected_index,
    reference_ts: pd.Series = None,
    alpha=0.5,
    title=None,
    x_label=None,
    y_label=None,
    show_legends=False,
    down_sample=None,
):
    if down_sample > 1:
        data = data.iloc[::down_sample, :]

    not_selected = data[[col for col in data.columns if col not in selected_index]]

    fig = go.Figure()

    for col in not_selected:
        fig.add_trace(
            go.Scattergl(
                name="Unselected",
                mode="markers",
                x=not_selected.index,
                y=np.array(not_selected[col]),
                marker=dict(
                    color=f"rgba(135, 135, 135, {alpha})",
                ),
            )
        )

    for col in selected_index:
        (
            fig.add_trace(
                go.Scattergl(
                    name="Selected",
                    mode="lines",
                    x=not_selected.index,
                    y=np.array(data[col]),
                    marker=dict(
                        color="darkorange",
                    ),
                )
            ),
        )

    if reference_ts is not None:
        fig.add_trace(
            go.Scattergl(
                name="Reference",
                mode="lines",
                x=reference_ts.index,
                y=np.array(reference_ts),
                marker=dict(
                    color="red",
                ),
            )
        )

    # Optimize layout for performance
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        showlegend=show_legends,
        template="plotly_white",
        margin=dict(l=0, r=0, b=0),
    )

    return fig


def interactive_sample_pcp_lines(
    sample_results: [[dict, dict, pd.DataFrame]],
    target: str = None,
    agg_method: dict[
        str, [tuple[Callable, str, pd.Series] | tuple[str, Callable]]
    ] = None,
    reference: pd.Series = None,
    color_by=None,
    down_sample: int = 0,
    ts_y_label: str = None,
    debug=False,
    open_web_browser=True,
):
    """
    Run Dash app to display interactive plot. A parallel plot is used to filter
    sample results holding time series.
    This app is mainly designed to help for model calibration using reference and
    computing indicators through aggregation methods.

        Parameters:
    -----------
    sample_results : List[Tuple[dict, dict, pd.DataFrame]]
        A list of lists where each list contains:
        - A dictionary of parameter
        - A dictionary of simulation options
        - A pandas DataFrame containing the results.

    target : str Optional:
        Name of the simulation output indicator in the results DataFrame to
        plot in the timeseries plot. Default will be the first result

    agg_method : Dict[str, Union[
        Tuple[Callable[[pd.Series, pd.Series], float], str, pd.Series],
        Tuple[Callable[[pd.Series], float], str]
        ]]
        A dictionary specifying the aggregation methods. The keys are the names of the
        aggregated columns. The values are either:
        - A tuple with three elements: a callable taking two pandas Series and returning
          a float, a string specifying the column name in the results, and a reference
          pandas Series.
        - A tuple with two elements: a callable taking one pandas Series and returning
          a float, and a string specifying the column name in the DataFrame.

        exemple of agg_method :
        from corrai.metrics import cv_rmse

        agg_method = {
            "cv_rmse_tin": (cv_rmse, "Tin", tin_measure_series),
            "mean_power": (np.mean, "Power")
        }

    reference : pd.Series Optional
        A reference time series to be plot in red color on the timeseries plot.
        Do not interfere with agg_method calculations

    color_by : str Optional
        The parameter or the aggregated indicator to be used as a reference for
        the parallel coordinate plot color map

    down_sample : int Optional
        Down sample the timeseries by removing values before plotting.
        This can be essential for large sample or long timeseries.
        Increase down sampling if lag or crashes happen.

    ts_y_label str Optional
        y label on the timeseries plot

    debug bool Optional
        Activate Dash app debug mode

    open_web_browser bool Optional
        Automatically open browser and connect to dash app
    """

    df_pcp = pd.DataFrame([sim[0] for sim in sample_results])
    if agg_method is not None:
        aggregated_results = aggregate_sample_results(
            sample_results=sample_results, agg_method=agg_method
        )
        df_pcp = pd.concat([df_pcp, aggregated_results], axis=1)

    # Default values
    color_by = df_pcp.columns[0] if color_by is None else color_by
    target = sample_results[0][2].columns[0] if target is None else target

    df_lines = pd.concat([res[2][target] for res in sample_results], axis=1)
    df_lines.columns = df_pcp.index

    fig_pcp = parallel_coordinates_plot(
        df=df_pcp, color_column=color_by, plot_unselected=False
    )

    fig_lines = sample_ts_plot(
        data=df_lines,
        selected_index=[],
        reference_ts=reference,
        down_sample=down_sample,
        title=target,
        y_label=ts_y_label,
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            html.H4("Filtering timeseries sample with Parallel Coordinates"),
            dcc.Graph(id="lines-graph", figure=fig_lines),
            dcc.Graph(id="my-graph", figure=fig_pcp),
            dcc.Store(id="activefilters", data={}),
        ]
    )

    @app.callback(
        Output("activefilters", "data"),
        Input("my-graph", "restyleData"),
    )
    def updateFilters(data):
        if data:
            key = list(data[0].keys())[0]
            col = df_pcp.columns[int(key.split("[")[1].split("]")[0])]
            newData = Patch()
            newData[col] = data[0][key]
            return newData
        return {}

    @app.callback(
        Output("lines-graph", "figure"),
        Input("activefilters", "data"),
    )
    def udpate_table(data):
        if data:
            dff = df_pcp.copy()
            for col in data:
                if data[col]:
                    rng = data[col][0]
                    if isinstance(rng[0], list):
                        # if multiple choices combine df
                        dff3 = pd.DataFrame(columns=df_pcp.columns)
                        for i in rng:
                            dff2 = dff[dff[col].between(i[0], i[1])]
                            dff3 = pd.concat([dff3, dff2])
                        dff = dff3
                    else:
                        # if one choice
                        dff = dff[dff[col].between(rng[0], rng[1])]
            return sample_ts_plot(
                data=df_lines,
                selected_index=dff.index,
                reference_ts=reference,
                down_sample=down_sample,
                title=target,
                y_label=ts_y_label,
            )
        return sample_ts_plot(
            data=df_lines,
            selected_index=df_pcp.index,
            reference_ts=reference,
            down_sample=down_sample,
            title=target,
            y_label=ts_y_label,
        )

    app.run_server(debug=debug)

    if open_web_browser:
        webbrowser.open("http://127.0.0.1:8050/")
