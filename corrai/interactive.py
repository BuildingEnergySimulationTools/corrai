from dash import Dash, dcc, Input, Output, Patch, html
import dash_bootstrap_components as dbc

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

    # Set title and axis labels
    if title:
        fig.update_layout(title=title)
    if x_label:
        fig.update_layout(xaxis_title=x_label)
    if y_label:
        fig.update_layout(yaxis_title=y_label)

    # Optimize layout for performance
    fig.update_layout(
        showlegend=show_legends,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def interactive_sample_pcp_lines(
    sample_results, target, agg_method, reference, color_by, down_sample=0, debug=False
):
    df_pcp = pd.concat(
        [
            pd.DataFrame([sim[0] for sim in sample_results]),
            aggregate_sample_results(
                sample_results=sample_results, agg_method=agg_method
            ),
        ],
        axis=1,
    )

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
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            html.H4("Filtering Dash AG Grid with Parallel Coordinates"),
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
            )
        return sample_ts_plot(
            data=df_lines,
            selected_index=df_pcp.index,
            reference_ts=reference,
            down_sample=down_sample,
        )

    app.run_server(debug=debug)
