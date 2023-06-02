import numpy as np
import pandas as pd
import json
from collections import defaultdict
from corrai.utils import check_datetime_index
import plotly.graph_objects as go
import datetime as dt


def missing_values_dict(df):
    return {
        "Number_of_missing": df.count(),
        "Percent_of_missing": (1 - df.count() / df.shape[0]) * 100,
    }


def interpolate_color(color1, color2, t):
    """Interpolate between two colors based on a parameter t"""
    r = int((1 - t) * int(color1[1:3], 16) + t * int(color2[1:3], 16))
    g = int((1 - t) * int(color1[3:5], 16) + t * int(color2[3:5], 16))
    b = int((1 - t) * int(color1[5:7], 16) + t * int(color2[5:7], 16))
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_color(color, factor):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    darkened_color = f"#{r:02X}{g:02X}{b:02X}"
    return darkened_color


def check_config_dict(config_dict):
    """
    Check if the input dictionary follows the expected structure and values.

    Parameters:
    -----------
    config_dict : dict
        A dictionary with two keys: "data_type_dict" and "corr_dict".

        "data_type_dict" is a dictionary that maps categories to a list of
        column names.

        "corr_dict" is a dictionary that maps categories to a dictionary of
        correction methods and their values.

    Raises:
    -------
    ValueError:
        If the input dictionary does not follow the expected structure and
        values.
    """
    if not list(config_dict.keys()) == ["data_type_dict", "corr_dict"]:
        raise ValueError("Invalid data_type or corr_dict")

    categories = list(config_dict["data_type_dict"].keys())
    corr_dict_type = list(config_dict["corr_dict"].keys())
    for cat in categories:
        if cat not in corr_dict_type:
            raise ValueError("Type present in data_type " "is missing in corr_dict")

    for tpe in config_dict["corr_dict"].keys():
        to_test = list(config_dict["corr_dict"][tpe].keys())
        for it in to_test:
            loc_corr = config_dict["corr_dict"][tpe][it]

            if it == "minmax":
                if list(loc_corr.keys()) != ["upper", "lower"]:
                    raise ValueError(f"Invalid configuration for {tpe} minmax")

            elif it == "derivative":
                for k in list(loc_corr.keys()):
                    if k not in ["upper_rate", "lower_rate"]:
                        raise ValueError(
                            f"Invalid configuration " f"for {tpe} derivative"
                        )
            elif it == "fill_nan":
                for elmt in loc_corr:
                    if elmt not in ["linear_interpolation", "bfill", "ffill"]:
                        raise ValueError(
                            f"Invalid configuration " f"for {tpe} fill_nan"
                        )
            elif it == "resample":
                if loc_corr not in ["mean", "sum"]:
                    raise ValueError(f"Invalid configuration " f"for {tpe} resample")


def select_data(df, cols=None, begin=None, end=None):
    if cols is None:
        cols = df.columns

    if begin is None:
        begin = df.index[0]

    if end is None:
        end = df.index[-1]

    return df.loc[begin:end, cols]


def find_gaps(data, cols=None, timestep=None):
    """
    Find gaps in time series data. Find individual columns gap and combined gap
    for all columns.

    Parameters:
    -----------
        data (pandas.DataFrame or pandas.Series): The time series data to check
            for gaps.

        cols (list, optional): The columns to check for gaps. Defaults to None,
            in which case all columns are checked.

        timestep (str or pandas.Timedelta, optional): The time step of the
            data. Can be either a string representation of a time period
            (e.g., '1H' for hourly data), or a pandas.Timedelta object.
            Defaults to None, in which case the time step is automatically
            determined.

    Raises:
    -------

        ValueError: If cols or timestep are invalid.

    Returns:
    --------
        dict: A dictionary containing the duration of the gaps for each
        specified column, as well as the overall combination of columns.
    """

    check_datetime_index(data)
    if not cols:
        cols = data.columns

    if not timestep:
        timestep = get_mean_timestep(data)

    # Aggregate in a single columns to know overall quality
    df = data.copy()
    df = ~df.isnull()
    df["combination"] = df.all(axis=1)

    # Index are added at the beginning and at the end to account for
    # missing values and each side of the dataset
    first_index = df.index[0] - (df.index[1] - df.index[0])
    last_index = df.index[-1] - (df.index[-2] - df.index[-1])

    df.loc[first_index] = np.ones(df.shape[1], dtype=bool)
    df.loc[last_index] = np.ones(df.shape[1], dtype=bool)
    df.sort_index(inplace=True)

    # Compute gaps duration
    res = {}
    for col in list(cols) + ["combination"]:
        time_der = df[col].loc[df[col]].index.to_series().diff()
        res[col] = time_der[time_der > timestep]

    return res


def gaps_describe(df_in, cols=None, timestep=None):
    res_find_gaps = find_gaps(df_in, cols, timestep)

    return pd.DataFrame({k: val.describe() for k, val in res_find_gaps.items()})


def get_mean_timestep(df):
    return df.index.to_frame().diff().mean()[0]


def add_scatter_and_gaps(
        figure, series, gap_series, color_rgb, alpha, y_min, y_max, yaxis
):
    figure.add_trace(
        go.Scattergl(
            x=series.index,
            y=series.to_numpy().flatten(),
            mode="lines+markers",
            name=series.name,
            yaxis=yaxis
            # line=dict(color=f'rgb{color_rgb}')
        )
    )

    for t_idx, gap in gap_series.items():
        figure.add_trace(
            go.Scattergl(
                x=[t_idx - gap, t_idx - gap, t_idx, t_idx],
                y=[y_min, y_max, y_max, y_min],
                mode="none",
                fill="toself",
                showlegend=False,
                fillcolor=f"rgba({color_rgb[0]}, {color_rgb[1]},"
                          f" {color_rgb[2]} , {alpha})",
                yaxis=yaxis,
            )
        )


class MeasuredDats:
    def __init__(
            self,
            data,
            data_type_dict=None,
            corr_dict=None,
            config_file_path=None,
            gaps_timedelta=None,
    ):
        """
        A class for handling time-series data with missing values.

        Parameters:
        -----------
        data : pandas.DataFrame
            The input data to be processed.

        data_type_dict : dict, optional
            A dictionary that maps data type categories to the 'data' columns
            that belong to that category. Default is None.

        corr_dict : dict, optional
            A dictionary that stores the correction method and parameters for each
            'data' type category. 'corr_dict" keys must match 'data_type_dict'
            keys. Default is None.

        config_file_path : str, optional
            The path to the configuration file.
            If specified, the data type dictionary and correction dictionary
            will be loaded from the file. Default is None.

        gaps_timedelta : pandas.Timedelta, optional
            The maximum allowed data gap size for each data series. Gaps
            smaller thn 'gaps_timedelta' will not be detected. They will be
            corrected during gaps filling processes. If None, the
            'gaps_timedelta' timestep is estimated automatically from the data
            index mean timestep.

        Attributes:
        ----------
        data : pandas.DataFrame
            The original input data.

        corrected_data : pandas.DataFrame
            The data after the correction process.

        data_type_dict : dict
            A dictionary that maps data type categories to the columns that
            belong to that category.

        corr_dict : dict
            A dictionary that stores the correction method and parameters for
            each data type category.

        correction_journal : dict
            A dictionary that stores the history of the correction process.

        gaps_timedelta : pandas.Timedelta
            The maximum allowed gap size for each data series.

        resample_func_dict : dict
            A dictionary that maps resampling functions to the corresponding method
            string.

        Methods:
        -------
        write_config_file(file_path)
            Writes the data type dictionary and correction dictionary to a
            JSON file.

        read_config_file(file_path)
            Reads the data type dictionary and correction dictionary from a J
            SON file.

        add_time_series(time_series, data_type, data_corr_dict=None)
            Adds a new time series to the data set and updates the data type
            and correction dictionaries accordingly.

        auto_correct()
            Runs the correction process automatically.

        remove_anomalies()
            Removes the anomalies in the data set using the correction methods
            specified  in the correction dictionary.

        fill_nan()
            Fills the missing values in the data set using the correction
            methods specified in the correction dictionary.

        resample(timestep=None)
            Resamples the data set to a specified timestep. If None, the
            timestep is the data index mean timestep.

        plot_gaps(cols=None, begin=None, end=None,
            gaps_timestep=dt.timedelta(hours=5), title="Gaps plot",
            raw_data=False, color_rgb=(243, 132, 48),  alpha=0.5):
            cols (list, optional): List of column names to plot. If not
                provided, all columns will be plotted. Default is None.
            begin (str, optional): String specifying the start date for the
                data selection. If not provided, the entire dataset will be
                used. Default is None.
            end (str, optional): String specifying the end date for the data
                selection. If not provided, the entire dataset will be used.
                Default is None.
            gaps_timestep (timedelta, optional): Minimum duration between data
                points to consider a gap. Default is 5 hours.
            title (str, optional): Title of the plot. Default is "Gaps plot".
            raw_data (bool, optional): If True, plot the raw data without gap
                correction. If False, plot the gap-corrected data.
                Default is False.
            color_rgb (tuple of int, optional): RGB color of the gaps.
                Default is (243, 132, 48).
            alpha (float, optional): Opacity of the gaps. Default is 0.5.

        plot(cols=None, title="Correction plot", plot_raw=False,
             begin=None, end=None)
            Generate a plot comparing the original and corrected values of the given
            columns over the specified time range.
            cols : list of str, optional
                The names of the columns to plot. If None (default), all
                columns are plotted.
            title : str, optional
                The title of the plot. Defaults to "Correction plot".
            plot_raw : bool, optional
                If True, plot the raw values in addition to the corrected
                values. Defaults to False.
            begin : str or datetime-like, optional
                A string or datetime-like object specifying the start of the
                time range to plot. If None (default), plot all data.
            end : str or datetime-like, optional
                A string or datetime-like object specifying the end of the
                time range to plot. If None (default), plot all data.
        """

        self.data = data.copy()
        self.corrected_data = None

        if config_file_path is None:
            self.data_type_dict = data_type_dict
            self.corr_dict = corr_dict
            check_config_dict(
                {"data_type_dict": data_type_dict, "corr_dict": corr_dict}
            )
        else:
            self.read_config_file(config_file_path)

        self.correction_journal = {
            "Entries": data.shape[0],
            "Init": missing_values_dict(data),
        }
        if gaps_timedelta is None:
            self.gaps_timedelta = get_mean_timestep(self.data)
        else:
            self.gaps_timedelta = gaps_timedelta

        self.resample_func_dict = {"mean": np.mean, "sum": np.sum}

    @property
    def columns(self):
        return self.data.columns

    def write_config_file(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            to_dump = {
                "data_type_dict": self.data_type_dict,
                "corr_dict": self.corr_dict,
            }
            json.dump(to_dump, f, ensure_ascii=False, indent=4)

    def read_config_file(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        check_config_dict(config_dict)
        self.data_type_dict = config_dict["data_type_dict"]
        self.corr_dict = config_dict["corr_dict"]

    def add_time_series(self, time_series, data_type, data_corr_dict=None):
        check_datetime_index(time_series)
        if data_corr_dict is None:
            data_corr_dict = {}

        if data_type in self.data_type_dict.keys():
            self.data_type_dict[data_type] += list(time_series.columns)
        else:
            self.data_type_dict[data_type] = list(time_series.columns)
            self.corr_dict[data_type] = data_corr_dict

        check_config_dict(
            {"data_type_dict": self.data_type_dict, "corr_dict": self.corr_dict}
        )

        self.data = pd.concat([self.data, time_series], axis=1)
        self.corrected_data = pd.concat([self.corrected_data, time_series], axis=1)

    def auto_correct(self):
        self.remove_anomalies()
        self.fill_nan()
        self.resample()

    def remove_anomalies(self):
        for data_type, cols in self.data_type_dict.items():
            if "minmax" in self.corr_dict[data_type].keys():
                self._minmax_corr(cols=cols, **self.corr_dict[data_type]["minmax"])
            if "derivative" in self.corr_dict[data_type].keys():
                self._derivative_corr(
                    cols=cols, **self.corr_dict[data_type]["derivative"]
                )
        self.correction_journal["remove_anomalies"] = {
            "missing_values": missing_values_dict(self.corrected_data),
            "gaps_stats": gaps_describe(
                self.corrected_data, timestep=self.gaps_timedelta
            ),
        }

    def fill_nan(self):
        for data_type, cols in self.data_type_dict.items():
            function_map = {
                "linear_interpolation": self._linear_interpolation,
                "bfill": self._bfill,
                "ffill": self._ffill,
            }

            for func in self.corr_dict[data_type]["fill_nan"]:
                function_map[func](cols)

        self.correction_journal["fill_nan"] = {
            "missing_values": missing_values_dict(self.corrected_data),
            "gaps_stats": gaps_describe(
                self.corrected_data, timestep=self.gaps_timedelta
            ),
        }

    def resample(self, timestep=None):
        if not timestep:
            timestep = get_mean_timestep(self.corrected_data)

        agg_arguments = {}
        for data_type, cols in self.data_type_dict.items():
            for col in cols:
                key = self.corr_dict[data_type]["resample"]
                agg_arguments[col] = self.resample_func_dict[key]

        resampled = self.corrected_data.resample(timestep).agg(agg_arguments)
        self.corrected_data = resampled

        self.correction_journal["Resample"] = f"Resampled at {timestep}"

    def _get_reversed_data_type_dict(self, cols=None):
        if cols is None:
            cols = self.data.columns

        rev_dict = {}
        for col in cols:
            for key, name_list in self.data_type_dict.items():
                if col in name_list:
                    rev_dict[col] = key
        return rev_dict

    def _get_yaxis_config(self, cols):
        ax_dict = self._get_reversed_data_type_dict(cols=cols)

        ordered_set_cat = list(dict.fromkeys(ax_dict.values()))
        ax_map = {cat: f"y{i + 1}" for i, cat in enumerate(ordered_set_cat)}

        ax_map[list(ax_map.keys())[0]] = "y"

        ax_dict = {k: ax_map[ax_dict[k]] for k in ax_dict.keys()}

        layout_ax_dict = {}
        ax_list = list(ax_map.keys())
        layout_ax_dict["yaxis"] = {"title": ax_list[0]}
        for i, ax in enumerate(ax_list[1:]):
            layout_ax_dict[f"yaxis{i + 2}"] = {"title": ax, "side": "right"}

        return ax_dict, layout_ax_dict

    def _minmax_corr(self, cols, upper, lower):
        df = self.corrected_data.loc[:, cols]
        upper_mask = df > upper
        lower_mask = df < lower
        mask = np.logical_or(upper_mask, lower_mask)
        self.corrected_data[mask] = np.nan

    def _derivative_corr(self, cols, upper_rate, lower_rate):
        df = self.corrected_data.loc[:, cols]
        time_delta = df.index.to_series().diff().dt.total_seconds()
        abs_der = abs(df.diff().divide(time_delta, axis=0))
        abs_der_two = abs(df.diff(periods=2).divide(time_delta, axis=0))

        mask_constant = abs_der <= lower_rate
        mask_der = abs_der >= upper_rate
        mask_der_two = abs_der_two >= upper_rate

        mask_to_remove = np.logical_and(mask_der, mask_der_two)
        mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

        self.corrected_data[mask_to_remove] = np.nan

    def _linear_interpolation(self, cols):
        self._interpolate(cols, method="linear")

    def _interpolate(self, cols, method):
        inter = self.corrected_data.loc[:, cols].interpolate(method=method)
        self.corrected_data.loc[:, cols] = inter

    def _ffill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(method="ffill")
        self.corrected_data.loc[:, cols] = filled

    def _bfill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(method="bfill")
        self.corrected_data.loc[:, cols] = filled

    def plot_gaps(
            self,
            cols=None,
            begin=None,
            end=None,
            gaps_timestep=None,
            title="Gaps plot",
            raw_data=False,
            color_rgb=(243, 132, 48),
            alpha=0.5,
    ):
        if cols is None:
            cols = self.columns

        if gaps_timestep is None:
            gaps_timestep = dt.timedelta(hours=5)

        if raw_data:
            to_plot = select_data(self.data, cols, begin, end)
        else:
            to_plot = select_data(self.corrected_data, cols, begin, end)

        reversed_data_type = self._get_reversed_data_type_dict(cols)
        cols_data_type = defaultdict(list)
        for key, value in reversed_data_type.items():
            cols_data_type[value].append(key)
        cols_data_type = dict(cols_data_type)

        ax_dict, layout_ax_dict = self._get_yaxis_config(cols)

        fig = go.Figure()

        for col in cols:
            y_min = to_plot[cols_data_type[reversed_data_type[col]]].min().min()
            y_max = to_plot[cols_data_type[reversed_data_type[col]]].max().max()

            add_scatter_and_gaps(
                figure=fig,
                series=to_plot[col],
                gap_series=find_gaps(data=to_plot, cols=[col], timestep=gaps_timestep)[
                    col
                ],
                color_rgb=color_rgb,
                alpha=alpha,
                y_min=y_min,
                y_max=y_max,
                yaxis=ax_dict[col],
            )

        fig.update_layout(**layout_ax_dict)
        fig.update_layout(
            dict(
                title=title,
            )
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5
            ),
        )

        fig.show()

    def plot(
            self,
            cols=None,
            title="Correction plot",
            plot_raw=False,
            plot_corrected=False,
            line_corrected=True,
            marker_corrected=True,
            line_raw=True,
            marker_raw=True,
            begin=None,
            end=None,
    ):
        if cols is None:
            cols = self.columns

        to_plot_raw = select_data(self.data, cols, begin, end)
        to_plot_corr = select_data(self.corrected_data, cols, begin, end)

        ax_dict, layout_ax_dict = self._get_yaxis_config(cols)

        fig = go.Figure()

        # Define the color palette
        color_palette = ["#FFAD85", "#FF8D70", "#ED665A", "#52E0B6", "#479A91"]

        num_cols = len(cols)

        for i, col in enumerate(cols):
            if i == 0:
                # Use the first color in the palette for the first column
                color = color_palette[0]
            elif i == 1:
                # Use the last color in the palette for the second column
                color = color_palette[-1]
            elif num_cols <= 5:
                # Use the specified colors for up to 5 columns
                color = color_palette[i % len(color_palette)]
            else:
                # Generate interpolated colors for more than 5 columns
                t = (i - 2) / (num_cols - 3)  # Interpolation parameter
                color = self.interpolate_color(color_palette[0], color_palette[-1], t)

            dark_color = self.darken_color(color, 0.7)

            if line_corrected and not marker_corrected:
                mode_corrected = "lines"
            elif line_corrected and marker_corrected:
                mode_corrected = "lines+markers"
            else:
                mode_corrected = "markers"

            if plot_corrected:
                fig.add_scattergl(
                    x=to_plot_corr.index,
                    y=to_plot_corr[col],
                    name=f"{col}_corrected",
                    mode=mode_corrected,
                    line=dict(width=2, color=dark_color),
                    marker=dict(color=dark_color, opacity=0.2),
                    yaxis=ax_dict[col],
                )

            if line_raw and not marker_raw:
                mode_raw = "lines"
            elif line_raw and marker_raw:
                mode_raw = "lines+markers"
            else:
                mode_raw = "markers"

            if plot_raw:
                fig.add_scattergl(
                    x=to_plot_raw.index,
                    y=to_plot_raw[col],
                    name=f"{col}_raw",
                    mode=mode_raw,
                    marker=dict(color=color, opacity=0.5),
                    yaxis=ax_dict[col],
                )

        fig.update_layout(**layout_ax_dict)
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5
            ),
        )
        fig.update_layout(dict(title=title))
        nb_right_y_axis = len(set(ax_dict.values()))
        x_right_space = 1 - 0.03 * (nb_right_y_axis - 1)
        fig.update_xaxes(domain=(0, x_right_space))
        ax_args = {
            f"yaxis{2 + i}": dict(position=x_right_space + i * 0.03)
            for i in range(nb_right_y_axis)
        }
        fig.update_layout(**ax_args)
        fig.show()
