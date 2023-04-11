import numpy as np
import pandas as pd
import json
from collections import defaultdict
from scipy import integrate
import plotly.graph_objects as go
import datetime as dt


def time_data_control(data):
    """
    Given a pandas Series or DataFrame `data`, checks that the input data
    as a DateTimeIndex. Returns a DataFrame

    Parameters:
    -----------
    data : pandas Series or DataFrame
        The input data to be checked. If `data` is a pandas Series, it will be
         converted to a DataFrame with a single column.

    Returns:
    --------
    output : pandas DataFrame
        A DataFrame that is guaranteed to have a pandas DateTimeIndex as
        its index.

    Raises:
    -------
    ValueError
        If `data` is not a pandas Series or DataFrame.
        If `data` has an index that is not a pandas DateTimeIndex.
    """
    if isinstance(data, pd.Series):
        output = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        raise ValueError(f"time_series is expecting pandas"
                         f" Series or DataFrame. Got {type(data)}"
                         f"instead")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "time_series index must be a pandas DateTimeIndex"
        )
    return output


def time_gradient(time_series, begin=None, end=None):
    time_series = time_data_control(time_series)

    if begin is None:
        begin = time_series.index[0]

    if end is None:
        end = time_series.index[-1]

    selected_ts = time_series.loc[begin: end, :]

    ts_list = []
    for col in selected_ts:
        col_ts = selected_ts[col].dropna()

        chrono = col_ts.index - col_ts.index[0]
        chrono_sec = chrono.to_series().dt.total_seconds()

        ts_list.append(pd.Series(
            np.gradient(col_ts, chrono_sec),
            index=col_ts.index,
            name=col
        ))

    return pd.concat(ts_list, axis=1)


def time_integrate(
        time_series,
        begin=None,
        end=None,
        interpolate=True,
        interpolation_method='linear'):
    time_series = time_data_control(time_series)

    if begin is None:
        begin = time_series.index[0]

    if end is None:
        end = time_series.index[-1]

    selected_ts = time_series.loc[begin:end, :]

    if interpolate:
        selected_ts = selected_ts.interpolate(method=interpolation_method)

    chrono = (selected_ts.index - selected_ts.index[0]).to_series()
    chrono = chrono.dt.total_seconds()

    res_series = pd.Series(dtype='float64')
    for col in time_series:
        res_series[col] = integrate.trapz(selected_ts[col], chrono)

    return res_series


def missing_values_dict(df):
    return {
        "Number_of_missing": df.count(),
        "Percent_of_missing": (1 - df.count() / df.shape[0]) * 100
    }


def check_config_dict(config_dict):
    if not list(config_dict.keys()) == ['data_type_dict', 'corr_dict']:
        raise ValueError("Invalid data_type or corr_dict")

    categories = list(config_dict["data_type_dict"].keys())
    corr_dict_type = list(config_dict['corr_dict'].keys())
    for cat in categories:
        if cat not in corr_dict_type:
            raise ValueError("Type present in data_type "
                             "is missing in corr_dict")

    for tpe in config_dict['corr_dict'].keys():
        to_test = list(config_dict['corr_dict'][tpe].keys())
        for it in to_test:
            loc_corr = config_dict['corr_dict'][tpe][it]

            if it == "minmax":
                if list(loc_corr.keys()) != ['upper', 'lower']:
                    raise ValueError(f"Invalid configuration for {tpe} minmax")

            elif it == "derivative":
                for k in list(loc_corr.keys()):
                    if k not in ['upper_rate', 'lower_rate']:
                        raise ValueError(f"Invalid configuration "
                                         f"for {tpe} derivative")
            elif it == "fill_nan":
                for elmt in loc_corr:
                    if elmt not in ['linear_interpolation', 'bfill', "ffill"]:
                        raise ValueError(f"Invalid configuration "
                                         f"for {tpe} fill_nan")
            elif it == "resample":
                if loc_corr not in ['mean', 'sum']:
                    raise ValueError(f"Invalid configuration "
                                     f"for {tpe} resample")


def select_data(df, cols=None, begin=None, end=None):
    if cols is None:
        cols = df.columns

    if begin is None:
        begin = df.index[0]

    if end is None:
        end = df.index[-1]

    return df.loc[begin:end, cols]


def find_gaps(df_in, cols=None, timestep=None):
    if not cols:
        cols = df_in.columns

    if not timestep:
        timestep = auto_timestep(df_in)

    # Aggregate in a single columns to know overall quality
    df = df_in.copy()
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

    return pd.DataFrame(
        {k: val.describe() for k, val in res_find_gaps.items()})


def auto_timestep(df):
    return df.index.to_frame().diff().mean()[0]


def add_scatter_and_gaps(figure, series, gap_series, color_rgb, alpha, y_min,
                         y_max, yaxis):
    figure.add_trace(go.Scattergl(
        x=series.index,
        y=series.to_numpy().flatten(),
        mode='lines+markers',
        name=series.name,
        yaxis=yaxis
        # line=dict(color=f'rgb{color_rgb}')
    ))

    for t_idx, gap in gap_series.items():
        figure.add_trace(go.Scattergl(
            x=[t_idx - gap, t_idx - gap, t_idx, t_idx],
            y=[y_min, y_max, y_max, y_min],
            mode='none',
            fill='toself',
            showlegend=False,
            fillcolor=f"rgba({color_rgb[0]}, {color_rgb[1]},"
                      f" {color_rgb[2]} , {alpha})",
            yaxis=yaxis
        ))


class MeasuredDats:
    def __init__(self,
                 data,
                 data_type_dict=None,
                 corr_dict=None,
                 config_file_path=None,
                 gaps_timedelta=None):

        self.data = data.apply(
            pd.to_numeric, args=('coerce',)
        ).copy()
        self.corrected_data = data.apply(
            pd.to_numeric, args=('coerce',)
        ).copy()

        if config_file_path is None:
            self.data_type_dict = data_type_dict
            self.corr_dict = corr_dict
            check_config_dict({
                "data_type_dict": data_type_dict,
                "corr_dict": corr_dict
            })
        else:
            self.read_config_file(config_file_path)

        self.correction_journal = {
            "Entries": data.shape[0],
            "Init": missing_values_dict(data)
        }
        if gaps_timedelta is None:
            self.gaps_timedelta = auto_timestep(self.data)
        else:
            self.gaps_timedelta = gaps_timedelta

        self.resample_func_dict = {
            'mean': np.mean,
            'sum': np.sum
        }

    @property
    def columns(self):
        return self.data.columns

    def write_config_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            to_dump = {"data_type_dict": self.data_type_dict,
                       "corr_dict": self.corr_dict}
            json.dump(to_dump, f, ensure_ascii=False, indent=4)

    def read_config_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        check_config_dict(config_dict)
        self.data_type_dict = config_dict["data_type_dict"]
        self.corr_dict = config_dict["corr_dict"]

    def add_time_series(self, time_series, data_type, data_corr_dict=None):
        time_series = time_data_control(time_series)
        if data_corr_dict is None:
            data_corr_dict = {}

        if data_type in self.data_type_dict.keys():
            self.data_type_dict[data_type] += list(time_series.columns)
        else:
            self.data_type_dict[data_type] = list(time_series.columns)
            self.corr_dict[data_type] = data_corr_dict

        check_config_dict({
            "data_type_dict": self.data_type_dict,
            "corr_dict": self.corr_dict
        })

        self.data = pd.concat([self.data, time_series], axis=1)
        self.corrected_data = pd.concat(
            [self.corrected_data, time_series], axis=1)

    def auto_correct(self):
        self.remove_anomalies()
        self.fill_nan()
        self.resample()

    def remove_anomalies(self):
        for data_type, cols in self.data_type_dict.items():
            if "minmax" in self.corr_dict[data_type].keys():
                self._minmax_corr(
                    cols=cols,
                    **self.corr_dict[data_type]["minmax"]
                )
            if "derivative" in self.corr_dict[data_type].keys():
                self._derivative_corr(
                    cols=cols,
                    **self.corr_dict[data_type]["derivative"]
                )
        self.correction_journal["remove_anomalies"] = {
            "missing_values": missing_values_dict(self.corrected_data),
            "gaps_stats": gaps_describe(
                self.corrected_data, timestep=self.gaps_timedelta)
        }

    def fill_nan(self):
        for data_type, cols in self.data_type_dict.items():
            function_map = {
                "linear_interpolation": self._linear_interpolation,
                "bfill": self._bfill,
                "ffill": self._ffill
            }

            for func in self.corr_dict[data_type]["fill_nan"]:
                function_map[func](cols)

        self.correction_journal["fill_nan"] = {
            "missing_values": missing_values_dict(self.corrected_data),
            "gaps_stats": gaps_describe(
                self.corrected_data, timestep=self.gaps_timedelta)
        }

    def resample(self, timestep=None):
        if not timestep:
            timestep = auto_timestep(self.corrected_data)

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
        ax_map = {cat: f"y{i + 1}"
                  for i, cat in enumerate(ordered_set_cat)}

        ax_map[list(ax_map.keys())[0]] = 'y'

        ax_dict = {k: ax_map[ax_dict[k]] for k in ax_dict.keys()}

        layout_ax_dict = {}
        ax_list = list(ax_map.keys())
        layout_ax_dict["yaxis"] = {"title": ax_list[0]}
        for i, ax in enumerate(ax_list[1:]):
            layout_ax_dict[f"yaxis{i + 2}"] = {
                "title": ax,
                "side": "right"
            }

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
        abs_der = abs(
            df.diff().divide(time_delta, axis=0)
        )
        abs_der_two = abs(
            df.diff(periods=2).divide(time_delta, axis=0)
        )

        mask_constant = abs_der <= lower_rate
        mask_der = abs_der >= upper_rate
        mask_der_two = abs_der_two >= upper_rate

        mask_to_remove = np.logical_and(mask_der, mask_der_two)
        mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

        self.corrected_data[mask_to_remove] = np.nan

    def _linear_interpolation(self, cols):
        self._interpolate(cols, method='linear')

    def _interpolate(self, cols, method):
        inter = self.corrected_data.loc[:, cols].interpolate(method=method)
        self.corrected_data.loc[:, cols] = inter

    def _ffill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(
            method="ffill"
        )
        self.corrected_data.loc[:, cols] = filled

    def _bfill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(
            method="bfill"
        )
        self.corrected_data.loc[:, cols] = filled

    def plot_gaps(
            self,
            cols=None,
            begin=None,
            end=None,
            gaps_timestep=dt.timedelta(hours=5),
            title="Gaps plot",
            raw_data=False,
            color_rgb=(243, 132, 48),
            alpha=0.5):

        if cols is None:
            cols = self.columns

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
            y_min = to_plot[
                cols_data_type[reversed_data_type[col]]].min().min()
            y_max = to_plot[
                cols_data_type[reversed_data_type[col]]].max().max()

            add_scatter_and_gaps(
                figure=fig,
                series=to_plot[col],
                gap_series=find_gaps(
                    df_in=to_plot, cols=[col], timestep=gaps_timestep)[col],
                color_rgb=color_rgb,
                alpha=alpha,
                y_min=y_min,
                y_max=y_max,
                yaxis=ax_dict[col]
            )

        fig.update_layout(**layout_ax_dict)
        fig.update_layout(dict(
            title=title,
        ))
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5),
        )

        fig.show()

    def plot(self, cols=None, title="Correction plot", plot_raw=False,
             begin=None, end=None):

        if cols is None:
            cols = self.columns

        to_plot_raw = select_data(self.data, cols, begin, end)
        to_plot_corr = select_data(self.corrected_data, cols, begin, end)

        ax_dict, layout_ax_dict = self._get_yaxis_config(cols)

        fig = go.Figure()

        for col in cols:
            if plot_raw:
                fig.add_scattergl(
                    x=to_plot_raw.index,
                    y=to_plot_raw[col],
                    name=f"{col}_raw",
                    mode='lines+markers',
                    line=dict(color=f'rgb(216,79,86)'),
                    yaxis=ax_dict[col]
                )

            fig.add_scattergl(
                x=to_plot_corr.index,
                y=to_plot_corr[col],
                name=f"{col}_corrected",
                mode='lines+markers',
                yaxis=ax_dict[col]
            )

        fig.update_layout(**layout_ax_dict)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5),
        )
        fig.update_layout(dict(title=title))

        fig.show()
