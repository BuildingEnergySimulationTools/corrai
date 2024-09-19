import datetime as dt
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

import corrai.transformers as ct
from corrai.base.math import time_integrate
from corrai.transformers import PdIdentity
from corrai.base.utils import check_datetime_index


class Transformer(Enum):
    DROPNA = "DROPNA"
    RENAME_COLUMNS = "RENAME_COLUMNS"
    SK_TRANSFORMER = "SK_TRANSFORMER"
    DROP_THRESHOLD = "DROP_THRESHOLD"
    DROP_TIME_GRADIENT = "DROP_TIME_GRADIENT"
    APPLY_EXPRESSION = "APPLY_EXPRESSION"
    TIME_GRADIENT = "TIME_GRADIENT"
    FILL_NA = "FILL_NA"
    BFILL = "BFILL"
    FFILL = "FFILL"
    RESAMPLE = "RESAMPLE"
    INTERPOLATE = "INTERPOLATE"
    GAUSSIAN_FILTER = "GAUSSIAN_FILTER"
    REPLACE_DUPLICATED = "REPLACE_DUPLICATED"


TRANSFORMER_MAP = {
    "DROPNA": ct.PdDropna,
    "RENAME_COLUMNS": ct.PdRenameColumns,
    "SK_TRANSFORMER": ct.PdSkTransformer,
    "DROP_THRESHOLD": ct.PdDropThreshold,
    "DROP_TIME_GRADIENT": ct.PdDropTimeGradient,
    "APPLY_EXPRESSION": ct.PdApplyExpression,
    "TIME_GRADIENT": ct.PdTimeGradient,
    "FILL_NA": ct.PdFillNa,
    "BFILL": ct.PdBfill,
    "FFILL": ct.PdFfill,
    "RESAMPLE": ct.PdResampler,
    "INTERPOLATE": ct.PdInterpolate,
    "GAUSSIAN_FILTER": ct.PdGaussianFilter1D,
    "REPLACE_DUPLICATED": ct.PdReplaceDuplicated,
}

ENCODING_MAP = {"Transformer": Transformer}


class AggMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
    CUMSUM = "CUMSUM"
    DIFF = "DIFF"
    TIME_INTEGRATE = "TIME_INTEGRATE"


AGG_METHOD_MAP = {
    "MEAN": "mean",
    "SUM": "sum",
    "CUMSUM": "cusmsum",
    "DIFF": "diff",
    "TIME_INTEGRATE": time_integrate,
}

COLOR_PALETTE = ["#FFAD85", "#FF8D70", "#ED665A", "#52E0B6", "#479A91"]


def missing_values_dict(df):
    return {
        "Number_of_missing": df.count(),
        "Percent_of_missing": (1 - df.count() / df.shape[0]) * 100,
    }


def set_multi_yaxis_layout(figure, ax_dict, axis_space):
    nb_right_y_axis = len(set(ax_dict.values()))
    x_right_space = 1 - axis_space * (nb_right_y_axis - 1)
    figure.update_xaxes(domain=(0, x_right_space))
    ax_args = {
        f"yaxis{2 + i}": dict(position=x_right_space + i * axis_space)
        for i in range(nb_right_y_axis)
    }
    figure.update_layout(**ax_args)


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


def select_data(df, cols=None, begin=None, end=None):
    cols = df.columns if cols is None else cols
    begin = df.index[0] if begin is None else begin
    end = df.index[-1] if end is None else end

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
    cols = data.columns if cols is None else cols
    timestep = get_mean_timestep(data) if timestep is None else timestep

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
            yaxis=yaxis,
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


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj.__class__.__name__), "value": obj.value}
        return json.JSONEncoder.default(self, obj)


class CustomDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if "__enum__" in d:
            enum_class = ENCODING_MAP[d["__enum__"]]
            return enum_class(d["value"])
        return d


class MeasuredDats:
    def __init__(
        self,
        data: pd.DataFrame = None,
        category_dict: dict[str, list[str]] = None,
        category_transformations: dict[str, dict[str, [[Transformer, dict]]]] = None,
        common_transformations: dict[str, [[[Transformer, dict]]]] = None,
        resampler_agg_methods: dict[str, AggMethod] = None,
        transformers_list: list[str] = None,
        config_file_path: str | Path = None,
    ):
        """
        A class for handling time-series data with missing values.
        Use scikit learn Pipelines to perform operations.
        Plot methods to check the effects of pipelines et plot data on multiple
        y axis.

        Parameters:
        -----------
        data (pandas.DataFrame): The measured data.
        category_dict (dict, optional): A dictionary mapping data categories to
            column names. Defaults to None.
        category_transformations (dict, optional): A dictionary specifying
            category-specific transformations. Defaults to None. The dictionary
            keys must match the category name. For each category, a dictionary is
            specified. The keys are the transformer name, the value is a list
            ['transformer_map_name', {Corrai transformer args}]. Use only
            transformers defined in TRANSFORMER_MAP. If necessary a specific key
            "RESAMPLE" may be provided to specify an aggregation method. Method
            must be in RESAMPLE_METHS dict. If not specified, default aggreagation
            method is mean. An exemple of configuration is given below

        common_transformations (dict, optional): A dictionary specifying common
            transformations. The keys are the transformer name, the value is a list
            ['transformer_map_name', {Corrai transformer args}]. Use only
            transformers defined in TRANSFORMER_MAP. An example of configuration
            is given below

        resampler_agg_methods (dict, optional): A dictionary specifying the aggrgation
            method for the categories. Method must be corrai.measure RESAMPLE_METHS
            dict keys(). If no method is provided for a category, default method
            is numpy mean. Default value for this parameter is an empty dict, meaning
            aggregation method will be numpy mean for all categories.

        transformers_list (list, optional): A list of transformer names.
            Defaults to None. A list of transformer name. The order determines the
            order of the transformers in the pipeline. Note that RESAMPLE will always
            be added at the end of the pipeline. If None, a default order will be
            specified as follows: ["CATEGORY_TRANSFORMER_1", ...,
            "CATEGORY_TRANSFORMER_n", "COMMON_TRANSFORMER_1", ...,
            "CATEGORY_TRANSFORMER_n", "RESAMPLE"]

        config_file_path (str, optional): The file path for reading a json
        configuration file. Defaults to None.

        Properties:
        -----------
            columns: Returns the column names of the data.

            category_trans_names: Returns the names of category-specific
                transformations.

            common_trans_names: Returns the names of common
                transformations.

        Example:
        --------
        >>>my_data = MeasuredDats(
            data = raw_data,
            category_dict = {
                "temperatures": [
                    'T_Wall_Ins_1', 'T_Wall_Ins_2', 'T_Ins_Ins_1', 'T_Ins_Ins_2',
                    'T_Ins_Coat_1', 'T_Ins_Coat_2', 'T_int_1', 'T_int_2', 'T_ext',
                    'T_garde'
                ],
                "illuminance": ["Lux_CW"],
                "radiation": ["Sol_rad"]
            },
            category_transformations = {
                "temperatures": {
                    "ANOMALIES": [
                        ["drop_threshold", {"upper": 100, "lower": -20}],
                        ["drop_time_gradient", {"upper_rate": 2, "lower_rate": 0}]
                    ],
                },
                "illuminance": {
                    "ANOMALIES": [
                        ["drop_threshold", {"upper": 1000, "lower": 0}],
                    ],
                },
                "radiation": {
                    "ANOMALIES": [
                        ["drop_threshold", {"upper": 1000, "lower": 0}],
                    ],
                }
            },
            common_transformations={
                "COMMON": [
                    ["interpolate", {"method": 'linear'}],
                    ["fill_na", {"method": 'bfill'}],
                    ["fill_na", {"method": 'bfill'}]
                ]
            },
            resampler_agg_methods={
                "radiation": "sum" # Method is just an example.
            },
            transformers_list=["ANOMALIES", "COMMON"]
        )

        >>>my_data.get_corrected_data()
        """
        if data is not None:
            self.set_data(data)
        else:
            self.data = None

        if config_file_path is None:
            self.category_dict = category_dict
            self.category_trans = category_transformations
            self.common_trans = common_transformations
            self.transformers_list = transformers_list
            self.resampler_agg_methods = resampler_agg_methods
        else:
            self.read_config_file(config_file_path)

        if self.category_dict is None and self.data is not None:
            self.category_dict = {"data": self.data.columns}

        if self.category_trans is None:
            self.category_trans = {}

        if self.common_trans is None:
            self.common_trans = {}

        if self.transformers_list is None:
            self.transformers_list = self.category_trans_names + self.common_trans_names

        if self.resampler_agg_methods is None:
            self.resampler_agg_methods = {}

    @property
    def columns(self):
        return self.data.columns

    @property
    def category_trans_names(self):
        lst = [list(val.keys()) for val in self.category_trans.values()]
        lst = sum(lst, [])
        return list(dict.fromkeys(lst))

    @property
    def common_trans_names(self):
        lst = list(self.common_trans.keys())
        return list(dict.fromkeys(lst))

    def _select_columns(self, cols, category):
        if cols is None:
            cols = self.columns
        elif isinstance(cols, str):
            cols = [cols]

        if category is not None:
            try:
                cols = self.category_dict[category]
            except KeyError:
                raise ValueError(f"{category} not found in category_dict")
        return cols

    def set_data(self, data: pd.DataFrame):
        """
        The proper way or reset DataFrame. Check index is valid, before assigning data
        to self.data.
        """
        check_datetime_index(data)
        self.data = data

    def get_missing_value_stats(
        self,
        transformers_list: [str] = None,
        resampling_rule: str | dt.timedelta = None,
    ):
        """
        Returns statistics on missing values NaN for the corresponding
        transformers_list pipeline. Number of missing values for all columns
        and corresponding % of missing values
        """
        data = self.get_corrected_data(transformers_list, resampling_rule)
        return missing_values_dict(data)

    def get_gaps_description(
        self,
        cols: str | list[str] = None,
        transformers_list: list[str] = None,
        resampling_rule: str | dt.timedelta = False,
        gaps_timedelta: dt.timedelta = None,
    ):
        """
        Returns statistics on gaps duration for specified columns for the
        specified transformation. The column "combination" returns "aggregated"
        gaps statistics
        """

        if gaps_timedelta is None:
            gaps_timedelta = get_mean_timestep(self.data)
        data = self.get_corrected_data(transformers_list, resampling_rule)
        return gaps_describe(df_in=data, cols=cols, timestep=gaps_timedelta)

    def get_pipeline(
        self,
        transformers_list: list[str] = None,
        resampling_rule: str | dt.timedelta = False,
    ):
        """
        Creates and returns a data processing pipeline. Custom transformer list may be
         specified. resampling_rule add a resampler to the pipeline.
        - transformers_list: custom transformers list. If None, MeasuredDat
        transformers_list is used
        - rule: Timedelta for resampling if used,
        """
        if transformers_list is None:
            transformers_list = self.transformers_list.copy()

        if Transformer.RESAMPLE.value in transformers_list and not resampling_rule:
            raise ValueError(
                "RESAMPLE is present in transformers_list but no rule"
                "have been specified. use resampling_rule argument"
            )

        if resampling_rule and Transformer.RESAMPLE.value not in transformers_list:
            transformers_list += [Transformer.RESAMPLE.value]

        if not transformers_list:
            obj_list = [PdIdentity()]
        else:
            obj_list = []
            for trans in transformers_list:
                if trans in self.category_trans_names:
                    obj_list.append(self.get_category_transformer(trans))
                elif trans == Transformer.RESAMPLE.value:
                    obj_list.append(self.get_resampler(resampling_rule))
                else:
                    obj_list.append(self.get_common_transformer(trans))

        return make_pipeline(*obj_list)

    def get_corrected_data(
        self,
        transformers_list: list[str] = None,
        resampling_rule: str | dt.timedelta = False,
    ):
        """
        Applies the pipeline to the data and returns the corrected data.
        Custom transformer list may be specified. resampling_rule add a
        resampler to the pipeline or configures it if RESAMPLE is set in
        transformers_list.
        """
        pipe = self.get_pipeline(
            transformers_list=transformers_list, resampling_rule=resampling_rule
        )
        return pipe.fit_transform(self.data)

    def get_common_transformer(self, transformation: str):
        """
        Returns a pipeline for a common transformation.
        """
        common_trans = self.common_trans[transformation]
        return make_pipeline(
            *[TRANSFORMER_MAP[trans[0].value](**trans[1]) for trans in common_trans]
        )

    def get_category_transformer(self, transformation: str):
        """
        Returns a pipeline for a category-specific transformation.
        """
        column_config_list = []
        for data_cat, cols in self.category_dict.items():
            if data_cat in self.category_trans.keys():
                if transformation in self.category_trans[data_cat].keys():
                    transformations = self.category_trans[data_cat][transformation]
                else:
                    transformations = []
                if transformations:
                    column_config_list.append(
                        (
                            f"{transformation}_{data_cat}",
                            make_pipeline(
                                *[
                                    TRANSFORMER_MAP[trans[0].value](**trans[1])
                                    for trans in transformations
                                ]
                            ),
                            cols,
                        )
                    )

        return ColumnTransformer(
            column_config_list, verbose_feature_names_out=False, remainder="passthrough"
        ).set_output(transform="pandas")

    def get_resampler(self, rule: str | dt.timedelta, remainder_method=AggMethod.MEAN):
        """
        Returns a resampler for data resampling
        """
        column_config_list = []
        for data_cat, cols in self.category_dict.items():
            try:
                method = self.resampler_agg_methods[data_cat]
                column_config_list.append((cols, AGG_METHOD_MAP[method.value]))
            except KeyError:
                pass

        if not column_config_list:
            return ct.PdResampler(rule=rule, method=AGG_METHOD_MAP["MEAN"])
        else:
            return ct.PdColumnResampler(
                rule=rule,
                columns_method=column_config_list,
                remainder=AGG_METHOD_MAP[remainder_method.value],
            )

    def write_config_file(self, file_path: str | Path):
        """
        Writes the current configuration to a json file.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            to_dump = {
                "category_dict": self.category_dict,
                "category_transformations": self.category_trans,
                "common_transformations": self.common_trans,
                "transformers_list": self.transformers_list,
                "resampler_agg_methods": self.resampler_agg_methods,
            }
            json.dump(to_dump, f, indent=4, cls=CustomEncoder)

    def read_config_file(self, file_path: str | Path):
        """
        Reads the configuration from a file
        """
        with open(file_path, encoding="utf-8") as f:
            config_dict = json.load(f, cls=CustomDecoder)

        attribute_list = [
            ("category_dict", "category_dict"),
            ("category_transformations", "category_trans"),
            ("common_transformations", "common_trans"),
            ("transformers_list", "transformers_list"),
            ("resampler_agg_methods", "resampler_agg_methods"),
        ]
        for attr in attribute_list:
            try:
                setattr(self, attr[1], config_dict[attr[0]])
            except KeyError:
                setattr(self, attr[1], None)

    def add_time_series(
        self,
        time_series: pd.Series | pd.DataFrame,
        category: str,
        category_transformations=None,
    ):
        """
        Adds a time series to the data.
        """
        if isinstance(time_series, pd.Series):
            time_series = time_series.to_frame()

        check_datetime_index(time_series)
        if category_transformations is None:
            category_transformations = {}

        if category in self.category_dict.keys():
            self.category_dict[category] += list(time_series.columns)
        else:
            self.category_dict[category] = list(time_series.columns)
            self.category_trans[category] = category_transformations

        self.data = pd.concat([self.data, time_series], axis=1)

    def _get_reversed_category_dict(self, cols=None):
        if cols is None:
            cols = self.data.columns

        rev_dict = {}
        for col in cols:
            for key, name_list in self.category_dict.items():
                if col in name_list:
                    rev_dict[col] = key
        return rev_dict

    def _get_yaxis_config(self, cols):
        ax_dict = self._get_reversed_category_dict(cols=cols)

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

    def plot_gaps(
        self,
        cols: str | list[str] = None,
        category: str = None,
        begin: str | dt.datetime = None,
        end: str | dt.datetime = None,
        gaps_timestep: dt.timedelta = None,
        title: str = "Gaps plot",
        plot_raw: bool = False,
        color_rgb: set[int, int, int] = (100, 100, 100),
        alpha: float = 0.5,
        resampling_rule: str | dt.timedelta = None,
        transformers_list: list[str] = None,
        axis_space: float = 0.03,
    ):
        """
        Plot data (raw or corrected, default is corrected), plots grey boxes
        when values are missing beyond a given gaps_timestep

        cols (str,  list, optional): a column name or a List of column names
            to plot. If not provided, all columns will be plotted. Default is None.

        category (str): A category name. It will plot all the columns in the given
            category. Based on category dict. It overwrites cols argument

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
            resampling_rule: data resampling rule

        transformers_list: transformations order list. If None it uses default
            transformers_list
        """
        cols = self._select_columns(cols, category)

        gaps_timestep = (
            dt.timedelta(hours=5) if gaps_timestep is None else gaps_timestep
        )

        if plot_raw:
            to_plot = select_data(self.data, cols, begin, end)
        else:
            to_plot = select_data(
                self.get_corrected_data(
                    transformers_list=transformers_list, resampling_rule=resampling_rule
                ),
                cols,
                begin,
                end,
            )

        reversed_data_type = self._get_reversed_category_dict(cols)
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
        set_multi_yaxis_layout(figure=fig, ax_dict=ax_dict, axis_space=axis_space)

        return fig

    def plot(
        self,
        cols: str | list[str] = None,
        category: str = None,
        begin: str | dt.datetime = None,
        end: str | dt.datetime = None,
        title: str = "Gaps plot",
        plot_raw: bool = False,
        plot_corrected: bool = True,
        line_corrected: bool = True,
        marker_corrected: bool = True,
        line_raw: bool = True,
        marker_raw: bool = True,
        resampling_rule: str | dt.timedelta = None,
        transformers_list: list[str] = None,
        axis_space: float = 0.03,
    ):
        """
        Generate a plot of specified columns or category.
        Can be used to compare the original and corrected values of the given
        columns over the specified time range.

        cols : str, list of str, optional
            The names of the column, or a list of columns names to plot.
            If None (default), all columns are plotted.

        category (str): A category name. It will plot all the columns in the given
            category. Based on category dict. It overwrites cols argument

        title : str, optional
            The title of the plot. Defaults to "Correction plot".

        begin : str or datetime-like, optional
            A string or datetime-like object specifying the start of the
            time range to plot. If None (default), plot all data.

        end : str or datetime-like, optional
            A string or datetime-like object specifying the end of the
            time range to plot. If None (default), plot all data.

        plot_raw : bool, optional
            If True, plot the raw values

        plot_corrected : bool, optional
            If True, plot the corrected values.

        line_corrected: bool, optional
            If True, plot corrected values using lines

        line_raw: bool, optional
            If True, plot raw values using lines

        marker_corrected: bool, optional
            If True, plot corrected values using markers

        marker_raw: bool, optional
            If True, plot raw values using markers

        resampling_rule: False
            If resampling rule is specified, resample corrected data using
            resampler and aggregation methods specified in category_transformers.
            It will not affect raw data

        transformers_list: list, Optional
            transformations order list. Default None uses default
            transformers_list
        """
        cols = self._select_columns(cols, category)

        to_plot_raw = select_data(self.data, cols, begin, end)
        to_plot_corr = select_data(
            self.get_corrected_data(
                transformers_list=transformers_list, resampling_rule=resampling_rule
            ),
            cols,
            begin,
            end,
        )

        ax_dict, layout_ax_dict = self._get_yaxis_config(cols)

        fig = go.Figure()

        num_cols = len(cols)

        for i, col in enumerate(cols):
            if i == 0:
                # Use the first color in the palette for the first column
                color = COLOR_PALETTE[0]
            elif i == 1:
                # Use the last color in the palette for the second column
                color = COLOR_PALETTE[-1]
            elif num_cols <= 5:
                # Use the specified colors for up to 5 columns
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            else:
                # Generate interpolated colors for more than 5 columns
                t = (i - 2) / (num_cols - 3)  # Interpolation parameter
                color = interpolate_color(COLOR_PALETTE[0], COLOR_PALETTE[-1], t)

            dark_color = darken_color(color, 0.7)

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
        fig.update_layout(dict(title=title))
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5
            ),
        )
        set_multi_yaxis_layout(figure=fig, ax_dict=ax_dict, axis_space=axis_space)

        return fig
