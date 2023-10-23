import numpy as np
import pandas as pd
import json

from corrai.measure import MeasuredDats
from corrai.measure import missing_values_dict
from corrai.measure import gaps_describe
from corrai.measure import select_data, Transformer
from copy import deepcopy

from pathlib import Path

import pytest

RESOURCES_DIR = Path(__file__).parent / "resources"

TEST_DF = pd.DataFrame(
    {
        "dumb_column": [-1, 5, 100, 5, 5.1, 5.1, 6, 7, 22, 6, 5],
        "dumb_column2": [-10, 50, 1000, 50, 50.1, 50.1, 60, 70, 220, 60, 50],
        "dumb_column3": [-100, 500, 10000, 500, 500.1, 500.1, 600, 700, 2200, 600, 500],
    },
    index=pd.date_range("2021-01-01 00:00:00", freq="H", periods=11),
)


@pytest.fixture(scope="session")
def my_measure():
    tested_obj = MeasuredDats(
        data=TEST_DF,
        category_dict={
            "col_1": ["dumb_column"],
            "col_2": ["dumb_column2"],
            "col_3": ["dumb_column3"],
        },
        category_transformations={
            "col_1": {
                "ANOMALIES": [
                    [Transformer.DROP_THRESHOLD, {"upper": 50, "lower": 0}],
                    [
                        Transformer.DROP_TIME_GRADIENT,
                        {"lower_rate": 0, "upper_rate": 0.004},
                    ],
                ],
                "PROCESS": [[Transformer.APPLY_EXPRESSION, {"expression": "X * 2"}]],
            },
            "col_2": {
                "ANOMALIES": [
                    [Transformer.DROP_THRESHOLD, {"upper": 500, "lower": 0}],
                ],
            },
            "col_3": {},
        },
        common_transformations={
            "COMMON": [
                [Transformer.INTERPOLATE, {"method": "linear"}],
                [Transformer.FILL_NA, {"method": "bfill"}],
                [Transformer.FILL_NA, {"method": "ffill"}],
            ]
        },
        resampler_agg_methods={"col_2": "sum"},
    )

    return tested_obj


class TestMeasuredDats:
    def test_rw_config_file(self, my_measure, tmp_path_factory):
        test_save_path = tmp_path_factory.mktemp("save")

        my_measure_loc = deepcopy(my_measure)
        my_measure_loc.transformers_list = ["PROCESS", "COMMON"]
        my_measure_loc.write_config_file(test_save_path / "save.json")

        to_test = MeasuredDats(
            data=TEST_DF, config_file_path=test_save_path / "save.json"
        )

        assert to_test.category_dict == my_measure_loc.category_dict
        assert to_test.category_trans == my_measure_loc.category_trans
        assert to_test.common_trans == my_measure_loc.common_trans
        assert to_test.transformers_list == my_measure_loc.transformers_list

        with open(test_save_path / "save.json", "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

        to_test = MeasuredDats(
            data=TEST_DF, config_file_path=test_save_path / "save.json"
        )

        assert to_test.category_dict == {"data": TEST_DF.columns}
        assert to_test.category_trans == {}
        assert to_test.common_trans == {}
        assert to_test.transformers_list == []

    def test_select_data(self):
        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [1, 2],
            }
        )

        pd.testing.assert_frame_equal(select_data(df), df)

        ref = pd.DataFrame({"a": [1]})

        pd.testing.assert_frame_equal(select_data(df, begin=0, end=0, cols=["a"]), ref)

    def test_remove_anomalies(self, my_measure):
        ref_anomalies = pd.DataFrame(
            {
                "dumb_column": [
                    np.nan,
                    5,
                    np.nan,
                    np.nan,
                    5.1,
                    np.nan,
                    6,
                    7,
                    np.nan,
                    6,
                    5,
                ],
                "dumb_column2": [
                    np.nan,
                    50.0,
                    np.nan,
                    50.0,
                    50.1,
                    50.1,
                    60,
                    70,
                    220.0,
                    60,
                    50,
                ],
                "dumb_column3": [
                    -100,
                    500,
                    10000,
                    500,
                    500.1,
                    500.1,
                    600,
                    700,
                    2200,
                    600,
                    500,
                ],
            },
            index=TEST_DF.index,
        )

        assert ref_anomalies.equals(
            my_measure.get_corrected_data(transformers_list=["ANOMALIES"])
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [10.08, 12.42, 10.0],
                "dumb_column2": [250.10, 460.10, 50.0],
                "dumb_column3": [2280.02000, 920.02000, 500.00000],
            },
            index=pd.date_range("2021-01-01", freq="5H", periods=3),
        )

        pd.testing.assert_frame_equal(
            my_measure.get_corrected_data(resampling_rule="5H"), ref
        )

    def test_missing_values_dict(self):
        time_index_df = pd.date_range("2021-01-01 00:00:00", freq="30T", periods=4)

        df = pd.DataFrame(
            {
                "dumb_column": [5.0, np.nan, 6.0, 6.0],
                "dumb_column2": [5.0, 5.0, np.nan, np.nan],
            },
            index=time_index_df,
        )

        res = {
            "Number_of_missing": pd.Series([3, 2], index=df.columns),
            "Percent_of_missing": pd.Series([25.0, 50.0], index=df.columns),
        }

        to_test = missing_values_dict(df)

        assert res["Number_of_missing"].equals(to_test["Number_of_missing"])
        assert res["Percent_of_missing"].equals(to_test["Percent_of_missing"])

    def test_gaps_describe(self):
        time_index_df = pd.date_range("2021-01-01 00:00:00", freq="H", periods=5)

        df = pd.DataFrame(
            {
                "dumb_column1": [np.nan, 5, 5, 5, 5],
                "dumb_column2": [5, np.nan, 5, 5, 5],
                "dumb_column3": [5, np.nan, np.nan, 5, 5],
                "dumb_column4": [5, 5, 5, 5, np.nan],
            },
            index=time_index_df,
        )

        one_hour_dt = pd.to_timedelta("2H")
        one_n_half = pd.to_timedelta("2H30min")
        two_hour_dt = pd.to_timedelta("3H")
        two_n_half = pd.to_timedelta("3H30min")
        three_hour = pd.to_timedelta("4H")
        spec_std = pd.to_timedelta("0 days 01:24:51.168824543")
        nat = pd.NaT

        ref = pd.DataFrame(
            {
                "dumb_column1": [
                    1,
                    one_hour_dt,
                    nat,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                ],
                "dumb_column2": [
                    1,
                    one_hour_dt,
                    nat,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                ],
                "dumb_column3": [
                    1,
                    two_hour_dt,
                    nat,
                    two_hour_dt,
                    two_hour_dt,
                    two_hour_dt,
                    two_hour_dt,
                    two_hour_dt,
                ],
                "dumb_column4": [
                    1,
                    one_hour_dt,
                    nat,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                    one_hour_dt,
                ],
                "combination": [
                    2,
                    two_hour_dt,
                    spec_std,
                    one_hour_dt,
                    one_n_half,
                    two_hour_dt,
                    two_n_half,
                    three_hour,
                ],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )

        assert ref.equals(gaps_describe(df))

    def test_get_reversed_data_type_dict(self, my_measure):
        to_test = my_measure._get_reversed_category_dict(
            ["dumb_column", "dumb_column2"]
        )

        assert to_test == {"dumb_column": "col_1", "dumb_column2": "col_2"}

    def test_get_yaxis_config(self, my_measure):
        ax_dict, layout_ax_dict = my_measure._get_yaxis_config(
            cols=["dumb_column", "dumb_column2", "dumb_column3"]
        )

        ax_dict_ref = {"dumb_column": "y", "dumb_column2": "y2", "dumb_column3": "y3"}
        layout_ax_dict_ref = {
            "yaxis": {"title": "col_1"},
            "yaxis2": {"title": "col_2", "side": "right"},
            "yaxis3": {"title": "col_3", "side": "right"},
        }

        assert ax_dict == ax_dict_ref
        assert layout_ax_dict == layout_ax_dict_ref

    def test_add_time_series(self):
        data = pd.DataFrame(
            {"col": [1, 2, 3]},
            index=pd.date_range("2009-01-01 00:00:00", freq="H", periods=3),
        )

        mdata = MeasuredDats(
            data.copy(),
            category_dict={"dumb_type": ["col"]},
            category_transformations={"dumb_type": {}},
            common_transformations={},
        )

        new_dat = pd.DataFrame(
            {"col1": [2, 3, 4]},
            index=pd.date_range("2009-01-01 01:00:00", freq="H", periods=3),
        )

        mdata.add_time_series(new_dat, category="dumb_type")

        assert mdata.category_dict == {"dumb_type": ["col", "col1"]}
        assert mdata.category_trans == {"dumb_type": {}}

        new_dat.columns = ["col2"]

        mdata.add_time_series(new_dat, category="new_dumb_type")

        assert mdata.category_dict == {
            "dumb_type": ["col", "col1"],
            "new_dumb_type": ["col2"],
        }
        assert mdata.category_trans == {"dumb_type": {}, "new_dumb_type": {}}

        new_dat.columns = ["col3"]

        mdata.add_time_series(
            new_dat,
            category="new_dumb_type2",
            category_transformations={"resample": "mean"},
        )

        assert mdata.category_dict == dict(
            dumb_type=["col", "col1"], new_dumb_type=["col2"], new_dumb_type2=["col3"]
        )
        assert mdata.category_trans == {
            "dumb_type": {},
            "new_dumb_type": {},
            "new_dumb_type2": {"resample": "mean"},
        }

    def test_plot_gap(self, my_measure):
        import datetime as dt

        my_measure.plot_gaps(
            gaps_timestep=dt.timedelta(hours=1), transformers_list=["ANOMALIES"]
        )

    def test_plot(self, my_measure):
        my_measure.plot(
            begin="2021-01-01 02:00:00",
            plot_raw=True,
            resampling_rule="2H",
        )
