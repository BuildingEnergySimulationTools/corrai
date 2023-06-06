import numpy as np
import pandas as pd

from corrai.measure import MeasuredDats
from corrai.measure import missing_values_dict
from corrai.measure import gaps_describe
from corrai.measure import select_data


class TestMeasuredDats:
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

    def test_remove_anomalies(self):
        time_index = pd.date_range("2021-01-01 00:00:00", freq="H", periods=11)

        df = pd.DataFrame(
            {
                "dumb_column": [-1, 5, 100, 5, 5.1, 5.1, 6, 7, 22, 6, 5],
                "dumb_column2": [-10, 50, 1000, 50, 50.1, 50.1, 60, 70, 220, 60, 50],
                "dumb_column3": [-100, 500, 10000, 500, 500.1, 500.1, 600, 700, 2200,
                                 600, 500]
            },
            index=time_index,
        )

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
                    50.,
                    np.nan,
                    50.,
                    50.1,
                    50.1,
                    60,
                    70,
                    220.,
                    60,
                    50,
                ],
                "dumb_column3": [-100, 500, 10000, 500, 500.1, 500.1, 600, 700, 2200,
                                 600, 500]
            },
            index=time_index,
        )

        tested_obj = MeasuredDats(
            data=df,
            category_dict={
                "col_1": ["dumb_column"],
                "col_2": ["dumb_column2"],
                "col_3": ["dumb_column3"],
            },
            category_transformations={
                "col_1": {
                    "ANOMALIES": [
                        ["drop_threshold", {"upper": 50, "lower": 0}],
                        ["drop_time_gradient", {"lower_rate": 0, "upper_rate": 0.004}],
                    ],
                    "PROCESS": [
                        ['apply_expression', {"expression": "X * 2"}]
                    ],
                    "RESAMPLE": "mean"
                },
                "col_2": {
                    "ANOMALIES": [
                        ["drop_threshold", {"upper": 500, "lower": 0}],
                    ],
                    "RESAMPLE": "sum"
                },
                "col_3": {}
            },
            common_transformations=[
                ["interpolate", {"method": 'linear'}],
                ["fill_na", {"method": 'bfill'}],
                ["fill_na", {"method": 'ffill'}],
            ]
        )

        assert ref_anomalies.equals(
            tested_obj.get_corrected_data(pipes_list=["ANOMALIES"]))

        ref = pd.DataFrame({"dumb_column": [10.08, 12.42, 10.0],
                            "dumb_column2": [250.10, 460.10, 50.0]},
                           index=pd.date_range("2021-01-01", freq='5H', periods=3))

        assert ref.equals(tested_obj.get_corrected_data(resampling_rule='5H'))

    def test_fill_nan(self):
        time_index = pd.date_range("2021-01-01 00:00:00", freq="H", periods=5)

        df = pd.DataFrame(
            {
                "dumb_column": [np.nan, 5, np.nan, 7, np.nan],
                "dumb_column2": [np.nan, 5, np.nan, 7, np.nan],
                "dumb_column3": [np.nan, 5, np.nan, 7, np.nan],
            },
            index=time_index,
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [5.0, 5.0, 6.0, 7.0, 7.0],
                "dumb_column2": [5.0, 5.0, 7.0, 7.0, 7.0],
                "dumb_column3": [5.0, 5.0, 5.0, 7.0, 7.0],
            },
            index=time_index,
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={
                "col_1": ["dumb_column"],
                "col_2": ["dumb_column2"],
                "col_3": ["dumb_column3"],
            },
            corr_dict={
                "col_1": {"fill_nan": ["linear_interpolation", "bfill", "ffill"]},
                "col_2": {"fill_nan": ["bfill", "ffill"]},
                "col_3": {"fill_nan": ["ffill", "bfill"]},
            },
        )

        tested_obj.fill_nan()

        print(tested_obj.corrected_data)

        assert ref.equals(tested_obj.corrected_data)

    def test_resample(self):
        time_index_df = pd.date_range("2021-01-01 00:00:00", freq="30T", periods=4)

        df = pd.DataFrame(
            {
                "dumb_column": [5.0, 5.0, 6.0, 6.0],
                "dumb_column2": [5.0, 5.0, 6.0, 6.0],
            },
            index=time_index_df,
        )

        time_index_res = pd.date_range("2021-01-01 00:00:00", freq="H", periods=2)

        ref = pd.DataFrame(
            {
                "dumb_column": [5.0, 6.0],
                "dumb_column2": [10.0, 12.0],
            },
            index=time_index_res,
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={
                "col_1": ["dumb_column"],
                "col_2": ["dumb_column2"],
            },
            corr_dict={
                "col_1": {"resample": "mean"},
                "col_2": {"resample": "sum"},
            },
        )

        tested_obj.resample("H")

        assert ref.equals(tested_obj.corrected_data)

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

    def test_get_reversed_data_type_dict(self):
        test = MeasuredDats(
            data=pd.DataFrame(),
            data_type_dict={
                "cat_1": ["dumb_column"],
                "cat_2": ["dumb_column2"],
            },
            corr_dict={
                "cat_1": {},
                "cat_2": {},
            },
        )

        to_test = test._get_reversed_data_type_dict(["dumb_column", "dumb_column2"])

        assert to_test == {"dumb_column": "cat_1", "dumb_column2": "cat_2"}

    def test_get_yaxis_config(self):
        test = MeasuredDats(
            data=pd.DataFrame(),
            data_type_dict={
                "cat_1": ["dumb_column"],
                "cat_2": ["dumb_column2"],
            },
            corr_dict={
                "cat_1": {},
                "cat_2": {},
            },
        )

        ax_dict, layout_ax_dict = test._get_yaxis_config(
            cols=["dumb_column", "dumb_column2"]
        )

        ax_dict_ref = {"dumb_column": "y", "dumb_column2": "y2"}
        layout_ax_dict_ref = {
            "yaxis": {"title": "cat_1"},
            "yaxis2": {"title": "cat_2", "side": "right"},
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
            data_type_dict={"dumb_type": ["col"]},
            corr_dict={"dumb_type": {}},
        )

        new_dat = pd.DataFrame(
            {"col1": [2, 3, 4]},
            index=pd.date_range("2009-01-01 01:00:00", freq="H", periods=3),
        )

        mdata.add_time_series(new_dat, data_type="dumb_type")

        assert mdata.data_type_dict == {"dumb_type": ["col", "col1"]}
        assert mdata.corr_dict == {"dumb_type": {}}
        pd.testing.assert_frame_equal(mdata.data, pd.concat([data, new_dat], axis=1))
        pd.testing.assert_frame_equal(
            mdata.corrected_data, pd.concat([data, new_dat], axis=1)
        )

        new_dat.columns = ["col2"]

        mdata.add_time_series(new_dat, data_type="new_dumb_type")

        assert mdata.data_type_dict == {
            "dumb_type": ["col", "col1"],
            "new_dumb_type": ["col2"],
        }
        assert mdata.corr_dict == {"dumb_type": {}, "new_dumb_type": {}}

        new_dat.columns = ["col3"]

        mdata.add_time_series(
            new_dat, data_type="new_dumb_type2", data_corr_dict={"resample": "mean"}
        )

        assert mdata.data_type_dict == dict(
            dumb_type=["col", "col1"], new_dumb_type=["col2"], new_dumb_type2=["col3"]
        )
        assert mdata.corr_dict == {
            "dumb_type": {},
            "new_dumb_type": {},
            "new_dumb_type2": {"resample": "mean"},
        }
