import pandas as pd
import numpy as np

import pytest

from corrai.base.utils import (
    _reshape_1d,
    as_1_column_dataframe,
    check_datetime_index,
    float_to_hour,
    hour_to_float,
    get_reversed_dict,
    get_data_blocks,
    get_outer_timestamps,
    get_gaps_gte_mask,
    get_gaps_lte_mask,
)


class TestUtils:
    def test_reshape_1d(self):
        x_in = np.array([[1], [2]])
        np.testing.assert_array_equal(_reshape_1d(x_in), np.array([1, 2]))

        x_in = pd.DataFrame({"a": [1, 2]})
        pd.testing.assert_series_equal(_reshape_1d(x_in), pd.Series([1, 2], name="a"))

    def test_as_1_column_dataframe(self):
        ref = pd.DataFrame({0: [1.0, 2.0]})

        # Test list in
        pd.testing.assert_frame_equal(as_1_column_dataframe([1.0, 2.0]), ref)

        # Test DataFrame in
        pd.testing.assert_frame_equal(as_1_column_dataframe(ref), ref)

        # Test Series in
        pd.testing.assert_frame_equal(as_1_column_dataframe(ref.squeeze()), ref)

        # Test 2D array in
        pd.testing.assert_frame_equal(
            as_1_column_dataframe(np.array([[1.0], [2.0]])), ref
        )

        # Test 1D array in
        pd.testing.assert_frame_equal(as_1_column_dataframe(np.array([1.0, 2.0])), ref)

        with pytest.raises(ValueError):
            as_1_column_dataframe(1)

        with pytest.raises(ValueError):
            as_1_column_dataframe(pd.DataFrame({"a": [1], "b": [2]}))

    def test_check_datetime_index(self):
        with pytest.raises(ValueError):
            check_datetime_index(1)

        with pytest.raises(ValueError):
            check_datetime_index(pd.Series([1, 2]))

    def test_float_to_hour(self):
        assert float_to_hour(1.5) == "01:30"
        assert float_to_hour([1.5]) == ["01:30"]
        np.testing.assert_array_equal(
            float_to_hour(np.array([1.5])), np.array(["01:30"])
        )
        np.testing.assert_array_equal(
            float_to_hour(np.array([[1.5]])), np.array([["01:30"]])
        )
        pd.testing.assert_series_equal(
            float_to_hour(pd.Series([1.5])), pd.Series(["01:30"])
        )
        pd.testing.assert_frame_equal(
            float_to_hour(pd.DataFrame({"a": [2.5]})), pd.DataFrame({"a": ["02:30"]})
        )

    def test_hour_to_float(self):
        assert hour_to_float("01:30") == 1.5
        assert hour_to_float(["01:30"]) == [1.5]
        np.testing.assert_array_equal(
            hour_to_float(np.array(["01:30"])), np.array([1.5])
        )
        np.testing.assert_array_equal(
            hour_to_float(np.array([["01:30"]])), np.array([[1.5]])
        )
        pd.testing.assert_series_equal(
            hour_to_float(pd.Series(["01:30"])), pd.Series([1.5])
        )
        pd.testing.assert_frame_equal(
            hour_to_float(pd.DataFrame({"a": ["02:30"]})), pd.DataFrame({"a": [2.5]})
        )

    def test_get_reversed_dict(self):
        dictionary = {"a": 2, "b": 3, "c": 4}

        assert get_reversed_dict(dictionary, 2) == {2: "a"}
        assert get_reversed_dict(dictionary, [2, 3]) == {2: "a", 3: "b"}

    def test_get_data_blocks(self):
        toy_df = pd.DataFrame(
            {"data_1": np.random.randn(24), "data_2": np.random.randn(24)},
            index=pd.date_range("2009-01-01", freq="h", periods=24),
        )

        toy_df.loc["2009-01-01 01:00:00", "data_1"] = np.nan
        toy_df.loc["2009-01-01 10:00:00":"2009-01-01 12:00:00", "data_1"] = np.nan
        toy_df.loc["2009-01-01 15:00:00":"2009-01-01 23:00:00", "data_2"] = np.nan

        res = get_data_blocks(
            toy_df,
            is_null=False,
            lower_td_threshold="1h30min",
            upper_td_threshold="8h",
        )
        assert len(res["data_1"]) == 1

        res = get_data_blocks(toy_df, is_null=True)
        assert len(res["combination"]) == 3
        pd.testing.assert_index_equal(
            res["data_1"][0], pd.DatetimeIndex(["2009-01-01 01:00:00"])
        )
        pd.testing.assert_index_equal(
            res["data_2"][0], pd.date_range("2009-01-01 15:00:00", freq="h", periods=9)
        )

        res = get_data_blocks(toy_df, is_null=True, lower_td_threshold="1h30min")
        assert len(res["data_1"]) == 1

        res = get_data_blocks(toy_df, return_combination=False)
        assert "combination" not in res.keys()

        # Remove timestamps to get indexes wtihout frequency
        toy_df.drop(
            pd.date_range("2009-01-01 02:00:00", "2009-01-01 04:00:00", freq="h"),
            axis=0,
            inplace=True,
        )

        # The gap from 01:00:00 to 04:00:00 shall be identified.
        res = get_data_blocks(toy_df, is_null=True, lower_td_threshold="3h")
        assert len(res["data_1"]) == 2

    def test_outer_timestamps(self):
        ref_index = pd.date_range("2009-01-01", freq="d", periods=5)
        idx = pd.date_range("2009-01-02", freq="d", periods=2)
        start, end = get_outer_timestamps(idx, ref_index)

        assert start == pd.to_datetime("2009-01-01")
        assert end == pd.to_datetime("2009-01-04")

        start, end = get_outer_timestamps(ref_index, ref_index)
        assert start == ref_index[0]
        assert end == ref_index[-1]

    def test_get_gaps_gte_mask(self):
        toy_series = pd.Series(
            np.random.randn(24),
            index=pd.date_range("2009", freq="h", periods=24),
            name="data",
        )

        toy_holes = toy_series.copy()
        toy_holes.loc["2009-01-01 09:00:00"] = np.nan
        toy_holes.loc["2009-01-01 11:00:00":"2009-01-01 13:00:00"] = np.nan
        toy_holes.loc["2009-01-01 19:00:00":"2009-01-01 23:00:00"] = np.nan

        res_1 = get_gaps_gte_mask(toy_holes)
        res_2 = get_gaps_lte_mask(toy_holes)

        np.testing.assert_array_equal(res_1, res_2)

        res_gte = get_gaps_gte_mask(toy_holes, size="3h")
        ref_gte = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
            ]
        )

        np.testing.assert_array_equal(res_gte, ref_gte)

        res_lte = get_gaps_lte_mask(toy_holes, size="3h")
        ref_lte = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )

        np.testing.assert_array_equal(res_lte, ref_lte)

        assert True