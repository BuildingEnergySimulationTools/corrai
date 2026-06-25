import pandas as pd
import numpy as np

from corrai.base.math import aggregate_time_series, apply_cuts


class TestMath:
    def test_apply_cuts_series(self):
        cuts = [
            ("2009-01-01 00:00:00", "2009-01-01 2:00:00"),
            ("2009-01-01 05:00:00", "2009-01-01 07:00:00"),
        ]

        index = pd.date_range("2009-01-01", freq="h", periods=8)

        s = pd.Series(range(len(index)), index=index)

        result = apply_cuts(s, cuts)

        mask = pd.Series(False, index=s.index)
        for start, end in cuts:
            mask |= (s.index >= start) & (s.index <= end)

        pd.testing.assert_series_equal(
            result,
            s.loc[mask],
        )

    def test_apply_cuts_frame(self):
        cuts = [
            ("2009-01-01 00:00:00", "2009-01-01 2:00:00"),
            ("2009-01-01 05:00:00", "2009-01-01 07:00:00"),
        ]

        index = pd.date_range("2009-01-01", freq="h", periods=8)

        df = pd.DataFrame(
            {"a": range(len(index)), "b": range(100, 100 + len(index))},
            index=index,
        )

        result = apply_cuts(df, cuts)

        mask = pd.Series(False, index=df.index)
        for start, end in cuts:
            mask |= (df.index >= start) & (df.index <= end)

        pd.testing.assert_frame_equal(
            result,
            df.loc[mask],
        )

    def test_apply_cuts_none(self):
        index = pd.date_range("2009-01-01", freq="h", periods=8)
        s = pd.Series(range(len(index)), index=index)
        df = pd.DataFrame(
            {"a": range(len(index)), "b": range(100, 100 + len(index))},
            index=index,
        )

        result_series = apply_cuts(s, None)
        result_frame = apply_cuts(df, None)

        pd.testing.assert_series_equal(
            result_series,
            s,
        )

        pd.testing.assert_frame_equal(
            result_frame,
            df,
        )

    def test_aggregate_time_series(self):
        sim_res = pd.Series(
            [
                pd.DataFrame(
                    {"a": [1, 2], "b": [3, 4]},
                    index=pd.date_range("2009-01-01", freq="h", periods=2),
                ),
                pd.DataFrame(
                    {"a": [3, 4], "b": [3, 4]},
                    index=pd.date_range("2009-01-01", freq="h", periods=2),
                ),
            ],
            index=[2, 3],
        )

        ref_df = pd.DataFrame(
            {"a": [1, 1], "b": [3, 4]},
            index=pd.date_range("2009-01-01", freq="h", periods=2),
        )

        res = aggregate_time_series(
            sim_res,
            "a",
            method="mean_absolute_error",
            reference_time_series=ref_df["a"],
            freq="h",
        )

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    pd.Timestamp("2009-01-01 00:00:00"): {2: 0.0, 3: 2.0},
                    pd.Timestamp("2009-01-01 01:00:00"): {2: 1.0, 3: 3.0},
                }
            ),
        )

        res = aggregate_time_series(
            sim_res,
            "a",
            freq="h",
        )

        res_fun = aggregate_time_series(
            sim_res,
            "a",
            method=np.mean,
            freq="h",
        )

        pd.testing.assert_frame_equal(res, res_fun)

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    pd.Timestamp("2009-01-01 00:00:00"): {2: 1.0, 3: 3.0},
                    pd.Timestamp("2009-01-01 01:00:00"): {2: 2.0, 3: 4.0},
                }
            ),
        )

        res = aggregate_time_series(sim_res, "a")
        pd.testing.assert_frame_equal(
            res, pd.Series([1.5, 3.5], index=[2, 3], name="aggregated_a").to_frame()
        )

        res = aggregate_time_series(
            sim_res,
            "a",
            method="mean_absolute_error",
            reference_time_series=ref_df["a"],
        )
        pd.testing.assert_frame_equal(
            res, pd.Series([0.5, 2.5], index=[2, 3], name="aggregated_a").to_frame()
        )
