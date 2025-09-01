import pandas as pd
from corrai.base.math import aggregate_time_series


class TestMath:
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
