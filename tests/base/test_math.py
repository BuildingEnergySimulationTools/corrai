import pandas as pd
from corrai.base.math import aggregate_time_series
from corrai.metrics import nmbe


class TestMath:
    def test_aggregate_time_series(self):
        sim_res = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4]},
            index=pd.date_range("2009-01-01", freq="h", periods=2),
        )
        ref_df = pd.DataFrame(
            {"a": [1, 1], "b": [3, 4]},
            index=pd.date_range("2009-01-01", freq="h", periods=2),
        )

        expected_default = pd.Series([1.5, 3.5], index=["a", "b"])

        expected_nmbe = pd.Series([50.0, 0.0], index=["a", "b"])

        pd.testing.assert_series_equal(aggregate_time_series(sim_res), expected_default)
        pd.testing.assert_series_equal(
            aggregate_time_series(sim_res, agg_method=nmbe, reference_df=ref_df),
            expected_nmbe,
        )
