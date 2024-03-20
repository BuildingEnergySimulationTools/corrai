import pandas as pd
from corrai.math import time_gradient, time_integrate, aggregate_time_series
from corrai.metrics import nmbe


class TestMath:
    def test_time_gradient(self):
        test = (
            pd.Series(
                [0, 1, 2, 2, 2, 3],
                index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6),
                name="cpt1",
            )
            * 3600
        )

        ref = pd.DataFrame(
            {"cpt1": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0]},
            index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6),
        )

        to_test = time_gradient(test)

        pd.testing.assert_frame_equal(ref, to_test, rtol=0.01)

        test = (
            pd.DataFrame(
                {"cpt1": [0, 1, 2, 2, 2, 3], "cpt2": [0, 1, 2, 2, 2, 3]},
                index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6),
            )
            * 3600
        )

        ref = pd.DataFrame(
            {
                "cpt1": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
                "cpt2": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
            },
            index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6),
        )

        to_test = time_gradient(test)

        pd.testing.assert_frame_equal(ref, to_test, rtol=0.01)

    def test_time_integrate(self):
        test = pd.Series(
            [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
            index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6),
            name="cpt",
        )

        ref = pd.Series({"cpt": 3.0})

        pd.testing.assert_series_equal(ref, time_integrate(test) / 3600)

        test = pd.DataFrame(
            {
                "cpt1": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
                "cpt2": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
            },
            index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6),
        )

        ref = pd.Series({"cpt1": 3.0, "cpt2": 3.0})

        pd.testing.assert_series_equal(ref, time_integrate(test) / 3600)

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
