import numpy as np
import pandas as pd
import datetime as dt

from sklearn.preprocessing import StandardScaler

from corrai.custom_transformers import PdDropna
from corrai.custom_transformers import PdSkTransformer
from corrai.custom_transformers import PdDropThreshold
from corrai.custom_transformers import PdDropTimeGradient
from corrai.custom_transformers import PdApplyExpression
from corrai.custom_transformers import PdTimeGradient
from corrai.custom_transformers import PdFillNa
from corrai.custom_transformers import PdResampler
from corrai.custom_transformers import PdColumnResampler
from corrai.custom_transformers import PdAddTimeLag


class TestCustomTransformers:
    def test_pd_dropna(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]})

        ref = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        dropper = PdDropna(how="any")

        pd.testing.assert_frame_equal(dropper.transform(df), ref)

    def test_pd_scaler(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        scaler = PdSkTransformer(StandardScaler())
        to_test = scaler.fit_transform(df)

        ref = pd.DataFrame({"a": [-1.0, 1.0], "b": [-1.0, 1.0]})

        pd.testing.assert_frame_equal(to_test, ref)

    def test_pd_drop_threshold(self):
        df = pd.DataFrame(
            {"col1": [1, 2, 3, np.nan, 4], "col2": [1, np.nan, np.nan, 4, 5]}
        )

        ref = pd.DataFrame(
            {"col1": [np.nan, 2, 3, np.nan, 4], "col2": [np.nan, np.nan, np.nan, 4, 5]}
        )

        dropper = PdDropThreshold(lower=1.1, upper=5)

        pd.testing.assert_frame_equal(dropper.transform(df), ref)

    def test_pd_drop_time_gradient(self):
        time_index = pd.date_range("2021-01-01 00:00:00", freq="H", periods=8)

        df = pd.DataFrame(
            {
                "dumb_column": [5, 5.1, 5.1, 6, 7, 22, 6, 5],
                "dumb_column2": [5, 5, 5.1, 6, 22, 6, np.nan, 6],
            },
            index=time_index,
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [5.0, 5.1, np.nan, 6.0, 7.0, np.nan, 6.0, 5.0],
                "dumb_column2": [5.0, np.nan, 5.1, 6.0, np.nan, 6.0, np.nan, np.nan],
            },
            index=time_index,
        )

        dropper = PdDropTimeGradient(lower_rate=0, upper_rate=0.004)

        pd.testing.assert_frame_equal(ref, dropper.transform(df))

    def test_pd_apply_expression(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        ref = pd.DataFrame({"a": [2.0, 4.0], "b": [6.0, 8.0]})

        transformer = PdApplyExpression("X * 2")

        pd.testing.assert_frame_equal(ref, transformer.transform(df))

    def test_pd_time_gradient(self):
        test = (
            pd.DataFrame(
                {"cpt1": [0, 1, 2, 2, 2, 3], "cpt2": [0, 1, 2, 2, 2, 3]},
                index=pd.date_range("2009-01-01 00:00:00", freq="10S", periods=6),
            )
            * 3600
        )

        ref = pd.DataFrame(
            {
                "cpt1": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
                "cpt2": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
            },
            index=pd.date_range("2009-01-01 00:00:00", freq="10S", periods=6),
        )

        derivator = PdTimeGradient()

        pd.testing.assert_frame_equal(ref, derivator.transform(test), rtol=0.01)

    def test_pd_fill_na(self):
        test = pd.DataFrame(
            {"cpt1": [0, np.nan, 2, 2, np.nan, 3], "cpt2": [0, 1, 2, 2, np.nan, 3]}
        )

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 2.0, 2.0, 2.0, 3.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            }
        )

        filler = PdFillNa(method="bfill")

        pd.testing.assert_frame_equal(ref, filler.fit_transform(test))

    def test_pd_resampler(self):
        test = pd.DataFrame(
            {"col": [1.0, 2.0, 3.0]},
            index=pd.date_range("2009-01-01 00:00:00", freq="H", periods=3),
        )

        ref = pd.DataFrame(
            {"col": [2.0]},
            index=pd.DatetimeIndex(["2009-01-01"], dtype="datetime64[ns]", freq="3H"),
        )

        transformer = PdResampler(rule="3H", method=np.mean)

        pd.testing.assert_frame_equal(ref, transformer.transform(test))

    def test_pd_columns_resampler(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "col0": np.arange(10) * 100,
                "col1": np.arange(10),
                "col2": np.random.random(10),
                "col3": np.random.random(10) * 10,
            },
            index=pd.date_range("2009-01-01", freq="H", periods=10),
        ).astype("float")

        ref = pd.DataFrame(
            {
                "col2": [0.56239, 0.47789],
                "col1": [2.0, 7.0],
                "col0": [400.0, 900.0],
                "col3": [9.69910, 5.24756],
            },
            index=pd.DatetimeIndex(
                ["2009-01-01 00:00:00", "2009-01-01 05:00:00"],
                dtype="datetime64[ns]",
                freq="5H",
            ),
        ).astype("float")

        column_resampler = PdColumnResampler(
            rule="5H",
            columns_method=[(["col2"], np.mean), (["col1"], np.mean)],
            remainder=np.max,
        )

        pd.testing.assert_frame_equal(
            ref, column_resampler.fit_transform(df).astype("float"), atol=0.01
        )

    def test_pd_add_time_lag(self):
        df = pd.DataFrame(
            {
                "col0": np.arange(2),
                "col1": np.arange(2) * 10,
            },
            index=pd.date_range("2009-01-01", freq="H", periods=2),
        )

        ref = pd.DataFrame(
            {
                "col0": [1.0],
                "col1": [10.0],
                "1:00:00_col0": [0.0],
                "1:00:00_col1": [0.0],
            },
            index=pd.DatetimeIndex(
                ["2009-01-01 01:00:00"], dtype="datetime64[ns]", freq="H"
            ),
        )

        lager = PdAddTimeLag(time_lag=dt.timedelta(hours=1), drop_resulting_nan=True)

        pd.testing.assert_frame_equal(ref, lager.fit_transform(df))