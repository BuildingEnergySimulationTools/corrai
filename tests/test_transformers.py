import datetime as dt

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from sklearn.preprocessing import StandardScaler

from corrai.transformers import (
    PdAddTimeLag,
    PdApplyExpression,
    PdColumnResampler,
    PdCombineColumns,
    PdDropThreshold,
    PdDropTimeGradient,
    PdDropna,
    PdFillNa,
    PdBfill,
    PdFfill,
    PdGaussianFilter1D,
    PdIdentity,
    PdRenameColumns,
    PdResampler,
    PdSkTransformer,
    PdTimeGradient,
    PdAddSineWave,
)


class TestCustomTransformers:
    def test_pd_identity(self):
        df = pd.DataFrame({"a": [1.0]})

        identity = PdIdentity()
        res = identity.fit_transform(df)

        assert df.columns == identity.get_feature_names_out()
        pd.testing.assert_frame_equal(df, res)

    def test_pd_dropna(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]})

        ref = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        dropper = PdDropna(how="any")

        dropper.fit(df)
        assert list(df.columns) == list(dropper.get_feature_names_out())
        pd.testing.assert_index_equal(df.index, dropper.index)
        pd.testing.assert_frame_equal(dropper.transform(df), ref)

    def test_pd_rename_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]})

        new_cols = ["c", "d"]

        renamer = PdRenameColumns(new_names=new_cols)

        renamer.fit(df)
        assert list(df.columns) == list(renamer.get_feature_names_out())
        pd.testing.assert_index_equal(df.index, renamer.index)
        assert list(renamer.transform(df).columns) == new_cols

        inversed = renamer.inverse_transform(np.zeros((2, 2)))
        assert list(inversed.columns) == new_cols

    def test_pd_sk_transformer(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        scaler = PdSkTransformer(StandardScaler())
        to_test = scaler.fit_transform(df)

        ref = pd.DataFrame({"a": [-1.0, 1.0], "b": [-1.0, 1.0]})

        pd.testing.assert_frame_equal(to_test, ref)
        assert list(df.columns) == list(scaler.get_feature_names_out())

        to_inverse = to_test.to_numpy()
        pd.testing.assert_frame_equal(scaler.inverse_transform(to_inverse), df)

    def test_pd_drop_threshold(self):
        df = pd.DataFrame(
            {"col1": [1, 2, 3, np.nan, 4], "col2": [1, np.nan, np.nan, 4, 5]}
        )

        ref = pd.DataFrame(
            {"col1": [np.nan, 2, 3, np.nan, 4], "col2": [np.nan, np.nan, np.nan, 4, 5]}
        )

        dropper = PdDropThreshold(lower=1.1, upper=5)
        dropper.fit(df)

        assert list(df.columns) == list(dropper.get_feature_names_out())

        pd.testing.assert_frame_equal(dropper.transform(df), ref)

        # check do nothing
        dropper = PdDropThreshold()
        pd.testing.assert_frame_equal(dropper.transform(df), df)

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

        pd.testing.assert_frame_equal(ref, dropper.fit_transform(df))

        # check do nothing
        dropper = PdDropTimeGradient()
        pd.testing.assert_frame_equal(dropper.transform(df), df)

    def test_pd_apply_expression(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        ref = pd.DataFrame({"a": [2.0, 4.0], "b": [6.0, 8.0]})

        transformer = PdApplyExpression("X * 2")

        pd.testing.assert_frame_equal(ref, transformer.fit_transform(df))

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

        pd.testing.assert_frame_equal(ref, derivator.fit_transform(test), rtol=0.01)

    def test_pd_ffill(self):
        test = pd.DataFrame(
            {
                "cpt1": [0.0, np.nan, 2.0, 2.0, np.nan, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            }
        )

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, 2.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 2.0, 3.0],
            }
        )

        filler = PdFfill()
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test))

    def test_pd_bfill(self):
        test = pd.DataFrame(
            {
                "cpt1": [0.0, np.nan, 2.0, 2.0, np.nan, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            }
        )

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 2.0, 2.0, 2.0, 3.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            }
        )

        filler = PdBfill()
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test))

    def test_pd_fill_na(self):
        test = pd.DataFrame(
            {
                "cpt1": [0.0, np.nan, 2.0, 2.0, np.nan, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            }
        )

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, 0.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 0.0, 3.0],
            }
        )

        filler = PdFillNa(value=0.0)
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

        pd.testing.assert_frame_equal(ref, transformer.fit_transform(test))

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
                "col0": [400.0, 900.0],
                "col1": [2.0, 7.0],
                "col2": [0.56239, 0.47789],
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

        column_resampler = PdColumnResampler(
            rule="5H",
            columns_method=[(["col2"], np.mean), (["col1"], np.mean)],
            remainder="drop",
        )

        pd.testing.assert_frame_equal(
            ref[["col2", "col1"]],
            column_resampler.fit_transform(df).astype("float"),
            atol=0.01,
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

    def test_pd_gaussian_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        gfilter = PdGaussianFilter1D()

        to_test = gfilter.fit_transform(df)

        np.testing.assert_almost_equal(
            gaussian_filter1d(
                df.to_numpy()[:, 0].T, sigma=5, mode="nearest", truncate=4.0
            ),
            to_test.to_numpy()[:, 0],
            decimal=5,
        )

        assert list(to_test.columns) == list(df.columns)

    def test_pd_combine_columns(self):
        x_in = pd.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})

        trans = PdCombineColumns(
            columns_to_combine=["a", "b"],
            function=np.sum,
            function_kwargs={"axis": 1},
            drop_columns=True,
        )

        pd.testing.assert_frame_equal(
            trans.fit_transform(x_in), pd.DataFrame({"c": [1, 2], "combined": [2, 4]})
        )

        ref = x_in.copy()
        ref["combined"] = [2, 4]
        trans.drop_columns = False

        pd.testing.assert_frame_equal(trans.fit_transform(x_in), ref)

    def test_pd_add_sine_wave(self):
        test_df = pd.DataFrame(
            data=np.arange(24),
            index=pd.date_range("2009-01-01 00:00:00", freq="H", periods=24),
            columns=["feat_1"],
        )

        signal = PdAddSineWave(frequency=1 / (24 * 3600))

        res = signal.fit_transform(test_df)

        test_df["Sine_f_1.1574074074074073e-05"] = [
            0.00000,
            0.25882,
            0.50000,
            0.70711,
            0.86603,
            0.96593,
            1.00000,
            0.96593,
            0.86603,
            0.70711,
            0.50000,
            0.25882,
            0.00000,
            -0.25882,
            -0.50000,
            -0.70711,
            -0.86603,
            -0.96593,
            -1.00000,
            -0.96593,
            -0.86603,
            -0.70711,
            -0.50000,
            -0.25882,
        ]

        pd.testing.assert_frame_equal(res, test_df)

        assert True

    def test_pd_time_window(self):
        test_timeseries = pd.DataFrame(
            {
                "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "feat_2": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            },
            index=pd.date_range("2009-01-01", freq="H", periods=7),
        )
        test_timeseries["label"] = (
            test_timeseries["feat_1"] * 2 + test_timeseries["feat_1"] * 3
        )

        windower = PdTimeWindow(
            feat_input_width=3,
            label_output_width=2,
            feat_label_shift=3,
            sampling_shift=1,
            labels_names="label",
        )

        x_ref = pd.DataFrame(
            np.array(
                [
                    [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
                ]
            ),
            columns=[
                "feat_1_n0",
                "feat_1_n1",
                "feat_1_n2",
                "feat_2_n0",
                "feat_2_n1",
                "feat_2_n2",
            ],
            index=pd.DatetimeIndex(
                ["2009-01-01 00:00:00", "2009-01-01 01:00:00", "2009-01-01 02:00:00"]
            ),
        )

        x_tot_test, y_to_test = windower.fit_transform(test_timeseries)

        pd.testing.assert_frame_equal(x_tot_test, x_ref)

    # def test_pd_time_window(self):
    #     test_timeseries = pd.DataFrame(
    #         {
    #             "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    #             "feat_2": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    #         },
    #         index=pd.date_range("2009-01-01", freq="H", periods=7),
    #     )
    #     test_timeseries["label"] = (
    #         test_timeseries["feat_1"] * 2 + test_timeseries["feat_1"] * 3
    #     )
    #
    #     windower = PdTimeWindow(
    #         feat_input_width=3,
    #         label_output_width=2,
    #         feat_label_shift=3,
    #         sampling_shift=1,
    #         labels_names="label",
    #     )
    #
    #     x_ref = pd.DataFrame(
    #         np.array(
    #             [
    #                 [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    #                 [2.0, 3.0, 4.0, 4.0, 5.0, 6.0],
    #                 [3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
    #             ]
    #         ),
    #         columns=[
    #             "feat_1_n0",
    #             "feat_1_n1",
    #             "feat_1_n2",
    #             "feat_2_n0",
    #             "feat_2_n1",
    #             "feat_2_n2",
    #         ],
    #         index=pd.DatetimeIndex(
    #             ["2009-01-01 00:00:00", "2009-01-01 01:00:00", "2009-01-01 02:00:00"]
    #         ),
    #     )
    #
    #     x_tot_test, y_to_test = windower.fit_transform(test_timeseries)
    #
    #     assert pd.testing.assert_frame_equal(x_tot_test, x_ref)