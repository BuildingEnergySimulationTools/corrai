import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from corrai.learning import KdeSetPointIdentificator
from corrai.learning import get_hours_switch
from corrai.learning import plot_kde_set_point, plot_ts_kde
from corrai.learning import _2d_n_1_dataframer
import corrai.custom_transformers as ct

from pathlib import Path

FILES_PATH = Path(__file__).parent / "resources"


class TestLearning:
    def test__2d_n_1_dataframer(self):
        ref = pd.DataFrame(np.array([1, 2, 3]))

        pd.testing.assert_frame_equal(ref, _2d_n_1_dataframer([1, 2, 3]))
        pd.testing.assert_frame_equal(ref, _2d_n_1_dataframer(np.array([1, 2, 3])))
        pd.testing.assert_frame_equal(
            ref, _2d_n_1_dataframer(np.array([[1], [2], [3]]))
        )
        pd.testing.assert_frame_equal(ref, _2d_n_1_dataframer(ref))

    def test_kde_set_point_identificator(self):
        data = pd.read_csv(
            FILES_PATH / "kde_test_dataset.csv", index_col=0, parse_dates=True
        )

        pipe = Pipeline(
            [
                ("drop_na", ct.PdDropna()),
                ("scaler", ct.PdSkTransformer(StandardScaler())),
                ("kde_cluster", KdeSetPointIdentificator()),
            ]
        )

        pipe.fit(data)

        np.testing.assert_array_almost_equal(
            pipe.named_steps["kde_cluster"].set_points,
            np.array([-0.80092, 1.25735]),
            decimal=3,
        )

        np.testing.assert_array_almost_equal(
            pipe.named_steps["kde_cluster"].set_points_likelihood,
            np.array([2.43101, 1.49684]),
            decimal=3,
        )

        clustered = pipe.predict(pd.Series([4400, 0, 156]).to_frame())

        np.testing.assert_array_almost_equal(clustered, np.array([1, 0, -1]))

    def test_get_hours_switch(self):
        test_series = pd.Series(
            [0, 0, 1000, 1020, 0.1, 0.3],
            index=pd.date_range(
                "2009-01-01 00:00:00", freq="H", periods=6, tz="Europe/Paris"
            ),
            name="flwr",
        )

        res = get_hours_switch(test_series, 200)
        ref = pd.Series(
            [2.0],
            index=pd.date_range(
                "2009-01-01 02:00:00", freq="H", periods=1, tz="Europe/Paris"
            ),
            name="hour_since_beg_day",
        )

        pd.testing.assert_series_equal(res, ref)

        res = get_hours_switch(test_series, 200, switch="negative")
        ref = pd.Series(
            [4.0],
            index=pd.date_range(
                "2009-01-01 04:00:00", freq="H", periods=1, tz="Europe/Paris"
            ),
            name="hour_since_beg_day",
        )

        pd.testing.assert_series_equal(res, ref)

        res = get_hours_switch(test_series, 200, switch="both")
        ref = pd.Series(
            [2.0, 4.0],
            index=pd.DatetimeIndex(
                ["2009-01-01 02:00:00", "2009-01-01 04:00:00"], tz="Europe/Paris"
            ),
            name="hour_since_beg_day",
        )

        pd.testing.assert_series_equal(res, ref)

    def test_plot_kde_set_point(self):
        # Generate test data
        x = pd.Series(np.random.randn(100), name="x")

        estimator = KdeSetPointIdentificator()
        estimator.fit(x.to_frame())

        # Call the function with default arguments
        plot_kde_set_point(estimator, x.to_frame())

        # Call the function with non-default arguments
        plot_kde_set_point(
            estimator,
            x.to_frame(),
            title="Clustered Timeseries",
            y_label="test_lab",
            fit=True,
        )

    def test_plot_ts_kde(self):
        # Generate test data
        x = pd.Series(np.random.randn(100), name="x")

        # Call the function with default arguments
        plot_ts_kde(x.to_frame())

        # Call the function with non-default arguments
        plot_ts_kde(
            x.to_frame(),
            title="Likelihood and data",
            x_label="x",
            scaled=False,
            bandwidth=0.2,
            xbins=200,
        )
