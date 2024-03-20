import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from corrai.learning.cluster import KdeSetPointIdentificator
from corrai.learning.cluster import get_hours_switch
from corrai.learning.cluster import plot_kde_set_point, plot_time_series_kde
from corrai.learning.cluster import (
    set_point_identifier,
    moving_window_set_point_identifier,
)
import corrai.transformers as ct

import datetime as dt
from pathlib import Path

FILES_PATH = Path(__file__).parent.parent / "resources"


class TestLearning:
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

    def test_switch(self):
        test_series = pd.Series(
            [0, 0, 1000, 1020, 0.1, 0.3],
            index=pd.date_range(
                "2009-01-01 00:00:00", freq="h", periods=6, tz="Europe/Paris"
            ),
            name="flwr",
        )
        assert True

        assert get_hours_switch(test_series, 200, switch="positive") == ["02:00"]
        assert get_hours_switch(test_series, 200, switch="negative") == ["04:00"]
        assert get_hours_switch(test_series, 200, switch="both") == ["02:00", "04:00"]

    def test_plot_kde_set_point(self):
        # Generate test data
        x = pd.Series(np.random.randn(100), name="x")

        estimator = KdeSetPointIdentificator()
        estimator.fit(x.to_frame())

        # Call the function with default arguments
        plot_kde_set_point(x.to_frame(), estimator)

        # Call the function with non-default arguments
        plot_kde_set_point(
            x.to_frame(),
            estimator,
            title="Clustered Timeseries",
            y_label="test_lab",
        )

    def test_plot_time_series_kde(self):
        # Generate test data
        x = pd.Series(np.random.randn(100), name="x")

        # Call the function with default arguments
        plot_time_series_kde(x.to_frame())

        # Call the function with non-default arguments
        plot_time_series_kde(
            x.to_frame(),
            title="Likelihood and data",
            x_label="x",
            scaled=False,
            bandwidth=0.2,
            xbins=200,
        )

    def test_set_point_identifier(self):
        f_data = pd.read_csv(
            FILES_PATH / "kde_false_data.csv", index_col=0, parse_dates=True
        )

        res = set_point_identifier(
            f_data,
            estimator=KdeSetPointIdentificator(bandwidth=0.1, lik_filter=0.6),
        )

        ref = pd.DataFrame(
            {
                "a": {
                    (
                        pd.Period("2009-01-01 00:00", "h"),
                        "set_point_0",
                    ): 122.65772294951199,
                    (
                        pd.Period("2009-01-01 00:00", "h"),
                        "set_point_1",
                    ): 241.75480686502476,
                    (
                        pd.Period("2009-01-01 00:00", "h"),
                        "set_point_2",
                    ): 387.6906702544559,
                }
            }
        )

        pd.testing.assert_frame_equal(res, ref)

    def test_moving_window_set_point_identifier(self):
        f_data = pd.read_csv(
            FILES_PATH / "kde_false_data.csv", index_col=0, parse_dates=True
        )

        res = moving_window_set_point_identifier(
            f_data,
            window_size=dt.timedelta(hours=10),
            slide_size=dt.timedelta(hours=10),
            estimator=KdeSetPointIdentificator(),
        )

        ref = pd.DataFrame(
            {
                "a": {
                    (
                        pd.Period("2009-01-01 00:00", "h"),
                        "set_point_0",
                    ): 122.28125863629094,
                    (
                        pd.Period("2009-01-01 00:00", "h"),
                        "set_point_1",
                    ): 384.76032496023265,
                    (
                        pd.Period("2009-01-01 10:00", "h"),
                        "set_point_0",
                    ): 245.13766266266265,
                }
            }
        )

        pd.testing.assert_frame_equal(res, ref)
