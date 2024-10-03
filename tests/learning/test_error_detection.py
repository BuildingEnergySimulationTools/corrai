import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from corrai.learning.error_detection import (
    timedelta_to_int,
    STLEDetector,
    SkSTLForecast,
)

RESOURCES_PATH = Path(__file__).parent / "resources"


class TestErrorDetection:
    def test_timedelta_to_int(self):
        X = pd.DataFrame(
            {"a": np.arange(10 * 6 * 24)},
            index=pd.date_range(dt.datetime.now(), freq="10min", periods=10 * 6 * 24),
        )

        assert timedelta_to_int("24h", X) == 144
        assert timedelta_to_int(144, X) == 144
        assert timedelta_to_int(dt.timedelta(hours=24), X) == 144

    def test_stl_e_detector(self):
        # A temperature timeseries with artificial errors (+0.7°C) at given time steps
        data = pd.read_csv(
            RESOURCES_PATH / "stl_data.csv", index_col=0, parse_dates=True
        )
        data = data.asfreq("15min")

        stl = STLEDetector(
            period="24h",
            trend="1d",
            stl_kwargs={"robust": True},
            absolute_threshold=0.6,
        )

        res = stl.predict(data)

        pd.testing.assert_index_equal(res.index, data.index)
        pd.testing.assert_index_equal(res.columns, data.columns)

        # Check that the 3 errors are found
        assert res.sum().iloc[0] == 3

    def test_stl_forecaster(self):
        index = pd.date_range("2009-01-01", "2009-12-31 23:00:00", freq="h")
        cumsum_second = np.arange(
            start=0, stop=(index[-1] - index[0]).total_seconds() + 1, step=3600
        )
        annual = 5 * -np.cos(
            2 * np.pi / dt.timedelta(days=360).total_seconds() * cumsum_second
        )
        daily = 5 * np.sin(
            2 * np.pi / dt.timedelta(days=1).total_seconds() * cumsum_second
        )
        toy_series = pd.Series(annual + daily + 5, index=index)

        toy_df = pd.DataFrame({"Temp_1": toy_series, "Temp_2": toy_series * 1.25 + 2})

        forecaster = SkSTLForecast(
            period="24h",
            trend="15d",
            ar_kwargs=dict(order=(1, 1, 0), trend="t"),
            backcast=False,
        )

        forecaster.fit(toy_df["2009-01-24":"2009-07-24"])

        reg_score = forecaster.score(
            toy_df["2009-07-27":"2009-07-30"], toy_df["2009-07-27":"2009-07-30"]
        )
        assert reg_score > 0.99

        backcaster = SkSTLForecast(
            period="24h",
            trend="15d",
            ar_kwargs=dict(order=(1, 1, 0), trend="t"),
            backcast=True,
        )

        backcaster.fit(toy_df["2009-01-24":"2009-07-24"])

        reg_score = backcaster.score(
            toy_df["2009-01-20":"2009-01-22"], toy_df["2009-01-20":"2009-01-22"]
        )
        assert reg_score > 0.99
