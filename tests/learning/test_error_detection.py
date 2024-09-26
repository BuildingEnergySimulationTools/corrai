import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from corrai.learning.error_detection import timedelta_to_int, STLEDetector

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
