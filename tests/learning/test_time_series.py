import numpy as np
import pandas as pd
from corrai.learning.model_selection import (
    time_series_sampling,
    sequences_train_test_split,
)
from corrai.learning.time_series import TsLinearModel


class TestTimeSeries:
    def test_ts_linear_model(self):
        datas = pd.DataFrame({"data": np.arange(10000)})

        sequences = time_series_sampling(datas, sequence_length=6)
        x_train, x_test, y_train, y_test = sequences_train_test_split(
            sequences, n_steps_history=4, n_steps_future=2
        )

        ts_linear = TsLinearModel()
        ts_linear.fit(x_train, y_train)

        assert ts_linear.score(x_test, y_test) > 0.999
