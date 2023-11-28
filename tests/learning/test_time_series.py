import numpy as np
import pandas as pd
from corrai.learning.model_selection import (
    time_series_sampling,
    sequences_train_test_split,
)
from corrai.learning.time_series import (
    TsLinearModel,
    reshape_target_sequence_to_sequence,
)


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

    def test_reshape_sequence_to_sequence(self):
        datas = pd.DataFrame({"data_0": np.arange(10), "data_1": np.arange(10) * 10})

        sequences = time_series_sampling(datas, sequence_length=5, shuffle=False)
        x_train, x_test, y_train, y_test = sequences_train_test_split(
            sequences, n_steps_history=3, n_steps_future=2, shuffle=False
        )

        res = reshape_target_sequence_to_sequence(x_train, y_train)

        np.testing.assert_array_equal(
            res,
            np.array(
                [
                    [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    [[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
                    [[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
                    [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                ]
            ),
        )
