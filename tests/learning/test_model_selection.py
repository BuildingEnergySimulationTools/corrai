from pathlib import Path

import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

from corrai.learning.model_selection import (
    ModelTrainer,
    MultiModelSO,
    time_series_sampling,
    sequences_train_test_split,
)

FILES_PATH = Path(__file__).parent / "resources"


class TestLearning:
    def test_mumoso_and_trainer(self):
        x = np.arange(0.0, 10.0, 1)
        y = 4.0 + 2.0 * x

        common_index = pd.date_range("2009-01-01", freq="h", periods=10)
        x_df = pd.DataFrame(x, index=common_index)
        y_series = pd.Series(y, index=common_index)

        model = MultiModelSO(cv=2, fine_tuning=True, n_jobs=-1, random_state=42)

        trainer = ModelTrainer(model)
        trainer.train(x_df, y_series)

        assert trainer.test_nmbe_score < 10e-10
        assert trainer.test_cvrmse_score < 10e-10

        pd.testing.assert_frame_equal(model.predict(trainer.x_test), trainer.y_test)

        # Check ability to bypass DataFrame conversion
        x_dict = {
            "feat1": ["win1", "win2"],
            "feat2": ["wall1", "wall2", "wall3"],
            "feat3": ["roof1", "roof2", "roof3", "roof4"],
        }

        x_df = pd.DataFrame(list(set(itertools.product(*list(x_dict.values())))))
        y = pd.Series(np.random.randn(x_df.shape[0]))

        model_pipe = make_pipeline(OneHotEncoder(), MultiModelSO(fine_tuning=False))
        trainer = ModelTrainer(model_pipe)
        trainer.train(X=x_df, y=y)

        model_pipe.predict(x_df)

        assert True

    def test_time_series_sampling(self):
        ts = pd.DataFrame(
            {"feat_1": np.arange(10), "feat_2": 10 * np.arange(10)},
            index=pd.date_range("2009-01-01 00:00:00", freq="H", periods=10),
        )

        res = time_series_sampling(ts, sequence_length=4, shuffle=False)

        np.testing.assert_array_equal(
            res[0], np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        )

    def test_sequences_train_test_split(self):
        ts = pd.DataFrame(
            {"feat_1": np.arange(10), "feat_2": 10 * np.arange(10)},
            index=pd.date_range("2009-01-01 00:00:00", freq="H", periods=10),
        )

        res = time_series_sampling(ts, sequence_length=4, shuffle=False)

        x_train, x_test, y_train, y_test = sequences_train_test_split(
            data=res,
            targets_index=0,
            n_steps_history=3,
            n_steps_future=1,
            test_size=0.2,
            shuffle=False,
        )

        np.testing.assert_array_equal(
            x_train[0], np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
        )
        np.testing.assert_array_equal(
            y_train, np.array([[3.0], [4.0], [5.0], [6.0], [7.0]])
        )
        np.testing.assert_array_equal(
            x_test[0], np.array([[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]])
        )
        np.testing.assert_array_equal(y_test, np.array([[8.0], [9.0]]))
