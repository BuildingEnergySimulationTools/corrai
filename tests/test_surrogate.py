import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from corrai.surrogate import ModelTrainer, MultiModelSO


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
