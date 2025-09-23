import itertools

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from corrai.surrogate import ModelTrainer, MultiModelSO, StaticScikitModel


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


class TestScikitWrapper:
    def test_scikit_wrapper(self):
        ds = pd.DataFrame(
            {
                "x_1": np.arange(0.0, 10.0, 1),
                "x_2": np.arange(10.0, 20.0, 1),
                "y": 4.0 * np.arange(10.0, 20.0, 1) + 2.0 * np.arange(0.0, 10.0, 1),
            }
        )

        in_df = {"x_1": 2.0, "x_2": 4.0}

        ref_df = pd.DataFrame({"y": 28.0}, index=[0])

        mumoso = MultiModelSO()
        mumoso.fit(ds[["x_1", "x_2"]], ds["y"])
        stat_mod = StaticScikitModel(mumoso)
        pd.testing.assert_frame_equal(stat_mod.simulate(in_df), ref_df)

        line_reg = LinearRegression()
        line_reg.fit(ds[["x_1", "x_2"]], ds["y"])
        scikit_mod = StaticScikitModel(line_reg, target_name="y")
        pd.testing.assert_frame_equal(scikit_mod.simulate(in_df), ref_df)
