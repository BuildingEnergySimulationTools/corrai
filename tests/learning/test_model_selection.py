from pathlib import Path

import pandas as pd
import numpy as np

from corrai.learning.model_selection import ModelTrainer, MultiModelSO


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
        pd.testing.assert_frame_equal(model.predict(trainer.x_test), trainer.y_test)
