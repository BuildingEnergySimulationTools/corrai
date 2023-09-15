from corrai.base.model import Model
from corrai.base.simulate import run_models_in_parallel
import pandas as pd
import numpy as np

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 02:00:00",
    "timestep": "H",
}


class ModelTest(Model):
    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        evaluate = lambda x: (
            np.sin(x["x1"])
            + 7.0 * np.power(np.sin(x["x2"]), 2)
            + 0.1 * np.power(x["x3"], 4) * np.sin(x["x1"])
        )

        return pd.DataFrame(
            {"res": [evaluate(parameter_dict)]},
            index=pd.date_range(
                simulation_options["start"],
                simulation_options["end"],
                freq=simulation_options["timestep"],
            ),
        )


class TestSimulate:
    def test_run_models_in_parallel(self):
        model = ModelTest()
        parameters_sample = pd.DataFrame(
            {
                "x1": [1.0, 2.0],
                "x2": [1.0, 2.0],
                "x3": [1.0, 2.0],
            }
        )

        res = run_models_in_parallel(
            model, parameters_sample, SIMULATION_OPTIONS, n_cpu=2
        )

        assert len(res) == 2
        assert res[0][0] == {"x1": 1.0, "x2": 1.0, "x3": 1.0}
        assert res[0][1] == {
            "end": "2009-01-01 02:00:00",
            "start": "2009-01-01 00:00:00",
            "timestep": "H",
        }
        pd.testing.assert_frame_equal(
            res[0][2],
            pd.DataFrame(
                {"res": [5.882132011203685, 5.882132011203685, 5.882132011203685]},
                index=pd.date_range(
                    "2009-01-01 00:00:00", "2009-01-01 02:00:00", freq="H"
                ),
            ),
        )

        assert True
