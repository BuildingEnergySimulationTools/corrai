from corrai.base import Model
from corrai.sensitivity import SAnalysis
import pandas as pd
import numpy as np

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 05:00:00",
    "timestep": "H",
}

PARAMETER_LIST = [
    {"name": "x1", "interval": (1.0, 3.0), "type": "Real"},
    {"name": "x2", "interval": (1.0, 3.0), "type": "Real"},
    {"name": "x3", "interval": (1.0, 3.0), "type": "Real"},
]


class TestModel(Model):
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


class TestSensitivity:
    def test_sanalysis(self):
        model = TestModel()

        res = model.simulate(
            parameter_dict={"x1": 1.0, "x2": 2.0, "x3": 3.0},
            simulation_options=SIMULATION_OPTIONS,
        )

        sa_analysis = SAnalysis(
            model=model, parameters_list=PARAMETER_LIST, method="Sobol"
        )

        sa_analysis.draw_sample(1, sampling_kwargs={"calc_second_order": False})

        sample_ref = pd.DataFrame(
            {
                "x1": [1.18750, 2.31250, 1.18750, 1.18750, 2.31250],
                "x2": [1.93750, 1.93750, 1.56250, 1.93750, 1.56250],
                "x3": [1.93750, 1.93750, 1.93750, 2.93750, 2.93750],
            }
        )

        pd.testing.assert_frame_equal(sa_analysis.sample, sample_ref)

        assert True
