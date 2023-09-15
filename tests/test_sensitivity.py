from corrai.sensitivity import SAnalysis
from tests.resources.pymodels import Ishigami
import numpy as np

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 05:00:00",
    "timestep": "H",
}

PARAMETER_LIST = [
    {
        "name": "x1",
        "interval": (-3.14159265359, 3.14159265359),
        "type": "Real",
    },
    {
        "name": "x2",
        "interval": (-3.14159265359, 3.14159265359),
        "type": "Real",
    },
    {
        "name": "x3",
        "interval": (-3.14159265359, 3.14159265359),
        "type": "Real",
    },
]


class TestSensitivity:
    def test_sanalysis(self):
        model = Ishigami()

        sa_analysis = SAnalysis(parameters_list=PARAMETER_LIST, method="Sobol")

        sa_analysis.draw_sample(1, sampling_kwargs={"calc_second_order": True})

        sa_analysis.evaluate(model, SIMULATION_OPTIONS, n_cpu=4)

        sa_analysis.analyze(
            "res", sensitivity_method_kwargs={"calc_second_order": True}
        )

        np.testing.assert_almost_equal(
            sa_analysis.sensitivity_results["S1"],
            np.array([0.26933607, 1.255609, -0.81162613]),
            decimal=3,
        )
