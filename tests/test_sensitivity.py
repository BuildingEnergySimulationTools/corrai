import numpy as np

from corrai.base.parameter import Parameter
from corrai.sensitivity import Method
from corrai.sensitivity import SAnalysis
from tests.resources.pymodels import Ishigami

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 05:00:00",
    "timestep": "H",
}

PARAMETER_LIST = [
    {
        Parameter.NAME: "x1",
        Parameter.INTERVAL: (-3.14159265359, 3.14159265359),
        Parameter.TYPE: "Real",
    },
    {
        Parameter.NAME: "x2",
        Parameter.INTERVAL: (-3.14159265359, 3.14159265359),
        Parameter.TYPE: "Real",
    },
    {
        Parameter.NAME: "x3",
        Parameter.INTERVAL: (-3.14159265359, 3.14159265359),
        Parameter.TYPE: "Real",
    },
]


class TestSensitivity:
    def test_sanalysis(self):
        model = Ishigami()

        sa_analysis = SAnalysis(parameters_list=PARAMETER_LIST, method=Method.SOBOL)

        sa_analysis.draw_sample(
            1, sampling_kwargs={"calc_second_order": True, "seed": 42}
        )

        sa_analysis.evaluate(model, SIMULATION_OPTIONS, n_cpu=4)

        sa_analysis.analyze(
            "res", sensitivity_method_kwargs={"calc_second_order": True}
        )

        np.testing.assert_almost_equal(
            sa_analysis.sensitivity_results["S1"],
            np.array([1.022, 2.229, -1.412]),
            decimal=3,
        )

        sa_analysis = SAnalysis(parameters_list=PARAMETER_LIST, method=Method.MORRIS)

        sa_analysis.draw_sample(15)

        sa_analysis.evaluate(model, SIMULATION_OPTIONS, n_cpu=4)

        sa_analysis.analyze(indicator="res")

        # Problem is highly non linear and result do not converge.
        # We are just checking the execution
