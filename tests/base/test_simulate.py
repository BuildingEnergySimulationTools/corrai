import pandas as pd

from corrai.base.parameter import Parameter
from corrai.base.simulate import run_simulations
from corrai.base.model import IshigamiDynamic

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 02:00:00",
    "timestep": "h",
}

PARAM_LIST = [
    Parameter("x1", (0.0, 3.0), model_property="x1"),
    Parameter("x2", (0.0, 3.0), model_property="x2"),
    Parameter("x3", (0.0, 3.0), model_property="x3"),
]

PARAMETER_PAIRS = [
    [(PARAM_LIST[0], 1.0), (PARAM_LIST[1], 1.0), (PARAM_LIST[2], 1.0)],
    [(PARAM_LIST[0], 2.0), (PARAM_LIST[1], 2.0), (PARAM_LIST[2], 2.0)],
]


class TestSimulate:
    def test_run_models_in_parallel(self):
        model = IshigamiDynamic()

        res = run_simulations(model, PARAMETER_PAIRS, SIMULATION_OPTIONS, n_cpu=1)

        assert len(res) == 2
        pd.testing.assert_frame_equal(
            res[0],
            pd.DataFrame(
                {"res": [5.882132011203685, 5.882132011203685, 5.882132011203685]},
                index=pd.date_range(
                    "2009-01-01 00:00:00", "2009-01-01 02:00:00", freq="h"
                ),
            ),
        )

        res = run_simulations(model, PARAMETER_PAIRS, SIMULATION_OPTIONS, n_cpu=-1)

        assert len(res) == 2
        pd.testing.assert_frame_equal(
            res[0],
            pd.DataFrame(
                {"res": [5.882132011203685, 5.882132011203685, 5.882132011203685]},
                index=pd.date_range(
                    "2009-01-01 00:00:00", "2009-01-01 02:00:00", freq="h"
                ),
            ),
        )
