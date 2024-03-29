import pandas as pd

from corrai.base.simulate import run_simulations
from tests.resources.pymodels import Ishigami

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 02:00:00",
    "timestep": "h",
}


class TestSimulate:
    def test_run_models_in_parallel(self):
        model = Ishigami()
        parameters_sample = pd.DataFrame(
            {
                "x1": [1.0, 2.0],
                "x2": [1.0, 2.0],
                "x3": [1.0, 2.0],
            }
        )

        res = run_simulations(model, parameters_sample, SIMULATION_OPTIONS, n_cpu=1)

        assert len(res) == 2
        assert res[0][0] == {"x1": 1.0, "x2": 1.0, "x3": 1.0}
        assert res[1][0] == {"x1": 2.0, "x2": 2.0, "x3": 2.0}
        assert res[0][1] == {
            "end": "2009-01-01 02:00:00",
            "start": "2009-01-01 00:00:00",
            "timestep": "h",
        }
        pd.testing.assert_frame_equal(
            res[0][2],
            pd.DataFrame(
                {"res": [5.882132011203685, 5.882132011203685, 5.882132011203685]},
                index=pd.date_range(
                    "2009-01-01 00:00:00", "2009-01-01 02:00:00", freq="h"
                ),
            ),
        )

        res = run_simulations(model, parameters_sample, SIMULATION_OPTIONS, n_cpu=-1)

        assert len(res) == 2
        assert res[0][0] == {"x1": 1.0, "x2": 1.0, "x3": 1.0}
        assert res[1][0] == {"x1": 2.0, "x2": 2.0, "x3": 2.0}
        assert res[0][1] == {
            "end": "2009-01-01 02:00:00",
            "start": "2009-01-01 00:00:00",
            "timestep": "h",
        }
        pd.testing.assert_frame_equal(
            res[0][2],
            pd.DataFrame(
                {"res": [5.882132011203685, 5.882132011203685, 5.882132011203685]},
                index=pd.date_range(
                    "2009-01-01 00:00:00", "2009-01-01 02:00:00", freq="h"
                ),
            ),
        )
