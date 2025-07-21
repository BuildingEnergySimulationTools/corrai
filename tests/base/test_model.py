from corrai.base.model import Model
from corrai.base.parameter import Parameter
from ..resources.pymodels import Pymodel

PARAM_LIST = [
    Parameter("par1", (0, 2), relabs="Absolute", model_property=("prop_1", "prop_2")),
    Parameter("par2", (0.8, 1.2), relabs="Relative", model_property="prop_3"),
]

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 05:00:00",
    "timestep": "h",
}


class TestModel:
    def test_pymodel(self):
        mod = Pymodel()

        res = mod.simulate(simulation_options=SIMULATION_OPTIONS)

        assert res.values.tolist() == [[5], [5], [5], [5], [5], [5]]
        assert mod.get_property_from_param(
            [(PARAM_LIST[0], 1), (PARAM_LIST[1], 0.5)]
        ) == {
            "prop_1": 1,
            "prop_2": 1,
            "prop_3": 1.5,
        }

        res = mod.simulate_parameter(
            [(PARAM_LIST[0], 1), (PARAM_LIST[1], 0.5)],
            SIMULATION_OPTIONS,
        )

        assert res.values.tolist() == [[2.5], [2.5], [2.5], [2.5], [2.5], [2.5]]
