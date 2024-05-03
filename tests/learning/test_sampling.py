from pathlib import Path
from corrai.base.model import Model
import numpy as np
import pandas as pd
from corrai.learning.sampling import SimulationSampler
from corrai.base.parameter import Parameter

FILES_PATH = Path(__file__).parent / "resources"

parameters = [
    {Parameter.NAME: "Parameter1", Parameter.INTERVAL: (0, 6), Parameter.TYPE: "Real"},
    {
        Parameter.NAME: "Parameter2",
        Parameter.INTERVAL: (1, 4),
        Parameter.TYPE: "Integer",
    },
    {
        Parameter.NAME: "Parameter3",
        Parameter.INTERVAL: (0.02, 0.04, 0.2),
        Parameter.TYPE: "Choice",
    },
    {
        Parameter.NAME: "Parameter4",
        Parameter.INTERVAL: (0, 1),
        Parameter.TYPE: "Binary",
    },
]


class Simul(Model):
    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        Parameter1, Parameter2, Parameter3, Parameter4 = parameter_dict.values()
        df = Parameter1 + Parameter2 - Parameter3 + 0 * Parameter4
        df_out = pd.DataFrame({"df": [df]})
        return df_out


class TestSimulationSampler:
    def test_get_boundary_sample(self):
        sampler = SimulationSampler(parameters, simulator=None)

        boundary_sample = sampler.get_boundary_sample()

        assert isinstance(boundary_sample, np.ndarray)
        assert boundary_sample.shape == (len(parameters), 2)

        for i, param in enumerate(parameters):
            lower_bound = param[Parameter.INTERVAL][0]
            if param[Parameter.TYPE] == "Choice":
                lower_bound = min(param[Parameter.INTERVAL])
            upper_bound = param[Parameter.INTERVAL][-1]
            if param[Parameter.TYPE] == "Choice":
                upper_bound = max(param[Parameter.INTERVAL])
            for j in range(2):  # Loop through lower and upper bounds
                assert lower_bound <= boundary_sample[i, j] <= upper_bound

    def test_add_sample(self):
        Simulator = Simul()
        sampler = SimulationSampler(parameters, simulator=Simulator)
        sample_size = 10
        sampler.add_sample(sample_size)

        assert len(sampler.sample_results) == sample_size + 2

        new_sample_size = 5
        sampler.add_sample(new_sample_size)

        assert len(sampler.sample_results) == sample_size + new_sample_size + 2

        for sample_values in sampler.sample:
            for par, val in zip(parameters, sample_values):
                interval = par[Parameter.INTERVAL]
                if par[Parameter.TYPE] == "Real":
                    assert interval[0] <= val <= interval[1]
                elif par[Parameter.TYPE] == "Integer":
                    assert interval[0] <= val <= interval[1]
                    assert val.is_integer()
                elif par[Parameter.TYPE] == "Choice":
                    assert val in interval
                elif par[Parameter.TYPE] == "Binary":
                    assert val in [0, 1]

    def test_clear_sample(self):
        Simulator = Simul()
        sampler = SimulationSampler(parameters, simulator=Simulator)
        sampler.add_sample(1)
        sampler.clear_sample()

        assert sampler.sample is None
