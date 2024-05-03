from pathlib import Path
from corrai.base.model import Model
import numpy as np
import pandas as pd
from corrai.learning.sampling import SimulationSampler
from corrai.base.parameter import Parameter

FILES_PATH = Path(__file__).parent / "resources"

parameters = [
    {Parameter.NAME: "Parameter1", Parameter.INTERVAL: (0, 6)},
    {Parameter.NAME: "Parameter2", Parameter.INTERVAL: (0.2, 0.95)},
    {Parameter.NAME: "Parameter3", Parameter.INTERVAL: (0.02, 0.2)},
]


class Simul(Model):
    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        Parameter1, Parameter2, Parameter3 = parameter_dict.values()
        df = Parameter1 + Parameter2 - Parameter3
        df_out = pd.DataFrame({"df": [df]})
        return df_out


class TestSpling:
    def test_get_boundary_sample(self):
        sampler = SimulationSampler(parameters, simulator=None)

        boundary_sample = sampler.get_boundary_sample()

        assert isinstance(boundary_sample, np.ndarray)
        assert boundary_sample.shape == (len(parameters), 2)

        for i, param in enumerate(parameters):
            for j in range(2):  # Loop through lower and upper bounds
                assert (
                    param[Parameter.INTERVAL][0]
                    <= boundary_sample[i, j]
                    <= param[Parameter.INTERVAL][1]
                )

    def test_add_sample(self):
        Simulator = Simul()
        sampler = SimulationSampler(parameters, simulator=Simulator)
        sample_size = 10
        sampler.add_sample(sample_size)

        assert len(sampler.sample_results) == sample_size + len(parameters)

        new_sample_size = 5
        sampler.add_sample(new_sample_size)

        assert len(sampler.sample_results) == sample_size + new_sample_size + len(
            parameters
        )
