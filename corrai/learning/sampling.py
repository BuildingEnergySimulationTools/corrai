import numpy as np
from corrai.base.parameter import Parameter

from scipy.stats.qmc import LatinHypercube
from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = force_console_behavior()


class SimulationSampler:
    def __init__(self, parameters, simulator, sampling_method="LatinHypercube"):
        self.parameters = parameters
        self.simulator = simulator
        self.sampling_method = sampling_method
        self.sample = np.empty(shape=(0, len(parameters)))
        self.sample_results = []
        if sampling_method == "LatinHypercube":
            self.sampling_method = LatinHypercube

    def get_boundary_sample(self):
        return np.array(
            [[par[Parameter.INTERVAL][i] for i in range(2)] for par in self.parameters]
        )

    def add_sample(self, sample_size, seed=None):
        sampler = LatinHypercube(d=len(self.parameters), seed=seed)
        new_sample = sampler.random(n=sample_size)
        new_sample_value = np.empty(shape=(0, len(self.parameters)))
        for s in new_sample:
            new_sample_value = np.vstack(
                (
                    new_sample_value,
                    [
                        par[Parameter.INTERVAL][0]
                        + val
                        * (par[Parameter.INTERVAL][1] - par[Parameter.INTERVAL][0])
                        for par, val in zip(self.parameters, s)
                    ],
                )
            )
        if self.sample.size == 0:
            bound_sample = self.get_boundary_sample()
            bound_sample = np.hstack(
                (
                    bound_sample,
                    np.tile(bound_sample[:, [-1]], (1, new_sample_value.shape[1] - 2)),
                )
            )
            new_sample_value = np.vstack((new_sample_value, bound_sample))

        prog_bar = progress_bar(range(new_sample_value.shape[0]))
        for _, simul in zip(prog_bar, new_sample_value):
            sim_config = {
                par[Parameter.NAME]: val for par, val in zip(self.parameters, simul)
            }
            prog_bar.comment = "Simulations"

            results = self.simulator.simulate(parameter_dict=sim_config)
            self.sample_results.append(results)

        self.sample = np.vstack((self.sample, new_sample_value))
