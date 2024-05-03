import numpy as np
from corrai.base.parameter import Parameter

from scipy.stats.qmc import LatinHypercube
from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = force_console_behavior()


class SimulationSampler:
    """
    A class for sampling parameter sets and running simulations using a simulator.

    Parameters:
    - parameters (list): A list of dictionaries describing the parameters to be sampled.
        Each dictionary should contain keys 'NAME', 'INTERVAL', and 'TYPE' to define
        the parameter name, its range (interval), and its data type
    (Integer, Real, etc.).
    - simulator: An instance of the simulator used for running simulations.
    - sampling_method (str): The name of the sampling method to be used.

    Attributes:
    - parameters (list): List of parameter dictionaries.
    - simulator: Instance of the simulator.
    - sampling_method (str): Name of the sampling method.
    - sample (numpy.ndarray): An array containing the sampled
        parameter sets.
    - sample_results (list): A list containing the results of
        simulations for each sampled parameter set.

    Methods:
    - get_boundary_sample(): Generates a sample covering the parameter space boundaries.
    - add_sample(sample_size, seed=None): Adds a new sample of parameter sets.
    """

    def __init__(self, parameters, simulator, sampling_method="LatinHypercube"):
        """
        Initialize the SimulationSampler instance.

        :param parameters: A list of dictionaries describing
            the parameters to be sampled.
        :param simulator: An instance of the simulator
            used for running simulations.
        :param sampling_method: The name of the sampling method
            to be used (default is "LatinHypercube").
        """
        self.parameters = parameters
        self.simulator = simulator
        self.sampling_method = sampling_method
        self.sample = np.empty(shape=(0, len(parameters)))
        self.sample_results = []
        if sampling_method == "LatinHypercube":
            self.sampling_method = LatinHypercube

    def get_boundary_sample(self):
        """
        Generate a sample covering the parameter space boundaries.

        :return: A numpy array containing the boundary sample.
        """
        boundary_sample = []
        for par in self.parameters:
            interval = par[Parameter.INTERVAL]
            param_type = par.get(Parameter.TYPE, "Real")  # Default type is Real

            if param_type == "Real" or param_type == "Integer":
                boundary_sample.append([interval[0], interval[1]])

            elif param_type == "Choice":
                boundary_sample.append([min(interval), max(interval)])

            elif param_type == "Binary":
                boundary_sample.append([0, 1])

        return np.array(boundary_sample)

    def add_sample(self, sample_size, seed=None):
        """
        Add a new sample of parameter sets.

        :param sample_size: The size of the new sample.
        :param seed: The seed for the random number generator (default is None).
        """
        sampler = LatinHypercube(d=len(self.parameters), seed=seed)
        new_sample = sampler.random(n=sample_size)
        new_sample_value = np.empty(shape=(0, len(self.parameters)))
        for s in new_sample:
            new_sample_value = np.vstack(
                (
                    new_sample_value,
                    [
                        self._sample_parameter(par, val)
                        for par, val in zip(self.parameters, s)
                    ],
                )
            )
        if self.sample.size == 0:
            bound_sample = self.get_boundary_sample()
            print(bound_sample.T, new_sample_value)
            new_sample_value = np.vstack((new_sample_value, bound_sample.T))

        prog_bar = progress_bar(range(new_sample_value.shape[0]))
        for _, simul in zip(prog_bar, new_sample_value):
            sim_config = {
                par[Parameter.NAME]: val for par, val in zip(self.parameters, simul)
            }
            prog_bar.comment = "Simulations"

            results = self.simulator.simulate(parameter_dict=sim_config)
            self.sample_results.append(results)

        self.sample = np.vstack((self.sample, new_sample_value))

    def _sample_parameter(self, parameter, value):
        """
        Sample a parameter based on its type.

        :param parameter: The parameter dictionary.
        :param value: The sampled value.
        :return: The adjusted sampled value based on the parameter type.
        """
        interval = parameter[Parameter.INTERVAL]
        if parameter[Parameter.TYPE] == "Integer":
            sampled_val = np.round(interval[0] + value * (interval[1] - interval[0]))
            sampled_val = max(interval[0], min(sampled_val, interval[1]))
            return int(sampled_val)
        elif parameter[Parameter.TYPE] == "Choice":
            return np.random.choice(interval)
        elif parameter[Parameter.TYPE] == "Binary":
            return np.random.choice([0, 1])
        else:
            return interval[0] + value * (interval[1] - interval[0])

    def clear_sample(self):
        """
        Clear the current sample set.
        """
        self.sample = None
