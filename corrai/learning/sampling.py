import numpy as np
from corrai.base.parameter import Parameter

from scipy.stats.qmc import LatinHypercube
from fastprogress.fastprogress import force_console_behavior

import random
import itertools

master_bar, progress_bar = force_console_behavior()


class VariantSubSampler:
    """
    A class to sample subsets of variant combinations from a
    given dictionary of variants, ensuring that each
    unique variant appears at least once in the final sample
    and avoiding duplicate combinations.

    Attributes:
        # modifier_dict (dict): Dictionary mapping modifier values
        to lists of variant names.
        combinations (list): List of all possible combinations
        from the product of modifier values.
        sample (list): List containing the selected unique samples.
    """

    def __init__(self, combinations):
        """
        Initializes the VariantSubSampler with a list of combinations.

        :param combinations: Pre-generated list of
        all possible combinations of variants.
        """
        self.combinations = combinations
        random.shuffle(self.combinations)  # Shuffle to ensure random selection
        self.sample = []
        self.all_variants = set(itertools.chain(*combinations))
        self.variant_coverage = {variant: False for variant in self.all_variants}

    def add_sample(self, sample_size):
        """
        Adds the exact number of new unique combinations
        requested to the existing sample list,
        ensuring each variant appears at least once initially.

        :param sample_size: Number of new samples to add.
        """
        current_sample_count = 0

        if not all(self.variant_coverage.values()):
            for combination in self.combinations:
                if any(not self.variant_coverage[variant] for variant in combination):
                    self.sample.append(combination)
                    current_sample_count += 1
                    for variant in combination:
                        self.variant_coverage[variant] = True

                    # Break if all variants are covered
                    if all(self.variant_coverage.values()):
                        break

        # Additional samples if more are requested and initial are done
        additional_needed = sample_size - current_sample_count
        if additional_needed > 0:
            for combination in self.combinations:
                if combination not in self.sample:
                    self.sample.append(combination)
                    additional_needed -= 1
                    if additional_needed == 0:
                        break

        if additional_needed > 0:
            print(
                "Warning: Not enough unique combinations "
                "to meet the additional requested sample size."
            )

    def clear_sample(self):
        """
        Clear the current sample set.
        """
        self.sample = []


class SimulationSampler:
    """
    A class for sampling parameter sets and running simulations using a simulator.

    Parameters:
    - parameters (list): A list of dictionaries describing the parameters to be sampled.
        Each dictionary should contain keys 'NAME', 'INTERVAL', and 'TYPE' to define
        the parameter name, its range (interval), and its data type
    (Integer, Real, etc.).
    - model: An instance of the model used for running simulations.
    - sampling_method (str): The name of the sampling method to be used.

    Attributes:
    - parameters (list): List of parameter dictionaries.
    - model: Instance of the model.
    - sampling_method (str): Name of the sampling method.
    - sample (numpy.ndarray): An array containing the sampled
        parameter sets.
    - sample_results (list): A list containing the results of
        simulations for each sampled parameter set.

    Methods:
    - get_boundary_sample(): Generates a sample covering the parameter space boundaries.
    - add_sample(sample_size, seed=None): Adds a new sample of parameter sets.
    """

    def __init__(self, parameters, model, sampling_method="LatinHypercube"):
        """
        Initialize the SimulationSampler instance.

        :param parameters: A list of dictionaries describing
            the parameters to be sampled.
        :param model: An instance of the simulator
            used for running simulations.
        :param sampling_method: The name of the sampling method
            to be used (default is "LatinHypercube").
        """
        self.parameters = parameters
        self.model = model
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
        if seed is not None:
            np.random.seed(seed)

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
            new_sample_value = np.vstack((new_sample_value, bound_sample.T))

        prog_bar = progress_bar(range(new_sample_value.shape[0]))
        for _, simul in zip(prog_bar, new_sample_value):
            sim_config = {
                par[Parameter.NAME]: val for par, val in zip(self.parameters, simul)
            }
            prog_bar.comment = "Simulations"

            results = self.model.simulate(parameter_dict=sim_config)
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
            return int(interval[0] + value * (interval[1] - interval[0]))
        elif parameter[Parameter.TYPE] == "Choice":
            return np.random.choice(interval)
        elif parameter[Parameter.TYPE] == "Binary":
            return np.random.randint(0, 2)
        else:
            return interval[0] + value * (interval[1] - interval[0])

    def clear_sample(self):
        """
        Clear the current sample set.
        And current results.
        """
        self.sample = np.empty(shape=(0, len(self.parameters)))
        self.sample_results = []
