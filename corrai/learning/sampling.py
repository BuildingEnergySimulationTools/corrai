import numpy as np
from pathlib import Path

from corrai.base.parameter import Parameter
from corrai.variant import simulate_variants

from scipy.stats.qmc import LatinHypercube
from fastprogress.fastprogress import force_console_behavior

import random
import itertools

master_bar, progress_bar = force_console_behavior()


class VariantSubSampler:
    """
    A class for subsampling variants from a given set of combinations.
    """

    def __init__(
        self,
        model,
        combinations,
        variant_dict=None,
        modifier_map=None,
        simulation_options=None,
        save_dir=None,
        file_extension=".txt",
    ):
        """
        Initialize the VariantSubSampler.

        Args:
            model: The model to be used for simulations.
            combinations: List of lists, each inner list
            representing a combination of variants.
            variant_dict (optional): A dictionary containing variant information where
                                     keys are variant names and values are dictionaries
                                     with keys from the VariantKeys enum.
            modifier_map (optional): A map of modifiers for simulation.
            simulation_options (optional): Options for simulations. Defaults to None.
            save_dir (optional): Directory to save simulation results. Defaults to None.
            file_extension (optional): File extension for
            saved simulation results. Defaults to ".txt".

        """
        self.model = model
        self.combinations = combinations
        self.variant_dict = variant_dict
        self.modifier_map = modifier_map
        self.simulation_options = simulation_options
        self.save_dir = save_dir
        self.file_extension = file_extension
        self.sample = []
        self.all_variants = set(itertools.chain(*combinations))
        self.sample_results = []
        self.not_simulated_combinations = []
        self.variant_coverage = {variant: False for variant in self.all_variants}

    def add_sample(
        self,
        sample_size,
        simulate=True,
        seed=None,
        ensure_full_coverage=False,
        n_cpu=-1,
        simulation_options=None,
    ):
        """
        Add a sample to the VariantSubSampler.

        Args:
            sample_size: The size of the sample to be added.
            simulate (optional): Whether to perform simulation
            after adding the sample. Defaults to True.
            seed (optional): Seed for random number generation. Defaults to None.
            ensure_full_coverage (optional): Whether to ensure
            full coverage of variants in the sample. Defaults to False.
            n_cpu (optional): Number of CPU cores to use for simulation. Defaults to -1.
            simulation_options (optional): Options for simulations. Defaults to None.
        """

        if seed is not None:
            random.seed(seed)

        shuffled_combinations = self.combinations[:]
        random.shuffle(shuffled_combinations)

        current_sample_count = 0

        if ensure_full_coverage and not all(self.variant_coverage.values()):
            for combination in shuffled_combinations:
                if (
                    any(not self.variant_coverage[variant] for variant in combination)
                    and combination not in self.sample
                ):
                    self.sample.append(combination)
                    self.not_simulated_combinations.append(combination)
                    current_sample_count += 1
                    for variant in combination:
                        self.variant_coverage[variant] = True

                    if all(self.variant_coverage.values()):
                        break

        additional_needed = sample_size - current_sample_count
        if additional_needed > 0:
            for combination in shuffled_combinations:
                if combination not in self.sample:
                    self.sample.append(combination)
                    self.not_simulated_combinations.append(combination)
                    additional_needed -= 1
                    if additional_needed == 0:
                        break

        if additional_needed > 0:
            print(
                "Warning: Not enough unique combinations "
                "to meet the additional requested sample size."
            )

        if simulate:
            effective_simulation_options = (
                simulation_options
                if simulation_options is not None
                else self.simulation_options
            )
            if effective_simulation_options is None:
                raise ValueError(
                    "Simulation options must be provided either "
                    "during initialization or when adding samples."
                )

            self.simulate_combinations(
                n_cpu=n_cpu,
                simulation_options=effective_simulation_options,
            )

    def draw_sample(
        self,
        sample_size,
        simulate=False,
        seed=None,
        ensure_full_coverage=False,
        n_cpu=-1,
        simulation_options=None,
    ):
        """
        Draw a sample from the VariantSubSampler.

        Args:
            sample_size: The size of the sample to be drawn.
            simulate (optional): Whether to perform simulation
            after drawing the sample. Defaults to False.
            seed (optional): Seed for random number generation.
            Defaults to None.
            ensure_full_coverage (optional): Whether to ensure
            full coverage of variants in the sample. Defaults to False.
            n_cpu (optional): Number of CPU cores to use for
            simulation. Defaults to -1.
            simulation_options (optional): Options for simulations.
            Defaults to None.
        """
        self.add_sample(
            sample_size,
            simulate=simulate,
            seed=seed,
            ensure_full_coverage=ensure_full_coverage,
            n_cpu=n_cpu,
            simulation_options=simulation_options,
        )

    def simulate_combinations(
        self,
        n_cpu=-1,
        simulation_options=None,
    ):
        """
        Simulate combinations in the VariantSubSampler.

        Args:
            n_cpu (optional): Number of CPU cores to use for simulation. Defaults to -1.
            simulation_options (optional): Options for simulations. Defaults to None.
        """
        if self.not_simulated_combinations:
            print("yes, not simulated", self.not_simulated_combinations)
            results = simulate_variants(
                self.model,
                self.variant_dict,
                self.modifier_map,
                simulation_options,
                n_cpu,
                custom_combinations=list(self.not_simulated_combinations),
                save_dir=Path(self.save_dir) if self.save_dir else None,
                file_extension=self.file_extension,
            )

            self.sample_results.extend(results)
            self.not_simulated_combinations = []  # Clear the list after simulation

    def clear_sample(self):
        """
        Clears all samples and related simulation data from the sampler. This method is
        useful for resetting the sampler's state, typically used when starting a new set
        of simulations or after a complete set of simulations has been processed and the
        data is no longer needed.

        Parameters:
        - None

        Returns:
        - None: The method resets the internal state related to samples but does not
          return any value.
        """
        self.sample = []
        self.sample_results = []
        self.not_simulated_combinations = []


class ModelSampler:
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
