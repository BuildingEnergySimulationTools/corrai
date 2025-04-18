from copy import deepcopy

import numpy as np
from pathlib import Path

from corrai.base.parameter import Parameter
from corrai.variant import simulate_variants, get_combined_variants

import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from scipy.stats.qmc import LatinHypercube
from fastprogress.fastprogress import force_console_behavior

import random
import itertools
from itertools import product

master_bar, progress_bar = force_console_behavior()


def expand_parameter_dict(parameter_dict, param_mappings):
    """
    Expands a sampled parameter dictionary based on predefined mappings.

    This method takes a dictionary of sampled parameters and expands it by applying
    predefined mappings stored in `self.param_mappings`. The expansion process works
    as follows:

    1. **If the mapping for a parameter is a dictionary**:
       - The method checks if the sampled value exists as a key in the mapping.
       - If found, the corresponding mapped values are added to the expanded dictionary.

    2. **If the mapping for a parameter is an iterable (non-dictionary)**:
       - The method applies the sampled value to each key in the mapping and adds
         these key-value pairs to the expanded dictionary.

    3. **If a parameter has no predefined mapping**:
       - The original parameter and its value are added directly to the expanded dictionary.

    Parameters
    ----------
    parameter_dict : dict
        The original parameter dictionary from the sample, with parameter names as keys
        and sampled values as values.
    param_mappings:  dict
    A dictionary defining how sampled parameters should be expanded.

    Returns
    -------
    dict
        An expanded parameter dictionary containing the original parameters along with
        additional key-value pairs derived from the predefined mappings.

    """
    expanded_dict = {}
    for param_name, value in parameter_dict.items():
        if param_name in param_mappings:
            mapping = param_mappings[param_name]
            if isinstance(mapping, dict):
                if value in mapping:
                    expanded_dict.update(mapping[value])
            else:
                expanded_dict.update({k: value for k in mapping})
        else:
            expanded_dict[param_name] = value
    return expanded_dict


def get_mapped_bounds(uncertain_param_list, param_mappings):
    """
    Return actual bounds (min, max) of all parameters after applying param_mappings.

    Args:
        uncertain_param_list (list): List of parameter dicts with NAME, INTERVAL, TYPE.
        param_mappings (dict): Mapping rules to expand parameters.

    Returns:
        List[Tuple[float, float]]: List of bounds for each expanded parameter.
    """
    reverse_mapping = {}
    for base_param, expanded_list in param_mappings.items():
        for expanded_param in expanded_list:
            reverse_mapping[expanded_param] = base_param

    bounds_dict = {
        param[Parameter.NAME]: tuple(param[Parameter.INTERVAL])
        for param in uncertain_param_list
    }

    base_values = {name: bounds_dict[name][0] for name in bounds_dict}
    expanded_dict = expand_parameter_dict(base_values, param_mappings)

    bounds = []
    for expanded_param in expanded_dict:
        base_param = reverse_mapping.get(
            expanded_param, expanded_param
        )  # fall back to self
        if base_param not in bounds_dict:
            raise ValueError(
                f"No matching bounds for parameter: {expanded_param} (base: {base_param})"
            )
        bounds.append(bounds_dict[base_param])

    return bounds


class VariantSubSampler:
    """
    A class for subsampling variants from a given set of combinations.
    """

    def __init__(
        self,
        model,
        add_existing=False,
        variant_dict=None,
        modifier_map=None,
        custom_combination=None,
        simulation_options=None,
        save_dir=None,
        file_extension=".txt",
    ):
        """
        Initialize the VariantSubSampler.

        Args:
            model: The model to be used for simulations.
            custom_combination: List of lists, each inner list
            representing a custom combination of variants. Otherwise, all variants are automatically
            deduced from variant_dict and modifier_map.
            add_existing: A boolean flag indicating whether to include existing
                variant to each modifier.
                If True, existing modifiers will be included;
                if False, only non-existing modifiers will be considered.
                Set to False by default.
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
        self.add_existing = add_existing
        self.variant_dict = variant_dict
        self.combinations = (
            custom_combination
            if custom_combination
            else get_combined_variants(self.variant_dict)
        )
        self.modifier_map = modifier_map
        self.simulation_options = simulation_options
        self.save_dir = save_dir
        self.file_extension = file_extension
        self.sample = []
        self.simulated_samples = []
        self.all_variants = set(itertools.chain(*self.combinations))
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

            seed (optional): Seed for random number generation. Defaults to None.
        Args:
            sample_size: The size of the sample to be added.
            simulate (optional): Whether to perform simulation
            after adding the sample. Defaults to True.
            ensure_full_coverage (optional): Whether to ensure
            full coverage of variants in the sample. Defaults to False.
            n_cpu (optional): Number of CPU cores to use for simulation. Defaults to -1.
            simulation_options (optional): Options for simulations. Defaults to None.
        """

        effective_simulation_options = (
            simulation_options
            if simulation_options is not None
            else self.simulation_options
        )
        if effective_simulation_options is None:
            raise ValueError(
                "Simulation options must be provided either during "
                "initialization or when adding samples."
            )

        shuffled_combinations = self.combinations[:]

        if seed is not None:
            random.Random(seed).shuffle(shuffled_combinations)

        else:
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
                "to meet the additional requested sampl"
                "e size."
            )

        if simulate:
            self.simulate_combinations(
                n_cpu=n_cpu, simulation_options=effective_simulation_options
            )

    def draw_sample(
        self,
        sample_size,
        seed=None,
        ensure_full_coverage=False,
        n_cpu=-1,
        simulation_options=None,
    ):
        """
        Draw a sample from the VariantSubSampler.

        Args:
            sample_size: The size of the sample to be drawn.
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
            simulate=False,
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
        effective_simulation_options = (
            simulation_options
            if simulation_options is not None
            else self.simulation_options
        )
        if not effective_simulation_options:
            raise ValueError("Simulation options are not available for simulation.")

        if self.not_simulated_combinations:
            results = simulate_variants(
                self.model,
                self.variant_dict,
                self.modifier_map,
                effective_simulation_options,
                n_cpu,
                add_existing=self.add_existing,
                custom_combinations=list(self.not_simulated_combinations),
                save_dir=Path(self.save_dir) if self.save_dir else None,
                file_extension=self.file_extension,
            )

            self.sample_results.extend(results)
            self.simulated_samples.extend(self.not_simulated_combinations)
            self.not_simulated_combinations = []  # Clear the list after simulation

    def simulate_all_variants_and_parameters(
        self,
        parameter,
        param_mapping=None,
        simulation_options=None,
        n_cpu=1,
    ):
        """
        Simulates all combinations of parameters and variants by applying parameter sets
        and generating multiple model variants based on the provided variant dictionary.


        Parameters
        ----------

        parameter : list
            A list of parameters.

        param_mapping : dict, optional
            A dictionary defining how sampled parameters should be expanded into additional key-value pairs
            before being applied to the model. Each key in `param_mapping` corresponds to a parameter name
            in `parameter_dict`. The value can be:
            - A dictionary: Maps discrete parameter values to new parameter sets.
            - An iterable: Directly applies the sampled value to a set of keys.

        simulation_options : dict, optional
            Simulation options to override the default `self.simulation_options` set during instantiation.

        n_cpu : int, optional
            Number of CPU cores to use for parallel simulation. Defaults to 1 (for now, -1 not working with both parameters and variants variations).

        Returns
        -------
        None
            The method updates the `self.sample` and `self.sample_results` attributes with the parameter and variant
            combinations used in each simulation and the corresponding simulation results.

        Notes
        -----
        - The function expects that `self.parameters` contains parameters with the `Choice` type, which defines a discrete
          set of possible values for each parameter.
        - It generates parameter combinations using `itertools.product` to exhaustively cover all possible choices.
        - The parameter and variant combinations are stored in `self.sample` using `np.vstack`.
        - The function calls `simulate_variants` to generate and simulate each variant combination.
        - The `self.param_mappings` attribute is used to dynamically expand the sampled parameters into additional
          key-value pairs, which can modify how the model parameters are applied during simulations.
        """
        effective_simulation_options = (
            simulation_options
            if simulation_options is not None
            else self.simulation_options
        )

        if not effective_simulation_options:
            raise ValueError("Simulation options must be provided for the simulation.")

        if not isinstance(self.sample, np.ndarray) or self.sample.size == 0:
            self.sample = np.empty((0, len(parameter) + len(self.combinations[0])))

        choice_parameters = [
            param for param in parameter if param[Parameter.TYPE] == "Choice"
        ]
        param_names = [param[Parameter.NAME] for param in choice_parameters]
        param_values = [param[Parameter.INTERVAL] for param in choice_parameters]

        all_parameter_dicts = [
            dict(zip(param_names, combination))
            for combination in product(*param_values)
        ]

        for idx, param_dict in enumerate(all_parameter_dicts):
            expanded_param_dict = expand_parameter_dict(param_dict, param_mapping)
            print(f"Simulating combination {idx + 1}/{len(all_parameter_dicts)}...")

            result = simulate_variants(
                n_cpu=n_cpu,
                add_existing=self.add_existing,
                model=deepcopy(self.model),
                variant_dict=self.variant_dict,
                modifier_map=self.modifier_map,
                simulation_options=effective_simulation_options,
                custom_combinations=self.combinations,
                parameter_dict=expanded_param_dict,
            )

            new_sample_value = np.array(
                [
                    list(param_dict.values()) + list(variant_tuple)
                    for variant_tuple in self.combinations
                ]
            )
            self.sample = np.vstack((self.sample, new_sample_value))
            self.sample_results.extend(result)

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
        self.simulated_samples = []
        self.sample_results = []
        self.not_simulated_combinations = []

    def dump_sample(self):
        """
        Clears all non-simulated samples.

        Parameters:
        - None

        Returns:
        - None: The method resets the internal state
        related to non-simulated samples.
        """
        self.sample = [
            comb for comb in self.sample if comb not in self.not_simulated_combinations
        ]
        self.not_simulated_combinations = []


## TODO issue with n_sample: when 1, makes 3 (2 boundaries +1)
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

    def __init__(
        self,
        parameters,
        model,
        sampling_method="LatinHypercube",
        simulation_options=None,
        param_mappings=None,
    ):
        """
        Initialize the SimulationSampler instance.

        :param parameters: A list of dictionaries describing
            the parameters to be sampled.
        :param model: An instance of the simulator
            used for running simulations.
        :param sampling_method: The name of the sampling method
            to be used (default is "LatinHypercube").
        :param simulation_options: A dictionary of options passed
        to the model when running simulations.
        :param param_mappings: A dictionary defining how
        sampled parameters should be expanded.


        """
        self.parameters = parameters
        self.model = model
        self.sampling_method = sampling_method
        self.sample = np.empty(shape=(0, len(parameters)))
        self.simulation_options = simulation_options
        self.sample_results = []
        self.param_mappings = param_mappings or {}
        self.not_simulated_samples = []

        if sampling_method == "LatinHypercube":
            self.sampling_method = LatinHypercube

    def draw_sample(self, sample_size, seed=None):
        """
        Draw a sample of parameter sets without running simulations.

        Args:
            sample_size (int): Number of samples to draw.
            seed (int, optional): Seed for reproducibility.
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

        self.sample = np.vstack((self.sample, new_sample_value))
        self.not_simulated_samples.extend(new_sample_value.tolist())

    def simulate_drawn_samples(self):
        """
        Simulate all drawn samples that haven't been simulated yet.
        Assumes that self.sample contains samples drawn via draw_sample.
        """
        start_idx = len(self.sample_results)
        drawn_samples = self.sample[start_idx:]

        if drawn_samples.size == 0:
            print("No new samples to simulate.")
            return

        prog_bar = progress_bar(range(drawn_samples.shape[0]))

        for i, sample_row in zip(prog_bar, drawn_samples):
            sim_config = {
                par[Parameter.NAME]: val
                for par, val in zip(self.parameters, sample_row)
            }
            expanded_config = expand_parameter_dict(sim_config, self.param_mappings)
            prog_bar.comment = f"Simulating {start_idx + i + 1}/{self.sample.shape[0]}"

            result = self.model.simulate(
                parameter_dict=expanded_config,
                simulation_options=self.simulation_options,
            )
            self.sample_results.append(result)

    def simulate_all_combinations(self):
        choice_parameters = [
            param for param in self.parameters if param[Parameter.TYPE] == "Choice"
        ]
        intervals = [param[Parameter.INTERVAL] for param in choice_parameters]
        all_combinations = list(product(*intervals))
        param_names = [param[Parameter.NAME] for param in choice_parameters]

        prog_bar = progress_bar(range(len(all_combinations)))

        applied_parameters = []
        expanded_parameters = []

        for idx, combination in zip(prog_bar, all_combinations):
            param_dict = dict(zip(param_names, combination))
            expanded_param_dict = expand_parameter_dict(param_dict, self.param_mappings)
            prog_bar.comment = f"Simulation {idx + 1}/{len(all_combinations)}"
            result = self.model.simulate(
                parameter_dict=expanded_param_dict,
                simulation_options=self.simulation_options,
            )
            self.sample_results.append(result)
            applied_parameters.append(combination)
            expanded_parameters.append(expanded_param_dict)

        applied_parameters_array = np.array(applied_parameters)
        self.sample = np.vstack((self.sample, applied_parameters_array))

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
            expanded_config = expand_parameter_dict(sim_config, self.param_mappings)
            prog_bar.comment = "Simulations"

            results = self.model.simulate(
                parameter_dict=expanded_config,
                simulation_options=self.simulation_options,
            )
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


def plot_pcp(
    sample_results,
    param_sample,
    parameters,
    indicators,
    aggregation_method=np.mean,
    html_file_path=None,
):
    """
    Plots a parallel coordinate plot for sensitivity analysis results.
      Notes
    -----
    - The function handles categorical parameters by converting them to numerical
      codes using `pandas.Categorical`. The tick values are shifted slightly to
      improve readability.
    - The first indicator in the list is used to color the lines in the plot.
    - The function requires the Plotly library (`plotly.graph_objects`) to create
      the interactive plot.
    """
    data_dict = [
        {param[Parameter.NAME]: value for param, value in zip(parameters, sample)}
        for sample in param_sample
    ]

    if isinstance(indicators, str):
        indicators = [indicators]

    for i, df in enumerate(sample_results):
        for indicator in indicators:
            if indicator in df.columns:
                data_dict[i][indicator] = aggregation_method(df[indicator])

    df_plot = pd.DataFrame(data_dict)
    dim_list = []

    for col in df_plot.select_dtypes(include="object").columns:
        cat = pd.Categorical(df_plot[col])
        df_plot[col] = cat.codes
        dim_list.append(
            dict(
                range=[0, len(cat.categories) - 1],
                label=col,
                tickvals=list(set(cat.codes)),
                ticktext=list(cat.categories),
                values=df_plot[col].tolist(),
            )
        )

    for col in indicators:
        dim_list.append(dict(label=col, values=df_plot[col].tolist()))

    color_indicator = indicators[0] if indicators else None

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df_plot[color_indicator],
                colorscale="Plasma",
                showscale=True,
            ),
            dimensions=dim_list,
        )
    )

    if html_file_path:
        pio.write_html(fig, html_file_path)

    return fig
