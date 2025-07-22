from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from copy import deepcopy
import random
import itertools

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from scipy.stats.qmc import LatinHypercube

from corrai.base.parameter import Parameter
from corrai.base.model import Model
from corrai.base.simulate import run_simulations
from corrai.variant import simulate_variants, get_combined_variants


@dataclass
class Sample:
    parameters: list[Parameter]
    values: np.ndarray = field(init=False)
    results: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))

    def __post_init__(self):
        self.values = np.empty((0, len(self.parameters)))

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice, list, np.ndarray)):
            return {"values": self.values[idx], "results": self.results[idx]}
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __setitem__(self, idx, item: dict):
        if "values" in item:
            self.values[idx] = item["values"]
        if "results" in item:
            if isinstance(idx, int):
                self.results.iloc[idx] = item["results"]
            else:
                self.results.iloc[idx] = pd.Series(
                    item["results"], index=self.results.index[idx]
                )

        self._validate()

    def _validate(self):
        assert len(self.results) == len(
            self.values
        ), f"Mismatch: {len(self.values)} values vs {len(self.results)} results"

    def get_pending_index(self) -> np.ndarray:
        return self.results.apply(lambda df: df.empty).values

    def get_parameters_intervals(self):
        if all(param.ptype == "Real" for param in self.parameters):
            return np.array([param.interval for param in self.parameters])
        elif any(param.ptype == "Integer" for param in self.parameters):
            raise NotImplementedError(
                "get_param_interval is not yet implemented for integer parameters"
            )
        else:
            raise ValueError("All parameter must have an ptype = 'Real'")

    def get_list_parameter_value_pairs(
        self, idx: int | list[int] | np.ndarray | slice = None
    ):
        idx = slice(None) if idx is None else idx

        if isinstance(idx, int) or (
            isinstance(idx, list) and all(isinstance(x, bool) for x in idx)
        ):
            idx = np.array(idx)

        selected_values = self.values[idx]

        if selected_values.ndim == 1:
            selected_values = selected_values[np.newaxis, :]

        return [
            [(par, val) for par, val in zip(self.parameters, row)]
            for row in selected_values
        ]

    def add_samples(self, values: np.ndarray, results: list[pd.DataFrame] = None):
        n_samples, n_params = values.shape
        assert n_params == len(self.parameters), "Mismatch in number of parameters"

        self.values = np.vstack([self.values, values])

        if results is None:
            new_results = pd.Series([pd.DataFrame()] * n_samples, dtype=object)
        else:
            assert len(results) == n_samples, "Mismatch between values and results"
            new_results = pd.Series(results, dtype=object)

        self.results = pd.concat([self.results, new_results], ignore_index=True)


class Sampler(ABC):
    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,
    ):
        self.simulation_options = (
            {} if simulation_options is None else simulation_options
        )
        self.parameters = parameters
        self.model = model
        self.sample = Sample(self.parameters)

    @property
    def values(self):
        return self.sample.values

    @property
    def results(self):
        return self.sample.results

    @abstractmethod
    def add_sample(self, *args, **kwargs) -> np.ndarray:
        pass

    def _post_draw_sample(
        self,
        new_sample,
        simulate=True,
        n_cpu: int = 1,
        sample_is_dimless: bool = False,
        simulation_kwargs: dict = None,
    ):
        if sample_is_dimless:
            intervals = self.sample.get_parameters_intervals()
            lower_bounds = intervals[:, 0]
            upper_bounds = intervals[:, 1]
            new_values = lower_bounds + new_sample * (upper_bounds - lower_bounds)
        else:
            new_values = new_sample

        self.sample.add_samples(new_values)

        if simulate:
            sample_starts = len(self.sample) - new_values.shape[0]
            sample_ends = len(self.sample)
            self.simulate_at(
                slice(sample_starts, sample_ends), n_cpu, simulation_kwargs
            )

    def simulate_at(
        self,
        idx: int | list[int] | np.ndarray | slice = None,
        n_cpu: int = 1,
        simulation_kwargs=None,
    ):
        list_param_value_pairs = self.sample.get_list_parameter_value_pairs(idx)
        res = run_simulations(
            self.model,
            list_param_value_pairs,
            self.simulation_options,
            n_cpu,
            simulation_kwargs,
        )
        if isinstance(idx, int):
            self.sample[idx] = {"results": res[0]}
        else:
            self.sample[idx] = {"results": [r for r in res]}

    def simulate_pending(self, n_cpu: int = 1, simulation_kwargs: dict = None):
        unsimulated_idx = self.sample.get_pending_index()
        self.simulate_at(unsimulated_idx, n_cpu, simulation_kwargs)

class RealSampler(Sampler, ABC):
    def __init__(self, parameters: list[Parameter],
        model: Model,
        simulation_options: dict = None,):
        super().__init__(parameters, model, simulation_options)

        if not all(param.ptype == "Real" for param in parameters):
            raise ValueError(
                f"All parameters must have a ptype 'Real'"
                f"Found {
                [(par.name, par.ptype) for par in parameters if par.ptype != "Real"]}"
            )


class LHCSampler(RealSampler):
    def __init__(
        self, parameters: list[Parameter], model: Model, simulation_options: dict = None
    ):
        super().__init__(parameters, model, simulation_options)

    def add_sample(self, n: int, rng: int = None, simulate=True, **lhs_kwargs):
        lhc = LatinHypercube(d=len(self.parameters), rng=rng, **lhs_kwargs)
        new_dimless_sample = lhc.random(n=n)
        self._post_draw_sample(new_dimless_sample, simulate, sample_is_dimless=True)


class SobolSampler(Sampler):
    def __init__(self, parameters: list[Parameter], model: Model):
        super().__init__(parameters, model)

    def add_sample(self, ni, custom_par, simulate=True):
        new_sample = draw_sobol(ni, custom_par)
        self._add_sample_post(new_sample, simulate)


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
