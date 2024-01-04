import enum
from typing import Any

from collections.abc import Callable
from corrai.base.model import Model
from corrai.base.simulate import run_list_of_models_in_parallel
from copy import deepcopy
import itertools


class VariantKeys(enum.Enum):
    MODIFIER = "MODIFIER"
    ARGUMENTS = "ARGUMENTS"
    DESCRIPTION = "DESCRIPTION"


def get_modifier_dict(
    variant_dict: dict[str, dict[VariantKeys, Any]], add_existing: bool = False
):
    """
    Generate a dictionary that maps modifier values (name) to associated variant names.

    This function takes a dictionary containing variant information and extracts
    the MODIFIER values along with their corresponding variants, creating a new
    dictionary where each modifier is associated with a list of variant names
    that share that modifier.

    :param variant_dict: A dictionary containing variant information where keys are
                        variant names and values are dictionaries with keys from the
                        VariantKeys enum (e.g., MODIFIER, ARGUMENTS, DESCRIPTION).
    :param add_existing: A boolean flag indicating whether to include existing
                        variant to each modifier.
                        If True, existing modifiers will be included;
                        if False, only non-existing modifiers will be considered.
                        Set to False by default.
    :return: A dictionary that maps modifier values to lists of variant names.
    """
    temp_dict = {}

    if add_existing:
        temp_dict = {
            variant_dict[var][VariantKeys.MODIFIER]: [
                f"EXISTING_{variant_dict[var][VariantKeys.MODIFIER]}"
            ]
            for var in variant_dict.keys()
        }
        for var in variant_dict.keys():
            temp_dict[variant_dict[var][VariantKeys.MODIFIER]].append(var)
    else:
        for var in variant_dict.keys():
            modifier = variant_dict[var][VariantKeys.MODIFIER]
            if modifier not in temp_dict:
                temp_dict[modifier] = []
            temp_dict[modifier].append(var)

    return temp_dict


def get_combined_variants(
    variant_dict: dict[str, dict[VariantKeys, Any]], add_existing: bool = False
):
    """
    Generate a list of combined variants based on the provided variant dictionary.

    This function takes a dictionary containing variant information and generates a list
    of combined variants by taking the Cartesian product of the variant names.
    The resulting list contains tuples, where each tuple represents a
    combination of variant to create a unique combination.

    :param variant_dict: A dictionary containing variant information where keys are
                        variant names and values are dictionaries with keys from the
                        VariantKeys enum (e.g., MODIFIER, ARGUMENTS, DESCRIPTION).
    :param add_existing: A boolean flag indicating whether to include existing
                        variant to each modifier.
                        If True, existing modifiers will be included;
                        if False, only non-existing modifiers will be considered.
                        Set to False by default.
    :return: A list of tuples representing combined variants based on the provided
             variant dictionary.
    """
    modifier_dict = get_modifier_dict(variant_dict, add_existing)
    return list(set(itertools.product(*list(modifier_dict.values()))))


def simulate_variants(
    model: Model,
    variant_dict: dict[str, dict[VariantKeys, Any]],
    modifier_map: dict[str, Callable],
    simulation_options: dict[str, Any],
    n_cpu: int = -1,
    add_existing: bool = False,
):
    """
    Simulate a list of mppodel variants combination in parallel with customizable
    modifiers.

    This function takes a base model, a dictionary of variant information, a modifier
    map that associates modifiers with variant modifiers, simulation options, and an
    optional number of CPUs for parallel execution. It generates a list of model
    variants combination by applying the specified modifiers to the base model and
    then simulates these variants in parallel.
    The results of each simulation are collected in a list.

    :param model: The model. Inherit from corrai.base.model Model.

    :param variant_dict: A dictionary containing variant information where keys are
                        variant names and values are dictionaries with keys from the
                        VariantKeys enum (e.g., MODIFIER, ARGUMENTS, DESCRIPTION).
    :param add_existing: A boolean flag indicating whether to include existing
                    variant to each modifier.
                    If True, existing modifiers will be included;
                    if False, only non-existing modifiers will be considered.
                    Set to False by default.
    :param modifier_map: A dictionary that maps variant modifiers to modifier functions
                        for customizing model variants.

    :param simulation_options: A dictionary containing options for the simulation.

    :param n_cpu: The number of CPU cores to use for parallel execution. Default is -1
        meaning all CPUs but one, 0 is all CPU, 1 is sequential, >1 is the number
        of cpus

    :return: A list of simulation results for each model variant.
    """
    model_list = []
    for simulation in get_combined_variants(variant_dict, add_existing):
        working_model = deepcopy(model)
        for variant in simulation:
            split_var = variant.split("_")
            if not split_var[0] == "EXISTING":
                modifier = modifier_map[variant_dict[variant][VariantKeys.MODIFIER]]
                modifier(
                    model=working_model,
                    description=variant_dict[variant][VariantKeys.DESCRIPTION],
                    **variant_dict[variant][VariantKeys.ARGUMENTS],
                )
        model_list.append(working_model)

    return run_list_of_models_in_parallel(model_list, simulation_options, n_cpu)
