from multiprocessing import Pool
from multiprocessing import cpu_count

import pandas as pd
from fastprogress.fastprogress import progress_bar

from corrai.base.model import Model


def run_model(args):
    parameters, model, simulation_options = args
    return model.simulate(parameters, simulation_options)


def run_simulations(
    model, parameter_samples: pd.DataFrame, simulation_options: dict, n_cpu: int = -1
):
    """
    Run a sample of simulation in parallel. The function will automatically divide
    the sample into batch depending on the number of used cpus.

    Parameters:
    - model: The simulation model to run.
    - parameter_samples: A DataFrame containing parameter samples for the model.
        columns are parameters name, index is the number of simulation
    - simulation_options: A dictionary of options for the simulation.
    Keys values depend on the model requirements.
    - n_cpu: The number of CPU cores to use for parallel execution. Default is -1
        meaning all CPUs but one, 0 is all CPU, 1 is sequential, >1 is the number
        of cpus

    Returns:
    - combined_result: A list of tuples, where each tuple contains:
        - param: A dictionary of parameter values used for the simulation.
        - simulation_options: The options used for the simulation.
        - res: The result of running the model with the given parameters.
    """
    if n_cpu <= 0:
        n_cpu = max(1, cpu_count() + n_cpu)

    sample_dict = parameter_samples.to_dict(orient="records")
    grouped_sample = [
        sample_dict[i : i + n_cpu] for i in range(0, len(sample_dict), n_cpu)
    ]
    prog_bar = progress_bar(range(len(grouped_sample)))
    collect = []
    for _mb, group in zip(prog_bar, grouped_sample):
        if n_cpu == 1:
            results = [model.simulate(group[0], simulation_options)]
        else:
            with Pool(n_cpu) as pool:
                results = pool.map(
                    run_model, [(param, model, simulation_options) for param in group]
                )

        combined_result = [
            (param, simulation_options, res) for param, res in zip(sample_dict, results)
        ]
        collect.append(combined_result)

    return [item for sublist in collect for item in sublist]


def run_list_of_models_in_parallel(
    models_list: list[Model], simulation_options: dict, n_cpu: int = -1
):
    """
    Run a list of models in parallel.

    :param n_cpu: The number of CPU cores to use for parallel execution. Default is -1
        meaning all CPUs but one, 0 is all CPU, 1 is sequential, >1 is the number
        of cpus
    :param models_list: A list of Model objects to be simulated in parallel.
    :param simulation_options: A dictionary containing simulation options.
    :return: A list of Pandas DataFrames with simulation results for each model.
    """
    if n_cpu <= 0:
        n_cpu = max(1, cpu_count() + n_cpu)

    grouped_sample = [
        models_list[i : i + n_cpu] for i in range(0, len(models_list), n_cpu)
    ]
    prog_bar = progress_bar(range(len(grouped_sample)))
    collect = []
    for _mb, group in zip(prog_bar, grouped_sample):
        if n_cpu == 1:
            results = [
                group[0].simulate(
                    parameter_dict=None, simulation_options=simulation_options
                )
            ]
        else:
            with Pool(n_cpu) as pool:
                results = pool.map(
                    run_model, [(None, model, simulation_options) for model in group]
                )
        collect.append(results)

    return [item for sublist in collect for item in sublist]
