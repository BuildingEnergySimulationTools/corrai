from multiprocessing import Pool
from corrai.base.model import Model

import pandas as pd


def run_model(args):
    parameters, model, simulation_options = args
    return model.simulate(parameters, simulation_options)


def run_models_in_parallel(
    model, parameter_samples: pd.DataFrame, simulation_options: dict, n_cpu: int
):
    """
    Run a sample of simulation in parallel.

    Parameters:
    - model: The simulation model to run.
    - parameter_samples: A DataFrame containing parameter samples for the model.
        columns is parameters name, index is the number of simulation
    - simulation_options: A dictionary of options for the simulation.
    Keys values depend on the model requirements.
    - n_cpu: The number of CPU cores to use for parallel execution.

    Returns:
    - combined_result: A list of tuples, where each tuple contains:
        - param: A dictionary of parameter values used for the simulation.
        - simulation_options: The options used for the simulation.
        - res: The result of running the model with the given parameters.
    """
    sample_dict = parameter_samples.to_dict(orient="records")
    with Pool(n_cpu) as pool:
        results = pool.map(
            run_model, [(param, model, simulation_options) for param in sample_dict]
        )

    combined_result = [
        (param, simulation_options, res) for param, res in zip(sample_dict, results)
    ]

    return combined_result


def run_list_of_models_in_parallel(
    models: list[Model], simulation_options: dict, n_cpu: int
):
    """
    Run a list of models in parallel.

    :param n_cpu:
    :param models: A list of Model objects to be simulated in parallel.
    :param simulation_options: A dictionary containing simulation options.
    :return: A list of Pandas DataFrames with simulation results for each model.
    """

    # Create a pool of worker processes
    with Pool(n_cpu) as pool:
        results = pool.map(
            run_model, [(None, model, simulation_options) for model in models]
        )

    return results