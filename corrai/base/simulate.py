from multiprocessing import Pool
from functools import partial

import pandas as pd


def run_model(parameters, model, simulation_options):
    return model.simulate(parameters, simulation_options)


def run_models_in_parallel(
    model, parameter_samples: pd.DataFrame, simulation_options:dict, n_cpu:int
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
        run_func = partial(
            run_model, model=model, simulation_options=simulation_options
        )
        results = pool.map(run_func, sample_dict)

    combined_result = [
        (param, simulation_options, res) for param, res in zip(sample_dict, results)
    ]

    return combined_result
