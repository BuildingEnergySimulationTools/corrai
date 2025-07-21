from multiprocessing import Pool, cpu_count

from joblib import Parallel, delayed
from fastprogress.fastprogress import progress_bar

from corrai.base.model import Model
from corrai.base.parameter import Parameter


def run_simulations(
    model: Model,
    list_parameter_value_pairs: list[list[tuple[Parameter, str | int | float]]],
    simulation_options: dict,
    n_cpu: int = -1,
    simulation_kwargs: dict = None,
):
    simulation_kwargs = simulation_kwargs or {}

    if n_cpu <= 0:
        n_cpu = max(1, cpu_count() + n_cpu)

    bar = progress_bar(list_parameter_value_pairs)

    results = Parallel(n_jobs=n_cpu)(
        delayed(
            lambda param: model.simulate_parameter(
                param, simulation_options, **simulation_kwargs
            )
        )(param)
        for param in bar
    )

    return results


def run_list_of_models_in_parallel(
    models_list: list[Model],
    simulation_options: dict,
    n_cpu: int = -1,
    parameter_dicts: list[dict[str, any]] = None,
    simulate_kwargs: dict = None,
):
    """
    Run a list of models in parallel.

    :param n_cpu: The number of CPU cores to use for parallel execution. Default is -1
        meaning all CPUs but one, 0 is all CPU, 1 is sequential, >1 is the number
        of cpus
    :param models_list: A list of Model objects to be simulated in parallel.
    :param simulation_options: A dictionary containing simulation options.
    :param parameter_dicts: A list of dictionaries containing parameter sets for each model.
                            If not provided, models will be simulated with `None` as the parameter dict.
    :return: A list of Pandas DataFrames with simulation results for each model.
    """
    simulate_kwargs = {} if simulate_kwargs is None else simulate_kwargs
    if n_cpu <= 0:
        n_cpu = max(1, cpu_count() + n_cpu)

    grouped_sample = [
        models_list[i : i + n_cpu] for i in range(0, len(models_list), n_cpu)
    ]
    grouped_params = (
        [parameter_dicts[i : i + n_cpu] for i in range(0, len(parameter_dicts), n_cpu)]
        if parameter_dicts is not None
        else [[None] * len(group) for group in grouped_sample]
    )

    prog_bar = progress_bar(range(len(grouped_sample)))
    collect = []

    for _mb, (group, params_group) in zip(
        prog_bar, zip(grouped_sample, grouped_params)
    ):
        if n_cpu == 1:
            results = [
                group[i].simulate(
                    property_dict=params_group[i],
                    simulation_options=simulation_options,
                    **simulate_kwargs,
                )
                for i in range(len(group))
            ]
        else:
            with Pool(n_cpu) as pool:
                results = pool.map(
                    run_model,
                    [
                        (param_dict, model, simulation_options, simulate_kwargs)
                        for param_dict, model in zip(params_group, group)
                    ],
                )
        collect.append(results)

    return [item for sublist in collect for item in sublist]
