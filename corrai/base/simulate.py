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
