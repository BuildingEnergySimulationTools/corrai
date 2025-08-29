from multiprocessing import cpu_count

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
    """
    Run multiple simulations of a model in parallel.

    This function dispatches simulations for a given model across multiple
    CPU cores using `joblib.Parallel`. Each simulation is defined by a list
    of (Parameter, value) pairs that specify which model properties to update
    before running the simulation.

    A progress bar is displayed using `fastprogress` to track the simulations.

    Parameters
    ----------
    model : Model
        An instance of a subclass of :class:`corrai.base.model.Model`.
        Must implement the method `simulate_parameter`.
    list_parameter_value_pairs : list of list of (Parameter, value)
        A list where each element corresponds to one simulation run.
        Each simulation run is represented as a list of tuples:
        `(Parameter, value)` indicating the parameter to update
        and the value to assign.
    simulation_options : dict
        Dictionary of options to pass to the model's `simulate` method.
        May include time horizon, timestep, solver options, etc.
    n_cpu : int, default=-1
        Number of CPU cores to use for parallelization.
        - If > 0, use exactly `n_cpu` cores.
        - If 0 or negative, use `(cpu_count() + n_cpu)` cores.
          For example, `n_cpu=-1` uses all available cores,
          `n_cpu=-2` uses (all but one), etc.
    simulation_kwargs : dict, optional
        Additional keyword arguments passed directly to the model's
        `simulate_parameter` method.

    Returns
    -------
    results : list of pandas.DataFrame
        Each element contains the simulation results for the corresponding
        parameter set. The structure of each DataFrame depends on the
        implementation of the `simulate` method of the model.

    Notes
    -----
    - Simulations are executed in parallel using `joblib.Parallel`.
    - A progress bar is displayed with `fastprogress`.
    - The model must implement :meth:`Model.simulate_parameter`.

    Examples
    --------
    >>> from corrai.base.model import Model
    >>> from corrai.base.parameter import Parameter
    >>> import pandas as pd

    >>> class SimpleModel(Model):
    ...     def simulate(self, property_dict=None, simulation_options=None, **kwargs):
    ...         # Simple example: return a constant series
    ...         return pd.DataFrame(
    ...             {"output": [property_dict.get("x", 0)] * 2},
    ...             index=pd.date_range("2020-01-01", periods=5, freq="h")
    ...         )

    >>> param_x = Parameter(name="x", interval=(0, 1))
    >>> model = SimpleModel()
    >>> param_sets = [[(param_x, 0.1)], [(param_x, 0.5)], [(param_x, 0.9)]]

    >>> results = run_simulations(model, param_sets, simulation_options={})
    >>> len(results)
    3
    >>> results[0].head()
                       output
    2020-01-01 00:00:00     0
    2020-01-01 01:00:00     0
    """
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
