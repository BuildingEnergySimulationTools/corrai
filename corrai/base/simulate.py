from multiprocessing import Pool
from multiprocessing import cpu_count

import pandas as pd
from fastprogress.fastprogress import progress_bar

from corrai.base.model import Model


def run_model(args):
    """
    Run a single model simulation with given arguments.

    Parameters
    ----------
    args : tuple
        Tuple of the form ``(parameters, model, simulation_options, simulate_kwargs)``, where:
        - parameters : dict
            Dictionary of model parameters.
        - model : Model
            The model object implementing `.simulate`.
        - simulation_options : dict
            Simulation options to be passed to the model.
        - simulate_kwargs : dict
            Extra keyword arguments for `.simulate`.

    Returns
    -------
    pandas.DataFrame
        Simulation result from the model.

    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.resources.pymodels import Ishigami
    >>> args = (
    ...     {"x1": 0.5, "x2": 1.0, "x3": 2.0},
    ...     Ishigami(),
    ...     {"start": "2020-01-01", "end": "2020-01-03", "timestep": "D"},
    ...     {}
    ... )
    >>> run_model(args)
                res
    2020-01-01  6.20302
    2020-01-02  6.20302
    2020-01-03  6.20302
    """
    parameters, model, simulation_options, simulate_kwargs = args
    return model.simulate(parameters, simulation_options, **simulate_kwargs)


def run_simulations(
    model,
    parameter_samples: pd.DataFrame,
    simulation_options: dict,
    n_cpu: int = -1,
    simulate_kwargs: dict = None,
):
    """
    Run a batch of simulations in parallel for a given model.

    Parameters
    ----------
    model : Model
        The model object implementing `.simulate`.
    parameter_samples : pandas.DataFrame
        DataFrame containing parameter sets, one row per simulation.
        Columns are parameter names, rows represent simulation cases.
    simulation_options : dict
        Dictionary of simulation options, keys depend on the model.
    n_cpu : int, default=-1
        Number of CPU cores to use:
        - ``-1`` → all CPUs but one.
        - ``0`` → all available CPUs.
        - ``1`` → sequential execution.
        - ``>1`` → specified number of CPUs.
    simulate_kwargs : dict, optional
        Additional keyword arguments to forward to the model's `.simulate`.

    Returns
    -------
    list of tuple
        Each tuple contains ``(parameters, simulation_options, result)``:
        - parameters : dict
            Parameter set used for the simulation.
        - simulation_options : dict
            Simulation options used.
        - result : pandas.DataFrame
            Simulation result.
        ...

        Examples
        --------
        >>> import pandas as pd
        >>> from tests.resources.pymodels import Ishigami
        >>> param_samples = pd.DataFrame([
        ...     {"x1": 0.1, "x2": 0.2, "x3": 0.3},
        ...     {"x1": 1.0, "x2": 2.0, "x3": 3.0},
        ... ])
        >>> results = run_simulations(
        ...     Ishigami(),
        ...     param_samples,
        ...     simulation_options={"start": "2020-01-01", "end": "2020-01-02", "timestep": "D"},
        ...     n_cpu=1
        ... )
        >>> results[0][2].head()
                    res
        2020-01-01  0.376201
        2020-01-02  00.376201

        >>> results[0][0]
        {'x1': 0.1, 'x2': 0.2, 'x3': 0.3}

    """
    simulate_kwargs = {} if simulate_kwargs is None else simulate_kwargs

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
            results = [model.simulate(group[0], simulation_options, **simulate_kwargs)]
        else:
            with Pool(n_cpu) as pool:
                results = pool.map(
                    run_model,
                    [
                        (param, model, simulation_options, simulate_kwargs)
                        for param in group
                    ],
                )

        combined_result = [
            (param, simulation_options, res) for param, res in zip(group, results)
        ]
        collect.append(combined_result)

    return [item for sublist in collect for item in sublist]


def run_list_of_models_in_parallel(
    models_list: list[Model],
    simulation_options: dict,
    n_cpu: int = -1,
    parameter_dicts: list[dict[str, any]] = None,
    simulate_kwargs: dict = None,
):
    """
    Run a list of models in parallel.

        Parameters
        ----------
        models_list : list of Model
            List of model objects to be simulated in parallel.
        simulation_options : dict
            Dictionary of simulation options shared across all models.
        n_cpu : int, default=-1
            Number of CPU cores to use:
            - ``-1`` → all CPUs but one.
            - ``0`` → all available CPUs.
            - ``1`` → sequential execution.
            - ``>1`` → specified number of CPUs.
        parameter_dicts : list of dict, optional
            List of parameter dictionaries, one per model.
            If None, models are simulated with ``None`` as parameter set.
        simulate_kwargs : dict, optional
            Additional keyword arguments to forward to `.simulate`.

        Returns
        -------
        list of pandas.DataFrame
            Simulation results for each model.

        ...

        Examples
        --------
        >>> from tests.resources.pymodels import Ishigami
        >>> models = [Ishigami(), Ishigami()]
        >>> params = [{"x1": 1.0, "x2": 1.0, "x3": 1.0},
        ...           {"x1": 1.0, "x2": 2.0, "x3": 2.0}]
        >>> results = run_list_of_models_in_parallel(
        ...     models,
        ...     simulation_options={"start": "2020-01-01", "end": "2020-01-01", "timestep": "D"},
        ...     n_cpu=-1,
        ...     parameter_dicts=params
        ... )
        >>> results
                    res
        2020-01-01  5.483882
                    res
        2020-01-01  7.975577]

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
                    parameter_dict=params_group[i],
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
