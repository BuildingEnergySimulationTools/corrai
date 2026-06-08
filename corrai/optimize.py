from abc import ABC
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Binary, Choice, Integer, Real
from scipy.optimize import differential_evolution, minimize_scalar, minimize, curve_fit

from corrai.base.math import METHODS
from corrai.base.model import Model
from corrai.base.parameter import Parameter
from corrai.base.utils import check_indicators_configs
from corrai.sampling import Sample, SampleMethodsMixin


def check_duplicate_params(params: list["Parameter"]) -> None:
    """
    Validate that parameter names are unique.

    Parameters
    ----------
    params : list of Parameter
        List of parameters to check.

    Raises
    ------
    ValueError
        If two parameters share the same name.

    Examples
    --------
    >>> from corrai.base.parameter import Parameter
    >>> p1 = Parameter(name="x1", model_property="a", interval=(0, 1))
    >>> p2 = Parameter(name="x1", model_property="b", interval=(0, 1))
    >>> check_duplicate_params([p1, p2])
    Traceback (most recent call last):
        ...
    ValueError: Duplicate parameter name: 'x1'
    """
    seen = set()
    for p in params:
        if p.name in seen:
            raise ValueError(f"Duplicate parameter name: {p.name}")
        seen.add(p.name)


class ModelEvaluator:
    """
    Evaluate a model with respect to a set of parameters and compute indicators
    Series from simulation results.

    This class acts as an interface between parameters, the model, and an optimizer.
    It provides and objective function suitable for SciPy optimizers.

    Parameters
    ----------
    parameters : list of Parameter
        List of corrai Parameters.
    model : Model
        Corrai Model object. `get_property_values` method must be implemented to use
        get_initial_values method.
    store_results : bool, optional
        If True, stores results in an internal `Sample` instance. Default is True.

    Attributes
    ----------
    parameters : list of Parameter
        Model parameters used in evaluation.
    model : Model
        The underlying model being evaluated.
    sample : Sample
        Stores samples and simulation results if `store_results=True`.

    Examples
    --------
    >>> from corrai.base.model import Ishigami
    >>> from corrai.optimize import ModelEvaluator

    >>> param_list = [
    ...     Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
    ...     Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
    ...     Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
    ... ]

    >>> my_evaluator = ModelEvaluator(
    ...     parameters=param_list,
    ...     model=Ishigami(),
    ... )

    >>> my_evaluator.intervals
    [(-3.14159265359, 3.14159265359),
    (-3.14159265359, 3.14159265359),
    (-3.14159265359, 3.14159265359)]

    >>> my_evaluator.get_initial_values()
    [1, 2, 3]

    >>> my_evaluator.evaluate(
    ...     parameter_value_pairs=[
    ...         (param_list[0], -3.14 / 2),
    ...         (param_list[1], 0),
    ...         (param_list[2], -3.14),
    ...     ],
    ...     indicators_configs=["res"],
    ... )
    res   -10.721168

    >>> my_evaluator.scipy_obj_function([-3.14 / 2, 0, -3.14], "res", None, None)
    -10.721167816657914
    """

    def __init__(
        self, parameters: list[Parameter], model: Model, store_results: bool = True
    ):
        self.parameters = parameters
        self.model = model
        if store_results:
            self.sample = Sample(self.parameters, is_dynamic=model.is_dynamic)

    @property
    def intervals(self) -> list[tuple[int | float, int | float]]:
        return [par.interval for par in self.parameters]

    def get_initial_values(self, relative_is_one: bool = True) -> list[int | float]:
        init_dict = {}

        for par in self.parameters:
            val = par.init_value

            if val is None:
                if par.relabs == "Relative" and relative_is_one:
                    val = 1.0
                elif par.relabs == "Absolute":
                    if isinstance(par.model_property, str):
                        val = self.model.get_property_values([par.model_property])
                    else:
                        raise ValueError(
                            f"Failed for parameter {par}: "
                            "Cannot retrieve several property values from a single "
                            "parameter"
                        )

            init_dict[par.name] = val

        return [
            x for v in init_dict.values() for x in (v if isinstance(v, list) else [v])
        ]

    def evaluate(
        self,
        parameter_value_pairs: list[tuple[Parameter, str | int | float]],
        indicators_configs: list[str]
        | list[tuple[str, str | Callable] | tuple[str, str | Callable, pd.Series]]
        | None,
        simulation_options: dict = None,
        simulation_kwargs=None,
    ) -> pd.Series:
        """
        Run a model simulation and compute indicators. Return a pandas Series with
        indicators name as index.

        Parameters
        ----------
        parameter_value_pairs : list of tuple(Parameter, str or int or float)
            List of parameters and their values to simulate.
        indicators_configs : list of str or list of tuple
            - If the model is **static**: list of indicator names (strings).
            - If the model is **dynamic**: list of tuples specifying how to
              aggregate results. Each tuple has the form:
              - (col, func)
              - (col, func, reference)

              where:
                * col : str
                  Column name in the simulation results.
                * func : str or Callable
                  Aggregation function (either a method name registered in
                  `METHODS` or a callable).
                * reference : optional
                  time series that will be a reference for error aggreation method
                  (eg. nmbe, cv_rmse, mean_squarred_error).
        simulation_options : dict, optional
            Simulation options passed to the model.
        simulation_kwargs : dict, optional
            Additional keyword arguments for the simulation.

        Returns
        -------
        pandas.Series
            - For static models: direct simulation results.
            - For dynamic models: aggregated indicator results.

        Raises
        ------
        ValueError
            If `indicators_configs` is invalid for the model type.
        """

        res = self.model.simulate_parameter(
            parameter_value_pairs, simulation_options, simulation_kwargs
        )

        self.sample.add_samples(
            np.array([[val[1] for val in parameter_value_pairs]]), [res]
        )

        check_indicators_configs(self.model.is_dynamic, indicators_configs)

        if self.model.is_dynamic:
            results = pd.Series()
            for config in indicators_configs:
                col, func, *extra = config
                series = res[col]
                if isinstance(func, str):
                    func = METHODS[func]
                results[col] = func(series, *extra)
            return pd.Series(results)
        else:
            return res[indicators_configs] if indicators_configs is not None else res

    def scipy_obj_function(self, x: np.ndarray, *args) -> float:
        indicator_config, simulation_options, simulation_kwargs = args
        res = self.evaluate(
            [(par, val) for par, val in zip(self.parameters, x)],
            [indicator_config],
            simulation_options,
            simulation_kwargs,
        )
        """
        Objective function compatible with SciPy optimizers.

        Parameters
        ----------
        x : numpy.ndarray
            Array of parameter values in the same order as `self.parameters`.
        *args : tuple
            A 3-element tuple containing:
            - indicator_config : str or tuple
              Indicator specification. If the model is static, must be a string
              (the column name). If the model is dynamic, must be a tuple of the
              form `(col, func)` or `(col, func, reference)` where `func` is either a
              registered method name or a callable.
            - simulation_options : dict
              Options to configure the model simulation. Must pass None if no 
              simulation_options are required.
            - simulation_kwargs : dict
              Additional keyword arguments for simulation. Must pass None if no 
              simulation_kwargs are required.

        Returns
        -------
        float
            The evaluated indicator value corresponding to `indicator_config`.

        Raises
        ------
        ValueError
            If the configuration does not match the model type.
        """
        if isinstance(indicator_config, str):
            if self.model.is_dynamic:
                raise ValueError(
                    "Model is dynamic. An aggregation method must be "
                    "passed ['indicator', method, method_kwargs]"
                )
            loc_idx = indicator_config

        else:
            if not self.model.is_dynamic:
                raise ValueError(
                    "An aggregation method was passed although model is not dynamic "
                    "'indicator_config' must be a string"
                )
            loc_idx = indicator_config[0]

        return res.loc[loc_idx]

    def scipy_scalar_obj_function(self, x: float, *args):
        return self.scipy_obj_function(np.array([x]), *args)

    def plot_parameter_forest(
        self,
        optimal_values: "dict[str, float] | list[float] | pd.Series",
        mode: str = "normalized",
        title: str = None,
        **plot_kwargs,
    ) -> go.Figure:
        """
        Forest plot of this evaluator's parameters with bounds and optimal values.

        Delegates to the module-level :func:`plot_parameter_forest`.
        See that function for full documentation.

        Parameters
        ----------
        optimal_values : dict, list, or pd.Series
            Optimal value per parameter. When a list, order must match
            ``self.parameters``.
        mode : {"normalized", "absolute", "relative"}, default "normalized"
        title : str, optional
        **plot_kwargs
            Forwarded to ``fig.update_layout`` / ``fig.update_traces``.
        """
        return plot_parameter_forest(
            self.parameters,
            optimal_values,
            mode=mode,
            title=title,
            **plot_kwargs,
        )


class PymooModelEvaluator(ModelEvaluator):
    """
    Specialization of ModelEvaluator for use with pymoo optimizers.

    This class wraps the evaluation logic so it can be used as a pymoo-compatible
    problem definition. It handles vectorized evaluations of parameter sets.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
        indicators_configs: list[str]
        | list[
            tuple[str, str | Callable] | tuple[str, str | Callable, pd.Series]
        ] = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ):
        super().__init__(parameters, model)
        self.indicators_configs = indicators_configs
        self.simulation_options = (
            {} if simulation_options is None else simulation_options
        )
        self.simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs

    def evaluate_indicators_configs(
        self, parameter_value_pairs: list[tuple[Parameter, str | int | float]]
    ) -> pd.Series:
        return self.evaluate(
            parameter_value_pairs,
            self.indicators_configs,
            self.simulation_options,
            self.simulation_kwargs,
        )


class CorraiProblem(ElementwiseProblem, ABC):
    def __init__(
        self,
        *,
        parameters: list[Parameter],
        evaluators: list[PymooModelEvaluator],
        objective_ids: list[str],
        constraint_ids: list[str] | None = None,
    ):
        self.parameters = parameters
        self.evaluators = evaluators
        self.objective_ids = objective_ids
        self.constraint_ids = constraint_ids if constraint_ids else []
        self.sample = Sample(parameters, is_dynamic=False)

        check_duplicate_params(parameters)
        for ev in self.evaluators:
            for par in ev.parameters:
                if par.name not in self.parameters_names:
                    raise ValueError(
                        f"Parameter {par.name} of {ev} was not found in CorraiProblem"
                        "Parameters list"
                    )

    @property
    def parameters_names(self):
        return [par.name for par in self.parameters]

    @property
    def values(self):
        return self.sample.values

    @property
    def results(self):
        return self.sample.get_static_results_as_df()

    def _post_evaluate(
        self,
        parameter_value_pairs: list[tuple[Parameter, str | int | float]],
        out,
        *args,
        **kwargs,
    ):
        list_res = []
        for modev in self.evaluators:
            res = modev.evaluate_indicators_configs(parameter_value_pairs)
            list_res.append(res.to_dict())
        acc = {k: v for d in list_res for k, v in d.items()}

        self.sample.add_samples(
            np.array([[conf[1] for conf in parameter_value_pairs]]), [pd.Series(acc)]
        )
        out["F"] = [acc[name] for name in self.objective_ids]
        out["G"] = (
            [acc[name] for name in self.constraint_ids] if self.constraint_ids else []
        )


class RealContinuousProblem(CorraiProblem):
    """
    Continuous optimization problem for real-valued parameters in pymoo.

    This class extends ``CorraiProblem`` and ``pymoo.ElementwiseProblem`` to
    represent optimization problems where all decision variables are real-valued.
    It wraps corrai ``Parameter`` objects, model evaluators, and objective/constraint
    definitions into a pymoo-compatible interface.

    Parameters
    ----------
    parameters : list of Parameter
        List ``Parameter`` with a Real `ptype`
    evaluators : list of PymooModelEvaluator
        Evaluator objects that run models and compute performance indicators
        (objectives and/or constraints) given simulation options simulation kwargs
        and indicators configurations.
    objective_ids : list of str
        Names of the indicators to be minimized or maximized as objectives.
        The order defines their position in the objective vector ``F``.
    constraint_ids : list of str, optional
        Names of the indicators to be treated as inequality constraints.
        If None (default), no constraints are applied.

    Attributes
    ----------
    parameters : list of Parameter
        Problem decision variables.
    evaluators : list of PymooModelEvaluator
        Model evaluators associated with the problem.
    objective_ids : list of str
        Ordered list of objective indicator names.
    constraint_ids : list of str
        Ordered list of constraint indicator names.
    sample : Sample
        Stores past evaluations (parameter values and results).
    parameters_names : list of str
        Names of all parameters.
    values : dict
        Current sample values keyed by parameter name.
    results : pandas.DataFrame
        Static results collected from evaluations.

    Methods
    -------
    _evaluate(x, out, *args, **kwargs)
        Evaluate objectives and constraints at the given point ``x``.

    Notes
    -----
    - All parameters must be real-valued (``ptype="Real"``).
    - Lower and upper bounds are extracted from the ``interval`` attribute
      of each parameter.
    - The class automatically constructs the pymoo-compatible problem with
      ``n_var``, ``n_obj``, ``n_ieq_constr``, ``xl``, and ``xu``.
    """

    def __init__(
        self,
        *,
        parameters: list[Parameter],
        evaluators: list[PymooModelEvaluator],
        objective_ids: list[str],
        constraint_ids: list[str] | None = None,
    ):
        super().__init__(
            parameters=parameters,
            evaluators=evaluators,
            objective_ids=objective_ids,
            constraint_ids=constraint_ids,
        )

        ElementwiseProblem.__init__(
            self,
            n_var=len(self.parameters),
            n_obj=len(self.objective_ids),
            n_ieq_constr=len(self.constraint_ids),
            xl=np.array([p.interval[0] for p in parameters], dtype=float),
            xu=np.array([p.interval[1] for p in parameters], dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pairs = [(p, v) for p, v in zip(self.parameters, x)]
        self._post_evaluate(pairs, out, *args, **kwargs)


class MixedProblem(CorraiProblem):
    """
    Mixed-variable optimization problem for real, integer, binary, and choice
    parameters in pymoo.

    ``MixedProblem`` extends ``CorraiProblem`` and ``pymoo.ElementwiseProblem`` to
    represent optimization problems where decision variables may be heterogeneous.
    It builds a pymoo-compatible variable dictionary with the appropriate type
    (Real, Integer, Binary, or Choice) for each parameter.

    Parameters
    ----------
    parameters : list of Parameter
        List of ``Parameter`` objects defining the optimization variables.
        Supported types are ``Real``, ``Integer``, ``Binary``, and ``Choice``.
    evaluators : list of PymooModelEvaluator
       Evaluator objects that run models and compute performance indicators
        (objectives and/or constraints) given simulation options simulation kwargs
        and indicators configurations.
    objective_ids : list of str
        Names of the indicators to be minimized or maximized as objectives.
        The order defines their position in the objective vector ``F``.
    constraint_ids : list of str, optional
        Names of the indicators to be treated as inequality constraints.
        Defaults to an empty list if not provided.

    Attributes
    ----------
    parameters : list of Parameter
        Problem decision variables.
    evaluators : list of PymooModelEvaluator
        Model evaluators associated with the problem.
    objective_ids : list of str
        Ordered list of objective indicator names.
    constraint_ids : list of str
        Ordered list of constraint indicator names.
    sample : Sample
        Stores past evaluations (parameter values and results).
    parameters_names : list of str
        Names of all parameters.
    values : dict
        Current sample values keyed by parameter name.
    results : pandas.DataFrame
        Static results collected from evaluations.

    Methods
    -------
    _evaluate(x, out, *args, **kwargs)
        Evaluate objectives and constraints at the given point ``x``.

    Notes
    -----
    - Each parameter type is mapped internally to the corresponding pymoo variable:

      * ``Real`` → :class:`pymoo.core.variable.Real` with bounds.
      * ``Integer`` → :class:`pymoo.core.variable.Integer` with bounds.
      * ``Binary`` → :class:`pymoo.core.variable.Binary`.
      * ``Choice`` → :class:`pymoo.core.variable.Choice` with enumerated options.

    - If a ``Choice`` parameter does not define ``values``, a ``ValueError`` is raised.
    - Objective and constraint values are automatically extracted from evaluator results.
    - See ``CorraiProblem`` for common attributes and evaluation workflow.
    """

    def __init__(
        self,
        *,
        parameters: list[Parameter],
        evaluators: list[PymooModelEvaluator],
        objective_ids: list[str],
        constraint_ids: list[str] | None = None,
    ):
        super().__init__(
            parameters=parameters,
            evaluators=evaluators,
            objective_ids=objective_ids,
            constraint_ids=constraint_ids,
        )

        vars_dict = {}
        for p in self.parameters:
            if p.ptype == "Real":
                lo, hi = p.interval
                vars_dict[p.name] = Real(bounds=(lo, hi))
            elif p.ptype == "Integer":
                lo, hi = p.interval
                vars_dict[p.name] = Integer(bounds=(lo, hi))
            elif p.ptype == "Binary":
                vars_dict[p.name] = Binary()
            elif p.ptype == "Choice":
                if p.values is None:
                    raise ValueError(
                        f"Parameter {p.name!r} of type Choice requires 'values'"
                    )
                vars_dict[p.name] = Choice(options=list(p.values))
            else:
                raise ValueError(
                    f"Unsupported ptype={p.ptype!r} for parameter {p.name!r}"
                )
        ElementwiseProblem.__init__(
            self,
            vars=vars_dict,
            n_obj=len(self.objective_ids),
            n_ieq_constr=len(self.constraint_ids),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pairs = [(p, x[p.name]) for p in self.parameters]
        self._post_evaluate(pairs, out)


class SciOptimizer(SampleMethodsMixin):
    """
    Optimization wrapper for models using SciPy.

    This class provides a convenient interface to optimize model parameters
    with respect to specified indicators. It leverages the `ModelEvaluator`
    for simulation and evaluation, and uses SciPy's global optimization
    algorithm such as `differential_evolution` to find optimal parameter sets.

    Parameters
    ----------
    parameters : list of Parameter
        List of corrai Parameters to be optimized.
    model : Model
        Corrai Model object that provides simulation capabilities.

    Attributes
    ----------
    model_evaluator : ModelEvaluator
        Underlying evaluator used for simulations and objective evaluation and results
        storage.

    Examples
    --------
    >>> from corrai.optimize import SciOptimizer
    >>> from corrai.base.model import Ishigami

    >>> param_list = [
    ...     Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
    ...     Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
    ...     Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
    ... ]

    >>> sci_opt = SciOptimizer(
    ...     parameters=param_list,
    ...     model=Ishigami(),
    ... )

    >>> res_opt = sci_opt.diff_evo_minimize("res")

    >>> res_opt.fun
    -10.74090910277037

    >>> res_opt.x
    array([-1.57080718e+00, -2.46536703e-07,  3.14159265e+00])
    """

    def __init__(
        self,
        parameters: list[Parameter],
        model: Model,
    ):
        self.model_evaluator = ModelEvaluator(parameters, model, True)

    @property
    def parameters(self):
        return self.model_evaluator.parameters

    @property
    def sample(self):
        return self.model_evaluator.sample

    @property
    def values(self):
        return self.model_evaluator.sample.values

    @property
    def results(self):
        if self.model_evaluator.model.is_dynamic:
            return self.model_evaluator.sample.results
        else:
            return self.model_evaluator.sample.get_static_results_as_df()

    def scalar_minimize(
        self,
        indicator_config: str
        | tuple[str, str | Callable]
        | tuple[str, str | Callable, pd.Series | None],
        simulation_options: dict = None,
        simulation_kwargs=None,
        bracket=None,
        method=None,
        tol=None,
        options=None,
    ):
        """
        Minimize a scalar indicator using SciPy's scalar optimization routines.

        This method is suitable for problems with a **single parameter** only.

        Parameters
        ----------
        indicator_config : str or tuple
            Indicator specification passed to the objective function:
            - If the model is **static**: a string representing the indicator name.
            - If the model is **dynamic**: a tuple of the form
              (col, func) or (col, func, reference) where:
                * col : str
                  Column name in the simulation results.
                * func : str or Callable
                  Aggregation function (either a method name registered in
                  `METHODS` or a Python callable).
                * reference : optional
                  reference time series for error function (eg. nmbe, cv_rmse).
        simulation_options : dict, optional
            Options for the simulation (e.g., stop time, solver settings).
        simulation_kwargs : dict, optional
            Additional keyword arguments for simulation.
        bracket : tuple of float, optional
            Bracketing interval for methods that require it (e.g., "brent").
        method : str, optional
            Optimization method to use. Can be one of {"brent", "bounded", "golden"}.
            See :func:`scipy.optimize.minimize_scalar` for details.
        tol : float, optional
            Tolerance for termination. Interpretation depends on the method.
        options : dict, optional
            Additional solver-specific options.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result of the optimization, including optimal parameter value `x`
            and corresponding function value `fun`.

        Raises
        ------
        ValueError
            If more than one parameter is defined, since scalar optimization
            only supports a single variable.

        Notes
        -----
        - Only use a list of one parameter.
        - Uses :func:`scipy.optimize.minimize_scalar`.
        """

        bounds = (
            None if method in ["Brent", "Golden"] else self.model_evaluator.intervals[0]
        )

        return minimize_scalar(
            fun=self.model_evaluator.scipy_scalar_obj_function,
            bounds=bounds,
            args=(indicator_config, simulation_options, simulation_kwargs),
            bracket=bracket,
            method=method,
            tol=tol,
            options=options,
        )

    def minimize(
        self,
        indicator_config: str
        | tuple[str, str | Callable]
        | tuple[str, str | Callable, pd.Series],
        simulation_options: dict = None,
        simulation_kwargs=None,
        x0: list[float] = None,
        method=None,
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
    ):
        """
        This method wraps `scipy.optimize.minimize`

        Parameters
        ----------
        indicator_config : str or tuple
            Indicator configuration passed to `ModelEvaluator.scipy_obj_function`:
            - If the model is **static**: a string representing the indicator name.
            - If the model is **dynamic**: a tuple of the form
              (indicator, func) or (indicator, func, reference) where:
                * indicator : str
                  Indicator name in the simulation results.
                * func : str or Callable
                  Aggregation function (method name registered in `METHODS`
                  or a Python callable).
                * reference : optional
                  Reference time series if the aggregation function is an
                  error metric such as nmbe, cv_rmse, or mean_squared_error.

        simulation_options : dict, optional
            Options for the simulation (e.g., stop time, solver settings).

        simulation_kwargs : dict, optional
            Additional keyword arguments for simulation.

        x0 : list of float, optional
            Initial guess for the optimization variables.
            If None, the initial values are set to the mean of each
            parameter interval.

        method : str or callable, optional
            Optimization method to use (e.g., 'BFGS', 'L-BFGS-B', 'SLSQP').
            Passed directly to `scipy.optimize.minimize`.

        jac : callable or bool, optional
            Function computing the gradient of the objective, or a boolean
            indicating whether the objective returns the gradient.

        hess : callable, optional
            Function computing the Hessian matrix of the objective.

        hessp : callable, optional
            Function computing the Hessian-vector product.

        bounds : sequence, optional
            Bounds on variables for bounded optimization methods.

        constraints : sequence, optional
            Constraints definition for constrained optimization.

        tol : float, optional
            Tolerance for convergence.

        callback : callable, optional
            Function called after each iteration.

        options : dict, optional
            Additional solver-specific options.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result of the optimization. Accessible also via the `result`
            attribute.

        Notes
        -----
        This method relies on `scipy.optimize.minimize`, which implements
        gradient-based and derivative-free local optimization algorithms.
        It is best suited for smooth problems and may converge to a local
        minimum depending on the initial guess.

        For global optimization, consider using `diff_evo_minimize`.
        """

        if x0 is None:
            x0 = [float(np.mean(par.interval)) for par in self.parameters]

        return minimize(
            self.model_evaluator.scipy_obj_function,
            x0,
            args=(indicator_config, simulation_options, simulation_kwargs),
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
        )

    def diff_evo_minimize(
        self,
        indicator_config: str
        | tuple[str, str | Callable]
        | tuple[str, str | Callable, pd.Series],
        simulation_options: dict = None,
        simulation_kwargs=None,
        maxiter=1000,
        tol=0.01,
        rng=None,
        workers=1,
    ):
        """
        Minimize an indicator using SciPy's differential evolution algorithm.

        Parameters
        ----------
        indicator_config : str or tuple
            Indicator configuration passed to `ModelEvaluator.scipy_obj_function`:
            - If the model is **static**: a string representing the indicator name.
            - If the model is **dynamic**: a tuple of the form
              (indicator, func) or (indicator, func, reference) where:
                * indicator : str
                  indicator name in the simulation results.
                * func : str or Callable
                  Aggregation function (method name registered in `METHODS`
                  or a Python callable).
                * reference : optional
                  reference time series if aggregation function is an error function
                  such as nmbe, cv_rmse, mean_squared_error.
        simulation_options : dict, optional
            Options for the simulation (e.g., stop time, solver settings).
        simulation_kwargs : dict, optional
            Additional keyword arguments for simulation.
        maxiter : int, optional
            Maximum number of generations for the optimizer. Default is 1000.
        tol : float, optional
            Tolerance for convergence. Default is 0.01.
        rng : int or RandomState or Generator, optional
            Random number generator seed or instance. Default is None.
        workers : int or map-like callable, optional
            Number of parallel workers. Can be set to -1 to use all processors.
            Default is 1 (no parallelism).

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result of the optimization. Accessible also via the `result` attribute.

        Notes
        -----
        This method uses `scipy.optimize.differential_evolution`, which is a
        global optimization algorithm suitable for continuous parameter spaces.
        """

        return differential_evolution(
            func=self.model_evaluator.scipy_obj_function,
            bounds=self.model_evaluator.intervals,
            args=(indicator_config, simulation_options, simulation_kwargs),
            maxiter=maxiter,
            tol=tol,
            rng=rng,
            workers=workers,
        )

    def curve_fit(
        self,
        indicator_config,
        simulation_options=None,
        simulation_kwargs=None,
        p0=None,
        bounds=None,
        **kwargs,
    ):
        """
        Use non-linear least squares to fit model parameters.

        This method wraps :func:`scipy.optimize.curve_fit` to calibrate the
        parameters of the underlying model so that simulated outputs best match
        reference data.

        Parameters
        ----------
        indicator_config : list or tuple
            Configuration(s) describing the outputs to match. Each config should be:
            (col, func, reference), where:
                * col : str
                  Column name in simulation results.
                * func : str or Callable
                  Aggregation function (currently unused here but kept for consistency).
                * reference : pd.Series
                  Reference data to fit against.
            Multiple configs can be provided as a list.
        simulation_options : dict, optional
            Options passed to the simulation routine.
        simulation_kwargs : dict, optional
            Additional keyword arguments passed to the simulation.
        p0 : array-like, optional
            Initial guess for the parameters. If None, uses the midpoint of intervals.
        bounds : 2-tuple of array-like, optional
            Lower and upper bounds on parameters. If None, derived from
            `self.model_evaluator.intervals`.
        **kwargs :
            Additional keyword arguments passed to :func:`scipy.optimize.curve_fit`.

        Returns
        -------
        popt : array
            Optimal values for the parameters.
        pcov : 2-D array
            Estimated covariance of `popt`.

        Notes
        -----
        - The function being fitted internally runs a full model simulation.
        - The independent variable is a dummy index, as only the output values matter.
        - Multiple indicators are concatenated into a single residual vector.
        """

        if bounds is None:
            bounds = list(zip(*self.model_evaluator.intervals))

        if p0 is None:
            p0 = [np.mean(b) for b in self.model_evaluator.intervals]

        configs = (
            indicator_config
            if isinstance(indicator_config, list)
            else [indicator_config]
        )

        references = [cfg[2].values for cfg in configs]
        reference = np.concatenate(references)

        def wrapped_func(x, *params):
            self.model_evaluator.scipy_obj_function(
                params,
                configs[0],
                simulation_options,
                simulation_kwargs,
            )

            res = self.model_evaluator.model.simulate(
                simulation_options=simulation_options,
                **(simulation_kwargs or {}),
            )

            outputs = [res[cfg[0]].values for cfg in configs]
            return np.concatenate(outputs)

        x_dummy = np.arange(len(reference))

        return curve_fit(
            wrapped_func,
            x_dummy,
            reference,
            p0=p0,
            bounds=bounds,
            **kwargs,
        )


_FOREST_MODES = ["normalized", "absolute", "relative"]


def _apply_figure_kwargs(fig: go.Figure, **kwargs) -> None:
    for key, val in kwargs.items():
        try:
            fig.update_layout(**{key: val})
        except ValueError:
            fig.update_traces(**{key: val})


def _forest_label(value: float, relabs: str, mode: str) -> str:
    """Format a bound or optimal value for forest plot annotation."""
    if mode == "normalized":
        return ""
    if mode == "relative" and relabs == "Relative":
        return f"{value * 100:.4g}%"
    return f"{value:.4g}"


def plot_parameter_forest(
    parameters: list[Parameter],
    optimal_values: dict[str, float] | list[float] | pd.Series,
    mode: str = "normalized",
    title: str = None,
    **plot_kwargs,
) -> go.Figure:
    """
    Forest plot of optimization parameters — parameters on the X-axis, normalized
    values on the Y-axis.

    Each parameter is drawn as a vertical bar spanning [0, 1] (normalized to its
    own bounds) with a diamond marker at the optimal value. Parameters with
    different units are thus comparable on the same scale.

    How bounds are labelled depends on ``mode``:

    * ``"normalized"`` — no value annotations; Y-axis ticks read 0 % … 100 %.
    * ``"absolute"``   — actual lower, upper, and optimal values are shown as
      text on each bar.
    * ``"relative"``   — parameters with ``relabs="Relative"`` are annotated in
      percent (e.g. ``interval=(0.2, 1.5)`` → ``"20 %"`` / ``"150 %"``);
      parameters with ``relabs="Absolute"`` fall back to actual values.

    Parameters without an ``interval`` (e.g. ``Choice``) are silently skipped.
    Hover is disabled on all traces.

    Parameters
    ----------
    parameters : list of Parameter
        Parameters defining the search space.
    optimal_values : dict, list, or pd.Series
        Optimal value per parameter after optimisation. When a list or array,
        order must match ``parameters``.
    mode : {"normalized", "absolute", "relative"}, default "normalized"
        Annotation style (see above).
    title : str, optional
        Plot title.
    **plot_kwargs
        Forwarded to ``fig.update_layout`` or ``fig.update_traces``.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> from corrai.base.parameter import Parameter
    >>> from corrai.optimize import plot_parameter_forest
    >>> params = [
    ...     Parameter("conductivity", interval=(0.03, 0.06), model_property="x"),
    ...     Parameter("thickness", interval=(0.05, 0.30), model_property="y"),
    ...     Parameter("temp_setpoint", interval=(18.0, 24.0), model_property="z"),
    ... ]
    >>> fig = plot_parameter_forest(
    ...     params,
    ...     {"conductivity": 0.04, "thickness": 0.12, "temp_setpoint": 21.0},
    ...     mode="absolute",
    ... )
    """
    if mode not in _FOREST_MODES:
        raise ValueError(f"mode must be one of {_FOREST_MODES}, got {mode!r}")

    # Include continuous (interval) and categorical (values/Choice) params; skip Binary
    params = [p for p in parameters if p.interval is not None or p.values is not None]
    if not params:
        raise ValueError("No parameters with interval bounds found.")

    all_param_names = {p.name for p in params}
    if isinstance(optimal_values, (list, np.ndarray)):
        opt_dict = {
            p.name: v
            for p, v in zip(parameters, optimal_values)
            if p.interval is not None or p.values is not None
        }
    elif isinstance(optimal_values, pd.Series):
        opt_dict = {k: v for k, v in optimal_values.items() if k in all_param_names}
    else:
        opt_dict = {k: v for k, v in optimal_values.items() if k in all_param_names}

    missing = [p.name for p in params if p.name not in opt_dict]
    if missing:
        raise ValueError(f"Missing optimal values for parameters: {missing}")

    names = [p.name for p in params]
    interval_params = [p for p in params if p.interval is not None]
    choice_params = [p for p in params if p.values is not None]

    # --- Normalized optimal positions
    opt_norms: dict[str, float] = {}
    for p in params:
        if p.interval is not None:
            lo, hi = p.interval
            v = float(opt_dict[p.name])
            opt_norms[p.name] = (v - lo) / (hi - lo) if hi != lo else 0.5
        else:
            n = len(p.values)
            positions = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]
            try:
                idx = list(p.values).index(opt_dict[p.name])
            except ValueError:
                raise ValueError(
                    f"Optimal value {opt_dict[p.name]!r} not among choices "
                    f"{p.values} for parameter {p.name!r}"
                )
            opt_norms[p.name] = positions[idx]

    # --- Labels
    annotate = mode != "normalized"
    lower_texts = {
        p.name: _forest_label(p.interval[0], p.relabs, mode) for p in interval_params
    }
    upper_texts = {
        p.name: _forest_label(p.interval[1], p.relabs, mode) for p in interval_params
    }
    # Optimal text: mode-aware for interval; empty for choice (tick labels already mark each position)
    all_opt_texts = {
        p.name: (
            _forest_label(float(opt_dict[p.name]), p.relabs, mode)
            if p.interval is not None
            else ""
        )
        for p in params
    }
    annotate_opt = annotate

    _bar_color = "darkblue"
    fig = go.Figure()

    # Trace: vertical lines for all params
    x_lines: list[str | None] = []
    y_lines: list[float | None] = []
    for name in names:
        x_lines.extend([name, name, None])
        y_lines.extend([0.0, 1.0, None])
    fig.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color=_bar_color, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Trace: interval lower bound ticks (y=0, text below)
    if interval_params:
        inames = [p.name for p in interval_params]
        fig.add_trace(
            go.Scatter(
                x=inames,
                y=[0.0] * len(inames),
                mode="markers+text" if annotate else "markers",
                marker=dict(
                    symbol="line-ew-open",
                    size=14,
                    color=_bar_color,
                    line=dict(width=2, color=_bar_color),
                ),
                text=[lower_texts[n] for n in inames],
                textposition="bottom center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Trace: interval upper bound ticks (y=1, text above)
    if interval_params:
        inames = [p.name for p in interval_params]
        fig.add_trace(
            go.Scatter(
                x=inames,
                y=[1.0] * len(inames),
                mode="markers+text" if annotate else "markers",
                marker=dict(
                    symbol="line-ew-open",
                    size=14,
                    color=_bar_color,
                    line=dict(width=2, color=_bar_color),
                ),
                text=[upper_texts[n] for n in inames],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Trace: choice tick marks — one entry per choice value, always labelled
    if choice_params:
        cx: list[str] = []
        cy: list[float] = []
        ctexts: list[str] = []
        ctextpositions: list[str] = []
        for p in choice_params:
            n = len(p.values)
            positions = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]
            for i, (val, pos) in enumerate(zip(p.values, positions)):
                cx.append(p.name)
                cy.append(pos)
                ctexts.append(str(val))
                if n == 1:
                    ctextpositions.append("top center")
                elif i == 0:
                    ctextpositions.append("bottom center")
                elif i == n - 1:
                    ctextpositions.append("top center")
                else:
                    ctextpositions.append("middle right")
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="markers+text",
                marker=dict(
                    symbol="line-ew-open",
                    size=14,
                    color=_bar_color,
                    line=dict(width=2, color=_bar_color),
                ),
                text=ctexts,
                textposition=ctextpositions,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Trace: optimal diamonds — always last
    fig.add_trace(
        go.Scatter(
            x=names,
            y=[opt_norms[n] for n in names],
            mode="markers+text" if annotate_opt else "markers",
            name="Optimal",
            marker=dict(
                symbol="diamond",
                size=13,
                color="orange",
                line=dict(width=1.5, color="darkorange"),
            ),
            text=[all_opt_texts[n] for n in names],
            textposition="middle right",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Lower text is "bottom center" → needs a bit of space below 0
    y_range = [-0.05, 1.1] if mode == "normalized" else [-0.18, 1.25]
    if mode == "normalized":
        y_tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
        y_ticktext = ["Lower (0%)", "25%", "50%", "75%", "Upper (100%)"]
        title_y = "Normalized position with bounds"
    else:
        y_tickvals = [0.0, 1.0]
        y_ticktext = ["Lower bound", "Upper bound"]
        title_y = "Position with bounds"

    b_margin = 100 if len(names) > 5 else 70

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickangle=-30 if len(names) > 5 else 0,
            ticklabelstandoff=4,
        ),
        yaxis=dict(
            title=title_y,
            range=y_range,
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=False,
        ),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        autosize=True,
        margin=dict(l=70, r=30, t=40, b=b_margin),
    )

    _apply_figure_kwargs(fig, **plot_kwargs)
    return fig
