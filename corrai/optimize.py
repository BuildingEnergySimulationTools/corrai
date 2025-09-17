from typing import Callable, Sequence, Iterable
from functools import wraps

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, minimize_scalar

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary

import plotly.graph_objects as go

from corrai.base.math import METHODS
from corrai.base.model import Model
from corrai.sampling import Sample
from corrai.base.parameter import Parameter


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
        | list[
            tuple[str, str | Callable] | tuple[str, str | Callable, pd.Series | None]
        ],
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

        if self.model.is_dynamic:
            if isinstance(indicators_configs[0], str):
                raise ValueError(
                    "Invalid 'indicators_configs'. Model is dynamic"
                    "At least 'method' is required"
                )
            results = pd.Series()
            for config in indicators_configs:
                col, func, *extra = config
                series = res[col]

                if isinstance(func, str):
                    func = METHODS[func]

                results[col] = func(series, *extra)
            return pd.Series(results)
        else:
            if isinstance(indicators_configs[0], tuple):
                raise ValueError(
                    "Invalid 'indicators_configs'. Model is static. "
                    "'indicators_configs' must be a list of string"
                )
            return res

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


class Problem(ElementwiseProblem):
    """
    Pymoo ``ElementwiseProblem`` wrapper for real-valued and mixed-variable optimization.

    This class bridges corrai ``Parameter`` objects with evaluation functions
    (``evaluators``) that compute objectives and constraints. It supports both:

    - **float mode**: when all parameters are real-valued (``ptype="Real"``),
      inputs are passed as numeric vectors (``list``/``ndarray``).
    - **mixed mode**: when at least one parameter is ``Choice``, ``Integer``,
      or ``Binary``, inputs are passed as dictionaries keyed by parameter name.

    Parameters
    ----------
    parameters : list of Parameter
        List of ``Parameter`` objects defining the optimization variables
        (name, type, bounds/values, and relative/absolute mode).
    evaluators : Sequence[Callable]
        Sequence of functions or callable objects. Each evaluator takes either:

        - a list of (Parameter, value) pairs,
        - or a dictionary mapping parameter names to values,

        and returns one of:

        - dict mapping indicator names to floats,
        - a pandas.Series,
        - or a scalar (only valid if one objective/constraint is defined).
    objective_ids : Sequence[str]
        Names of objectives to extract from evaluator results.
        Order defines their order in ``F``.
    constraint_ids : Sequence[str], optional
        Names of constraints to extract from evaluator results.
        Defaults to an empty list if no constraints are defined.

    Notes
    -----
    - In **float mode** (all Real parameters), ``x`` is a numeric vector.
    - In **mixed mode**, ``x`` is a dict with parameter names as keys.
    - Values returned by evaluators are automatically converted to floats.
    - Choice/Integer/Binary parameters are internally cast to the closest valid value.
    """

    def __init__(
        self,
        *,
        parameters: list[Parameter],
        evaluators: list[ModelEvaluator] | None = None,
        objective_ids: Sequence[str],
        constraint_ids: Sequence[str] | None = None,
    ):
        if not evaluators:
            raise ValueError("evaluators must be provided")

        self.parameters = list(parameters)
        check_duplicate_params(parameters)
        self.param_names = [p.name for p in self.parameters]
        self.evaluators = list(evaluators)
        self.objective_ids = list(objective_ids)
        self.constraint_ids = list(constraint_ids) if constraint_ids else []

        self.is_all_real = all(p.ptype == "Real" for p in self.parameters)
        if self.is_all_real:
            xl = np.array([p.interval[0] for p in self.parameters], dtype=float)
            xu = np.array([p.interval[1] for p in self.parameters], dtype=float)
            super().__init__(
                n_var=len(self.parameters),
                n_obj=len(self.objective_ids),
                n_ieq_constr=len(self.constraint_ids),
                xl=xl,
                xu=xu,
            )
            self._mode = "float"
        else:
            vars_dict = {}
            for p in self.parameters:
                if p.ptype == "Real":
                    lo, hi = p.interval
                    vars_dict[p.name] = Real(bounds=(float(lo), float(hi)))
                elif p.ptype == "Integer":
                    lo, hi = p.interval
                    vars_dict[p.name] = Integer(bounds=(int(lo), int(hi)))
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
            super().__init__(
                vars=vars_dict,
                n_obj=len(self.objective_ids),
                n_ieq_constr=len(self.constraint_ids),
            )
            self._mode = "mixed"

    @staticmethod
    def _cast_value(p, v):
        """
        Cast a raw value to the correct Python
        type for a given parameter.

        Parameters
        ----------
        p : Parameter
            Parameter definition.
        v : Any
            Raw value from the optimizer.

        Returns
        -------
        Any
            Value adjusted to match the parameter type.

        Examples
        --------
        >>> from corrai.base.parameter import Parameter
        >>> p = Parameter(name="b", ptype="Binary")
        >>> Problem._cast_value(p, 0.7)
        True

        >>> p = Parameter(name="alpha", ptype="Choice", values=[0.2, 0.4, 0.5])
        >>> Problem._cast_value(p, 0.2)
        0.2

        >>> p = Parameter(name="alpha", ptype="Choice", values=[0.2, 0.4, 0.5])
        >>> Problem._cast_value(p, 0.3)
        ValueError: Unexpected Choice value 1 for parameter alpha

        """
        if p.ptype == "Integer":
            return int(round(float(v)))
        if p.ptype == "Real":
            return float(v)
        if p.ptype == "Binary":
            return bool(round(float(v)))
        if p.ptype == "Choice":
            if v in p.values:
                return v
            else:
                raise ValueError(
                    f"Unexpected Choice " f"value {v} for parameter {p.name}"
                )
        return v

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate objectives and constraints for a given solution.

        Parameters
        ----------
        x : list | dict
            Candidate solution.
        out : dict
            Output container to store results.

        Notes
        -----
        Depending on the optimizer and variable type, `x` can have two forms:

        - **list / ndarray** (standard Pymoo encoding): values are ordered
          as `self.parameters`. Example:
          ``[False, "nothing", -1, -3.17]``

        - **dict** (MixedVariable encoding): keys are parameter names,
          values are already typed. Example:
          ``{"alpha": 0.2, "radiation": True}``

        This method normalizes both cases into a list of
        ``(Parameter, value)`` pairs, with values adjusted by
        :func:`_cast_value`.

        Examples
        --------
        Example with two real parameters:

        >>> from corrai.base.parameter import Parameter
        >>> def line_model(pairs):
        ...     params = {p.name: v for p, v in pairs}
        ...     return {"y": params["a"] * 10 + params["b"]}
        >>>
        >>> param_a = Parameter(name="a", interval=(0, 4))
        >>> param_b = Parameter(name="b", interval=(0, 10))
        >>>
        >>> problem = Problem(
        ...     parameters=[param_a, param_b],
        ...     evaluators=[line_model],
        ...     objective_ids=["y"],
        ... )
        >>>
        >>> x = [1.0, 5.0]
        >>> out = {}
        >>> problem._evaluate(x, out)
        >>> out["F"]
        [15.0]

        """
        acc = {}
        total_ids = len(self.objective_ids) + len(self.constraint_ids)
        if isinstance(x, dict):
            pairs = [(p, self._cast_value(p, x[p.name])) for p in self.parameters]
        else:
            pairs = [(p, self._cast_value(p, v)) for p, v in zip(self.parameters, x)]
        param_dict = {p.name: v for p, v in pairs}

        for block in self.evaluators:
            if hasattr(block, "function"):
                res = block.function(parameter_value_pairs=pairs)
            else:
                try:
                    res = block(param_dict)
                except TypeError:
                    res = block(pairs)

            if isinstance(res, dict):
                acc.update({k: float(v) for k, v in res.items()})
            elif hasattr(res, "to_dict"):
                acc.update({k: float(v) for k, v in res.to_dict().items()})
            else:
                if total_ids != 1:
                    raise TypeError(
                        "Scalar returned but multiple objectives/constraints defined"
                    )
                target = (
                    self.objective_ids[0]
                    if self.objective_ids
                    else self.constraint_ids[0]
                )
                acc[target] = float(res)

        F = [acc[name] for name in self.objective_ids]
        G = [acc[name] for name in self.constraint_ids] if self.constraint_ids else []

        out["F"] = F
        out["G"] = G


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


class ObjectiveFunction:
    """
    Configure and evaluate an objective function for calibration or optimization.

    This class connects a :class:`~corrai.base.model.Model` with calibration
    parameters and user-defined indicators. It provides a generic way to run
    simulations, compute indicators, and expose a SciPy-compatible objective.

    Parameters
    ----------
    model : Model
        The model to be calibrated.
    simulation_options : dict
        Options passed to ``model.simulate``.
    parameters : list[Parameter]
        Calibration parameters (continuous or mixed types).
    indicators_config : dict[str, Callable | tuple[Callable, pd.Series | pd.DataFrame | None]]
        Mapping of indicator names (columns in simulation output) to functions:

        - ``f(series) -> float``
        - ``(f, ref)`` where ``f(series, ref) -> float``

        Examples: ``np.mean``, ``np.sum``, ``sklearn.metrics.mean_squared_error``.
    scipy_obj_indicator : str, optional
        Indicator to be used as scalar objective in
        :func:`scipy_obj_function`. Defaults to the first key of
        ``indicators_config``.

    Attributes
    ----------
    parameters : list[Parameter]
    simulation_options : dict
    indicators_config : dict
    scipy_obj_indicator : str

    Properties
    ----------
    bounds : list[tuple[float, float]]
        Continuous bounds from parameters (absolute or relative).
    init_values : list[float] | None
        Initial values if defined for all parameters.
    """

    def __init__(
        self,
        model: Model,
        simulation_options: dict,
        parameters: list[Parameter],
        indicators_config: dict[
            str, Callable | tuple[Callable, pd.Series | pd.DataFrame | None]
        ],
        scipy_obj_indicator: str | None = None,
    ):
        self.model = model
        self.parameters = parameters
        self.indicators_config = indicators_config
        self.simulation_options = simulation_options
        self.scipy_obj_indicator = (
            list(indicators_config.keys())[0]
            if scipy_obj_indicator is None
            else scipy_obj_indicator
        )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        bnds: list[tuple[float, float]] = []
        for p in self.parameters:
            if p.interval is None:
                raise ValueError(
                    f"Parameter {p.name!r} has no 'interval'; "
                    "continuous optimization requires continuous bounds."
                )
            lo, hi = p.interval

            if getattr(p, "relabs", None) == "Relative":
                init_val = p.init_value
                if init_val is None:
                    init_vals = self.model.get_property_values(
                        (p.model_property,)
                        if not isinstance(p.model_property, tuple)
                        else p.model_property
                    )
                    init_val = sum(init_vals) / len(init_vals)

                lo, hi = lo * init_val, hi * init_val

            bnds.append((float(lo), float(hi)))
        return bnds

    @property
    def init_values(self) -> list[float] | None:
        vals: list[float] = []
        for p in self.parameters:
            iv = p.init_value
            if iv is None:
                return None
            if isinstance(iv, (tuple, list)):
                if len(iv) != 1:
                    return None
                iv = iv[0]
            vals.append(float(iv))
        return vals

    def function(
        self,
        parameter_value_pairs: list[tuple["Parameter", float]] | None = None,
        kwargs: dict | None = None,
    ) -> dict[str, float]:
        """
        Run the model and compute indicators.

        Parameters
        ----------
        parameter_value_pairs : list[tuple[Parameter, float]], optional
            (Parameter, value) pairs used in ``model.simulate_parameter``.
        kwargs : dict, optional
            Extra keyword arguments passed to ``model.simulate``.

        Returns
        -------
        dict[str, float]
            Computed indicator values.

        Raises
        ------
        KeyError
            If a requested indicator is missing in simulation output.
        TypeError
            If model output is not a pandas DataFrame or Series.

        Examples
        --------
        >>> import numpy as np
        >>> from corrai.base.parameter import Parameter
        >>> from corrai.base.objfunctions import ObjectiveFunction
        >>> from tests.resources.pymodels import Ishigami
        >>>
        >>> model = Ishigami()
        >>> param_x = Parameter(
        ...     name="x", model_property="x", interval=(-np.pi, np.pi), init_value=2.0
        ... )
        >>> param_y = Parameter(
        ...     name="y", model_property="y", interval=(-np.pi, np.pi), init_value=1.0
        ... )
        >>>
        >>> simulation_options = {
        ...     "start": "2000-01-01",
        ...     "end": "2000-01-01",
        ...     "timestep": "h",
        ... }
        >>>
        >>> obj_fun = ObjectiveFunction(
        ...     model,
        ...     simulation_options,
        ...     parameters=[param_x, param_y],
        ...     indicators_config={"res": np.mean},
        ... )
        >>>
        >>> obj_fun.function(parameter_value_pairs=[(param_x, 2.0), (param_y, 2.0)])
        {'res': 13.445138634774501}
        """

        kwargs = {} if kwargs is None else kwargs

        sim_df = self.model.simulate_parameter(
            parameter_value_pairs,
            self.simulation_options,
            kwargs,
        )

        if isinstance(sim_df, pd.Series):
            sim_df = sim_df.to_frame()
        if not isinstance(sim_df, pd.DataFrame):
            raise TypeError("Model.simulate must return a pandas DataFrame or Series.")

        out: dict[str, float] = {}
        for ind, cfg in self.indicators_config.items():
            if ind not in sim_df.columns:
                raise KeyError(f"Indicator {ind!r} not found in simulation output.")
            series = sim_df[ind]
            if isinstance(cfg, tuple):
                func, ref = cfg
                out[ind] = float(func(series, ref))
            else:
                out[ind] = float(cfg(series))
        return out

    def scipy_obj_function(
        self, x: float | Iterable[float], kwargs: dict | None = None
    ) -> float:
        """
        SciPy-compatible wrapper around :func:`function`.

        Parameters
        ----------
        x : float | Iterable[float]
            Parameter vector (same order as ``self.parameters``),
            or a single float if only one parameter.
        kwargs : dict, optional
            Extra keyword arguments for ``model.simulate``.

        Returns
        -------
        float
            Value of the indicator chosen as ``scipy_obj_indicator``.

        Raises
        ------
        ValueError
            If the length of ``x`` does not match ``parameters``.

        Examples
        --------
        >>> from scipy.optimize import minimize_scalar
        >>> import numpy as np
        >>> from corrai.base.parameter import Parameter
        >>> from corrai.base.objfunctions import ObjectiveFunction
        >>> from tests.resources.pymodels import Ishigami
        >>>
        >>> model = Ishigami()
        >>> param_x = Parameter(name="x", interval=(-np.pi, np.pi), init_value=2.0)

        >>> simulation_options = {
        ...     "start": "2000-01-01",
        ...     "end": "2000-01-01",
        ...     "timestep": "h",
        ... }

        >>> obj_fun = ObjectiveFunction(
        ...     model,
        ...     simulation_options,
        ...     parameters=[param_x],
        ...     indicators_config={"res": np.mean},
        ... )
        >>> res = minimize_scalar(
        ...     obj_fun.scipy_obj_function, bounds=obj_fun.bounds[0], method="bounded"
        ... )
        >>> res.fun
        13.445138634774501
        """
        if isinstance(x, (int, float)):
            x_vec = [float(x)]
        else:
            x_vec = list(x)

        if len(x_vec) != len(self.parameters):
            raise ValueError("Length of x does not match " "number of parameters.")

        parameter_value_pairs = list(zip(self.parameters, x_vec))

        res = self.function(parameter_value_pairs, kwargs)
        return float(res[self.scipy_obj_indicator])


class SciOptimizer:
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
    def values(self):
        if self.model_evaluator.model.is_dynamic:
            return self.model_evaluator.sample.values
        else:
            return self.model_evaluator.sample.get_static_results_as_df()

    @property
    def results(self):
        return self.model_evaluator.sample.results

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

    def diff_evo_minimize(
        self,
        indicators_configs: str
        | tuple[str, str | Callable]
        | tuple[str, str | Callable, pd.Series | None],
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
        indicators_configs : str or tuple
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
            args=(indicators_configs, simulation_options, simulation_kwargs),
            maxiter=maxiter,
            tol=tol,
            rng=rng,
            workers=workers,
        )

    @wraps(Sample.plot_sample)
    def plot_sample(
        self,
        indicator: str | None,
        reference_timeseries: pd.Series | None = None,
        title: str | None = None,
        y_label: str | None = None,
        x_label: str | None = None,
        alpha: float = 0.5,
        show_legends: bool = False,
        round_ndigits: int = 2,
        quantile_band: float = 0.75,
        type_graph: str = "area",
    ) -> go.Figure:
        return self.model_evaluator.sample.plot_sample(
            indicator=indicator,
            reference_timeseries=reference_timeseries,
            title=title,
            y_label=y_label,
            x_label=x_label,
            alpha=alpha,
            show_legends=show_legends,
            round_ndigits=round_ndigits,
            quantile_band=quantile_band,
            type_graph=type_graph,
        )
