from typing import Callable, Sequence, Iterable

import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary

from corrai.base.model import Model
from corrai.base.parameter import Parameter


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
        parameters: list,
        evaluators: Sequence[Callable] | None = None,
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
        ...     objective_ids=["y"]
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

        for block in self.evaluators:
            pairs = [(p, self._cast_value(p, v)) for p, v in zip(self.parameters, x)]
            param_dict = {p.name: v for p, v in pairs}

            if hasattr(block, "function"):
                res = block.function(parameter_value_pairs=pairs)
            else:  # plain function
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
    >>> p1 = Parameter(name="x1", model_property="a", interval=(0,1))
    >>> p2 = Parameter(name="x1", model_property="b", interval=(0,1))
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
        >>> param_x = Parameter(name="x", model_property="x",
        ...                     interval=(-np.pi, np.pi), init_value=2.0)
        >>> param_y = Parameter(name="y", model_property="y",
        ...                     interval=(-np.pi, np.pi), init_value=1.0)
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
        ...     indicators_config={"res": np.mean}
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
        ...     indicators_config={"res": np.mean}
        ... )
        >>> res = minimize_scalar(obj_fun.scipy_obj_function,
        ...                       bounds=obj_fun.bounds[0],
        ...                       method="bounded")
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
