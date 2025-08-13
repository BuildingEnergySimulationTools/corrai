from typing import Callable, Iterable

import pandas as pd

from corrai.base.model import Model
from corrai.base.parameter import Parameter


class ObjectiveFunction:
    """
    Configure and evaluate an objective function for calibration/optimization.

    This specific method is designed for scipy compatibility.

    Parameters
    ----------
    model : Model
        The model to be calibrated.
    simulation_options : dict
        Options passed to `model.simulate`.
    parameters : list[Parameter]
        Calibration parameters (must have `interval` defined for continuous
        optimization; `ptype` typically "Real" or "Integer").
    indicators_config : dict[str, Callable | tuple[Callable, pd.Series | pd.DataFrame | None]]
        Keys are indicator names (columns in simulation output).
        Values are either:
          - a function f(y) -> float
          - or a tuple (f, ref) where f(y, ref) -> float
        Examples: np.mean, np.sum, sklearn.metrics.mean_squared_error, etc.
    scipy_obj_indicator : str, optional
        The indicator to be used as the scalar objective in `scipy_obj_function`.
        Defaults to the first key of `indicators_config`.

    Attributes
    ----------
    parameters : list[Parameter]
    simulation_options : dict
    indicators_config : dict
    scipy_obj_indicator : str

    Properties
    ----------
    bounds : list[tuple[float, float]]
        Continuous bounds (taken from `Parameter.interval`). Raises if a parameter
        has no `interval`.
    init_values : list[float] | None
        Initial values if all parameters have `init_value` (and are scalar),
        otherwise None.

    Methods
    -------
    function(param_values: dict[str, float] | Iterable[float], kwargs: dict | None = None) -> dict[str, float]
        Run the simulation and compute indicator values.
    scipy_obj_function(x: float | Iterable[float], kwargs: dict | None = None) -> float
        SciPy-compatible objective using `scipy_obj_indicator`.
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

    def _as_vector(self, x: dict[str, float] | Iterable[float]) -> list[float]:
        """Accepts a dict keyed by parameter name OR a positional vector."""
        if isinstance(x, dict):
            try:
                return [float(x[p.name]) for p in self.parameters]
            except KeyError as e:
                raise ValueError(f"Missing value for parameter {e.args[0]!r}") from e
        else:
            vec = list(x)
            if len(vec) != len(self.parameters):
                raise ValueError(
                    "Length of values does not match number of parameters."
                )
            return [float(v) for v in vec]

    def _to_property_dict(self, vec: list[float]) -> dict[str, float]:
        """
        Map parameter values to the model properties.
        If `model_property` is a tuple of paths, apply the same scalar value to each path.
        """
        prop_dict: dict[str, float] = {}
        for val, par in zip(vec, self.parameters):
            mp = par.model_property
            if mp is None:
                prop_dict[par.name] = val
            elif isinstance(mp, tuple):
                for path in mp:
                    prop_dict[path] = val
            else:
                prop_dict[mp] = val
        return prop_dict

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
                return None
            vals.append(float(iv))
        return vals

    def function(
        self,
        param_values: dict[str, float] | Iterable[float],
        kwargs: dict | None = None,
    ) -> dict[str, float]:
        """
        Run the model and compute the configured indicators.

        Parameters
        ----------
        param_values : dict[str, float] | Iterable[float]
            Parameter values either as {parameter_name: value} or a positional vector
            in the same order as `self.parameters`.
        kwargs : dict, optional
            Extra kwargs forwarded to the model.

        Returns
        -------
        dict[str, float]
            Computed indicators: {indicator_name: value}
        """
        kwargs = {} if kwargs is None else kwargs
        vec = self._as_vector(param_values)
        property_dict = self._to_property_dict(vec)

        sim_df = self.model.simulate(
            property_dict=property_dict,
            simulation_options=self.simulation_options,
            simulation_kwargs=kwargs,
        )
        if not isinstance(sim_df, (pd.DataFrame, pd.Series)):
            raise TypeError("Model.simulate must return a pandas DataFrame or Series.")

        if isinstance(sim_df, pd.Series):
            sim_df = sim_df.to_frame()

        out: dict[str, float] = {}
        for ind, cfg in self.indicators_config.items():
            if ind not in sim_df.columns:
                raise KeyError(f"Indicator {ind!r} not found in simulation output.")
            series = sim_df[ind]
            if isinstance(cfg, tuple):
                func, ref = cfg
                out[ind] = float(func(series, ref))
            else:
                func = cfg
                out[ind] = float(func(series))
        return out

    def scipy_obj_function(
        self, x: float | Iterable[float], kwargs: dict | None = None
    ) -> float:
        """
        Wrapper for scipy.optimize that calculates the
        objective function for given parameter values.

        Parameters
        ----------
        x : float | Iterable[float]
            Parameter vector (same order as `self.parameters`) or a single value
            if there is only one parameter.
        kwargs : dict, optional
            Extra kwargs forwarded to the model.

        Returns
        -------
        float
        """
        if isinstance(x, (int, float)):
            x_vec = [float(x)]
        else:
            x_vec = list(x)
        if len(x_vec) != len(self.parameters):
            raise ValueError("Length of x does not match number of parameters.")
        res = self.function(x_vec, kwargs)
        return float(res[self.scipy_obj_indicator])
