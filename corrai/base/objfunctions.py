from typing import Callable

import pandas as pd

from corrai.base.model import Model
from corrai.base.parameter import Parameter


class ObjectiveFunction:
    """
    A class to represent configure an objective function for model calibration
    and optimization.
    A specific method is designed for scipy compatibility.

    Parameters
    ----------
    model : Model
        The model to be calibrated.
    simulation_options : dict
        Dictionary containing simulation options.
    param_list : list of dict
        List of dictionaries specifying the parameters to be calibrated,
        including bounds and initial values.
    indicators_config : dict[str, tuple[Callable, pd.Series | pd.DataFrame | None] | Callable]
        Dictionary where keys are indicator names corresponding to Model simulation
         output names, and values are either:
        - A aggregation function to compute the indicator (ex: np.mean, np.sum).
        - A tuple (function, reference data) if comparison when reference is needed.
        (ex. sklearn.metrics.mean_squared_error, corrai.metrics.mae)
    scipy_obj_indicator : str, optional
        The indicator to be used as the objective function for scipy optimization.
        Defaults to the first key in `indicators_config`.

    Attributes
    ----------
    model : Model
        The simulation model being calibrated.
    simulation_options : dict
        Options for running the simulation.
    param_list : list of dict
        List of parameter dictionaries, each containing:
        - `Parameter.NAME`: Name of the parameter.
        - `Parameter.INTERVAL`: Bounds for the parameter.
        - `Parameter.INIT_VALUE`: Initial value (optional).
    indicators_config : dict
        Dictionary mapping indicator names to their computation functions and optional reference data.
    scipy_obj_indicator : str
        The indicator used as the objective function in `scipy.optimize`.

    Properties
    ----------
    bounds : list[tuple[float, float]]
        List of parameter bounds extracted from `param_list`.
    init_values : list[float] or None
        List of initial values for parameters if provided; otherwise, None.

    Methods
    -------
    function(param_dict, kwargs: dict = None) -> dict[str, float]
        Runs the model simulation and computes indicator values.

        Parameters
        ----------
        param_dict : dict
            Dictionary containing parameter names and their values.
        kwargs : dict, optional
            Additional arguments for the model simulation.

        Returns
        -------
        dict[str, float]
            Dictionary containing computed indicator values.

    scipy_obj_function(x, kwargs: dict = None) -> float
        Computes the objective function value for scipy optimization.

        Parameters
        ----------
        x : float or list of float
            List of parameter values or a single parameter value.
        kwargs : dict, optional
            Additional arguments for the model simulation.

        Returns
        -------
        float
            The calculated value of the objective function based on `scipy_obj_indicator`.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> model = SomeModel()
    >>> simulation_options = {"duration": 100}
    >>> param_list = [{Parameter.NAME: "param1", Parameter.INTERVAL: [0, 1], Parameter.INIT_VALUE: 0.5}]
    >>> indicators_config = {"res": (mean_squared_error, reference_series)}
    >>> obj_func = ObjectiveFunction(model, simulation_options, param_list, indicators_config)
    >>> obj_func.function({"param1": 0.8})
    {'indicator1': 0.123}
    >>> obj_func.scipy_obj_function([0.8])
    0.123
    """

    def __init__(
        self,
        model: Model,
        simulation_options: dict,
        param_list: list[dict],
        indicators_config: dict[
            str, tuple[Callable, pd.Series | pd.DataFrame | None] | Callable
        ],
        scipy_obj_indicator: str = None,
    ):
        self.model = model
        self.param_list = param_list
        self.indicators_config = indicators_config
        self.simulation_options = simulation_options
        self.scipy_obj_indicator = (
            list(indicators_config.keys())[0]
            if scipy_obj_indicator is None
            else scipy_obj_indicator
        )

    @property
    def bounds(self):
        return [param[Parameter.INTERVAL] for param in self.param_list]

    @property
    def init_values(self):
        if all(Parameter.INIT_VALUE in param for param in self.param_list):
            return [param[Parameter.INIT_VALUE] for param in self.param_list]
        else:
            return None

    def function(self, param_dict, kwargs: dict = None):
        """
        Calculate the objective function for given parameter values.

        Parameters
        ----------
        x_dict : dict
            Dictionary containing parameter names and their values.

        Returns
        -------
        pd.Series
            A series containing the calculated values for each indicator.
        """
        kwargs = {} if kwargs is None else kwargs
        res = self.model.simulate(param_dict, self.simulation_options, **kwargs)

        function_results = {}
        for ind, (func, ref) in self.indicators_config.items():
            function_results[ind] = (
                func(res[ind]) if ref is None else func(res[ind], ref)
            )
        return function_results

    def scipy_obj_function(self, x, kwargs: dict = None):
        """
        Wrapper for scipy.optimize that calculates the
        objective function for given parameter values.

        Parameters
        ----------
        x : float or list of float
            List of parameter values or a single parameter value.

        Returns
        -------
        float
            The calculated value of the objective function.
        """
        x = [x] if isinstance(x, (float, int)) else x
        if len(x) != len(self.param_list):
            raise ValueError("Length of x does not match length of param_list")

        param_dict = {
            self.param_list[i][Parameter.NAME]: x[i]
            for i in range(len(self.param_list))
        }
        result = self.function(param_dict, kwargs)
        return result[self.scipy_obj_indicator]
