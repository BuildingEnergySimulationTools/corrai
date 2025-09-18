from typing import Union

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from corrai.base.parameter import Parameter


class Model(ABC):
    """
    Abstract base class for models in Corrai.

    A `Model` defines the interface for simulation-based, analytical,
    or FMU-driven systems. It provides utilities to map high-level
    `Parameter` objects to model-specific properties and to run simulations
    given parameter-value pairs.

    Subclasses must implement the :meth:`simulate` method.

    Parameters
    ----------
    is_dynamic : bool, default=True
        Indicates if the model returns time dependant results as DataFrame with
        DatetimeIndex, or is static and returns a Series of values

    Methods
    -------
    get_property_from_param(parameter_value_pairs)
        Convert (Parameter, value) pairs into a dictionary of model
        property assignments, handling relative and absolute values.
    simulate(property_dict, simulation_options, simulation_kwargs)
        Abstract method. Run the simulation and return a pandas DataFrame.
    simulate_parameter(parameter_value_pairs, simulation_options, simulation_kwargs)
        Helper that combines :meth:`get_property_from_param` and :meth:`simulate`.
    get_property_values(property_list)
        Retrieve current values of given model properties. Must be
        implemented in subclasses if relative parameters are used.
    save(file_path)
         Persists the model state or parameters to disk. Optional.
    """

    def __init__(self, is_dynamic: bool):
        self.is_dynamic = is_dynamic

    def get_property_from_param(
        self,
        parameter_value_pairs: list[tuple[Parameter, str | int | float]],
    ) -> dict[str, int | float | str]:
        """
        Map (Parameter, value) pairs to a dictionary of model properties.

        Handles both absolute and relative parameter definitions. For
        relative parameters, the initial property value is used as a baseline.

        Parameters
        ----------
        parameter_value_pairs : list of (Parameter, int | float | str)
            List of tuples linking a :class:`Parameter` object with the
            value assigned for this simulation.

        Returns
        -------
        property_dict : dict
            Mapping from property names (str) to updated values.
        """
        property_dict = {}
        for param, value in parameter_value_pairs:
            props = (
                param.model_property
                if isinstance(param.model_property, tuple)
                else (param.model_property,)
            )
            if param.relabs == "Relative":
                if param.init_value is None:
                    param.init_value = self.get_property_values(props)

                values = [nom_val * value for nom_val in param.init_value]
            else:
                values = [value] * len(props)
            for prop, val in zip(props, values):
                property_dict[prop] = val
        return property_dict

    @abstractmethod
    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame | pd.Series:
        """
        Run a simulation for given properties and options.

        Must be implemented in subclasses.

        Parameters
        ----------
        property_dict : dict, optional
            Mapping from model property names to values to override.
        simulation_options : dict, optional
            Options controlling the simulation (e.g., start/end times,
            timestep, solver parameters).
        simulation_kwargs : dict, optional
            Extra keyword arguments for the simulation routine.

        Returns
        -------
        pd.DataFrame
            Simulation results as a DataFrame with a DateTimeIndex and one
            or more output columns.
        """
        pass

    def simulate_parameter(
        self,
        parameter_value_pairs: list[tuple[Parameter, str | int | float]],
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Simulate the model given a set of parameter-value pairs.

        This combines :meth:`get_property_from_param` and :meth:`simulate`.

        Parameters
        ----------
        parameter_value_pairs : list of (Parameter, int | float | str)
            The parameters and their assigned values for this run.
        simulation_options : dict, optional
            Options passed to the simulation routine.
        simulation_kwargs : dict, optional
            Additional arguments for the simulation routine.

        Returns
        -------
        pd.DataFrame
            Simulation results as a DataFrame with a DateTimeIndex and one
            or more output columns.
        """
        return self.simulate(
            self.get_property_from_param(parameter_value_pairs),
            simulation_options,
            **{} if simulation_kwargs is None else simulation_kwargs,
        )

    def get_property_values(self, property_list: list[str]) -> list[str | int | float]:
        """
        Retrieve current values of given properties from the model.

        Must be implemented in subclasses if relative parameters are used.

        Parameters
        ----------
        property_list : tuple of str
            Names of model properties.

        Returns
        -------
        list of int | float | str
            Current values of the requested properties.

        Raises
        ------
        NotImplementedError
            If not overridden in a subclass.
        """
        raise NotImplementedError(
            "No get_property_values method was defined for this model."
            "If you use Relative values for parameters, consider switching to absolute,"
            " or specify the init values for the properties in the parameters"
        )

    def save(self, file_path: Path):
        """
        Save model state or parameters to disk.

        Parameters
        ----------
        file_path : Path
            Destination path.

        Raises
        ------
        NotImplementedError
            If not overridden in a subclass.
        """
        raise NotImplementedError("No save method was defined for this model")


class PyModel(Model, ABC):
    def __init__(self, is_dynamic: bool):
        super().__init__(is_dynamic)

    def get_property_values(self, property_list: list):
        return [getattr(self, name) for name in property_list]

    def set_property_values(self, property_dict: dict):
        for prop, val in property_dict.items():
            setattr(self, prop, val)


class IshigamiDynamic(PyModel):
    """
    Example implementation of the Ishigami function.

    The Ishigami function is a standard benchmark for sensitivity analysis:
        f(x) = sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1)

    Attributes
    ----------
    x1, x2, x3 : float
        Model parameters controlling the output.

    Methods
    -------
    get_property_values(property_list)
        Retrieve current values of x1, x2, x3.
    set_property_values(property_dict)
        Set properties from a dictionary.
    simulate(property_dict, simulation_options, simulation_kwargs)
        Evaluate the Ishigami function and return as a time series DataFrame.
    """

    def __init__(self):
        super().__init__(is_dynamic=True)
        self.x1 = 1
        self.x2 = 2
        self.x3 = 3

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        if property_dict is not None:
            self.set_property_values(property_dict)

        res = (
            np.sin(self.x1)
            + 7.0 * np.power(np.sin(self.x2), 2)
            + 0.1 * np.power(self.x3, 4) * np.sin(self.x1)
        )

        return pd.DataFrame(
            {"res": [res]},
            index=pd.date_range(
                simulation_options["start"],
                simulation_options["end"],
                freq=simulation_options["timestep"],
            ),
        )


class Ishigami(PyModel):
    """
    Example implementation of the Ishigami function.

    The Ishigami function is a standard benchmark for sensitivity analysis:
        f(x) = sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1)

    Attributes
    ----------
    x1, x2, x3 : float
        Model parameters controlling the output.

    Methods
    -------
    get_property_values(property_list)
        Retrieve current values of x1, x2, x3.
    set_property_values(property_dict)
        Set properties from a dictionary.
    simulate(property_dict, simulation_options, simulation_kwargs)
        Evaluate the Ishigami function and return as a Series.
    """

    def __init__(self):
        super().__init__(is_dynamic=False)
        self.x1 = 1
        self.x2 = 2
        self.x3 = 3

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.Series:
        if property_dict is not None:
            self.set_property_values(property_dict)

        res = (
            np.sin(self.x1)
            + 7.0 * np.power(np.sin(self.x2), 2)
            + 0.1 * np.power(self.x3, 4) * np.sin(self.x1)
        )

        return pd.Series({"res": res})


class PymodelDynamic(PyModel):
    def __init__(self):
        super().__init__(is_dynamic=True)
        self.prop_1 = 1
        self.prop_2 = 2
        self.prop_3 = 3

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.DataFrame:
        if property_dict is not None:
            for prop, val in property_dict.items():
                setattr(self, prop, val)

        return pd.DataFrame(
            {"res": [self.prop_1 * self.prop_2 + self.prop_3]},
            index=pd.date_range(
                simulation_options["start"],
                simulation_options["end"],
                freq=simulation_options["timestep"],
            ),
        )


class PymodelStatic(PyModel):
    def __init__(self):
        super().__init__(is_dynamic=False)
        self.prop_1 = 1
        self.prop_2 = 2
        self.prop_3 = 3

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        **simulation_kwargs,
    ) -> pd.Series:
        if property_dict is not None:
            for prop, val in property_dict.items():
                setattr(self, prop, val)

        return pd.Series({"res": self.prop_1 * self.prop_2 + self.prop_3})
