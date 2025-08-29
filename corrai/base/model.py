import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from corrai.base.parameter import Parameter


class Model(ABC):
    def get_property_from_param(
        self,
        parameter_value_pairs: list[tuple[Parameter, str | int | float]],
    ) -> dict[str, int | float | str]:
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
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        """
        Run simulation for given parameter_dict and simulation options.
        Return simulation results in the form of a Pandas DataFrame with
        DateTime index.
        """
        pass

    def simulate_parameter(
        self,
        parameter_value_pairs: list[tuple[Parameter, str | int | float]],
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        return self.simulate(
            self.get_property_from_param(parameter_value_pairs),
            simulation_options,
            simulation_kwargs,
        )

    def get_property_values(
        self, property_list: list[str]
    ) -> list[str | int | float]:
        raise NotImplementedError(
            "No get_property_values method was defined for this model."
            "If you use Relative values for parameters, consider switching to absolute,"
            " or specify the init values for the properties in the parameters"
        )

    def save(self, file_path: Path):
        """
        Save the current parameters of the model to a file.

        :param file_path: The file path where the parameters will be saved.
        """
        raise NotImplementedError("No save method was defined for this model")


class Ishigami(Model):
    def __init__(self):
        self.x1 = 1
        self.x2 = 2
        self.x3 = 3

    def get_property_values(self, property_list: list):
        return [getattr(self, name) for name in property_list]

    def set_property_values(self, property_dict: dict):
        for prop, val in property_dict.items():
            setattr(self, prop, val)

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
