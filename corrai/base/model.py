import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from corrai.base.parameter import Parameter


class Model(ABC):
    def get_property_from_param(
        self, parameter_dict: dict[Parameter, str | float | int]
    ) -> dict[str, int | float | str]:
        property_dict = {}
        for param, value in parameter_dict.items():
            props = (
                param.model_property
                if isinstance(param.model_property, tuple)
                else [param.model_property]
            )
            if param.relabs == "Relative":
                props_nominal_values = self.get_property_values(props)
                values = [nom_val * value for nom_val in props_nominal_values]
            else:
                values = [value] * len(props)
            for prop, val in zip(props, values):
                property_dict[prop] = val
        return property_dict

    @abstractmethod
    def simulate(
        self,
        property_dict: dict = None,
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
        parameter_dict: dict[Parameter, str|int|float] = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        return self.simulate(
            self.get_property_from_param(parameter_dict),
            simulation_options,
            simulation_kwargs
        )

    def get_property_values(self, property_list: list):
        raise NotImplementedError(
            "No get_property_values method was defined for this model."
            "If you use Relative values for parameters, consider switching to absolute"
        )

    def save(self, file_path: Path):
        """
        Save the current parameters of the model to a file.

        :param file_path: The file path where the parameters will be saved.
        """
        raise NotImplementedError("No save method was defined for this model")
