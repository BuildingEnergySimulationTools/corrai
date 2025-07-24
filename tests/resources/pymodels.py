from corrai.base.model import Model

import pandas as pd
import numpy as np
from pathlib import Path


class Pymodel(Model):
    def __init__(self):
        self.prop_1 = 1
        self.prop_2 = 2
        self.prop_3 = 3

    def get_property_values(self, property_list: list):
        return [getattr(self, name) for name in property_list]

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
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


class VariantModel(Model):
    def __init__(self):
        self.y1 = 1
        self.z1 = 2
        self.multiplier = 1

    def simulate(
        self,
        property_dict: dict = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        if property_dict is None:
            property_dict = {"x1": 1, "x2": 2}

        result = (
            self.y1 * property_dict["x1"] + self.z1 * property_dict["x2"]
        ) * self.multiplier

        # Create a DataFrame with a single row
        df = pd.DataFrame(
            {"res": [result]},
            index=pd.date_range(
                simulation_options["start"],
                simulation_options["end"],
                freq=simulation_options["timestep"],
            ),
        )

        return df

    def save(self, file_path: Path):
        """
        Save the current parameters of the model to a file.

        :param file_path: The file path where the parameters will be saved.
        """
        with open(f"{file_path}", "w") as file:
            file.write(f"y1={self.y1}\n")
            file.write(f"z1={self.z1}\n")
            file.write(f"multiplier={self.multiplier}\n")
