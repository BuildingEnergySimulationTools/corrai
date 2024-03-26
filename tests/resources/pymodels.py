from corrai.base.model import Model

import pandas as pd
import numpy as np
from pathlib import Path


class Ishigami(Model):
    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        def equation(x):
            return (
                np.sin(x["x1"])
                + 7.0 * np.power(np.sin(x["x2"]), 2)
                + 0.1 * np.power(x["x3"], 4) * np.sin(x["x1"])
            )

        return pd.DataFrame(
            {"res": [equation(parameter_dict)]},
            index=pd.date_range(
                simulation_options["start"],
                simulation_options["end"],
                freq=simulation_options["timestep"],
            ),
        )

    def save(self, file_path: Path, extension: str = None):
        pass


class VariantModel(Model):
    def __init__(self):
        self.y1 = 1
        self.z1 = 2
        self.multiplier = 1

    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        if parameter_dict is None:
            parameter_dict = {"x1": 1, "x2": 2}

        result = (
            self.y1 * parameter_dict["x1"] + self.z1 * parameter_dict["x2"]
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

    def save(self, file_path: str, extension: str = ".txt"):
        """
        Save the current parameters of the model to a file.

        :param file_path: The file path where the parameters will be saved.
        :param extension: The file extension to use for saving the model.
                          If not provided, it defaults to ".txt".
        """
        with open(f"{file_path}{extension}", "w") as file:
            file.write(f"y1={self.y1}\n")
            file.write(f"z1={self.z1}\n")
            file.write(f"multiplier={self.multiplier}\n")
