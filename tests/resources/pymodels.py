from corrai.base.model import Model

import pandas as pd
import numpy as np


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
