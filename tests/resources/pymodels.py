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
