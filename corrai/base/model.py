import pandas as pd
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        """
        Run simulation for given parameter_dict and simulation options.
        Return simulation results in the form of a Pandas DataFrame with
        DateTime index.
        """
        pass
