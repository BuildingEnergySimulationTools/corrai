import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path


class Model(ABC):
    @abstractmethod
    def simulate(
        self,
        parameter_dict: dict = None,
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        """
        Run simulation for given parameter_dict and simulation options.
        Return simulation results in the form of a Pandas DataFrame with
        DateTime index.
        """
        pass

    @abstractmethod
    def save(self, file_path: Path):
        """
        Save the current parameters of the model to a file.

        :param file_path: The file path where the parameters will be saved.
        """
        pass
