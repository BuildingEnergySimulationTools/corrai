from fmpy import simulate_fmu
import tempfile
import warnings

from pathlib import Path
import pandas as pd
import datetime as dt
from modelitool.combitabconvert import seconds_to_datetime, df_to_combitimetable

from corrai.base.model import Model


class FMUModel(Model):
    def __init__(
        self,
        model_path,
        simulation_options,
        output_list,
        init_parameters=None,
        boundary_df=None,
        year=None,
    ):
        self.model_path = (
            Path(model_path) if isinstance(model_path, str) else model_path
        )
        self._simulation_path = Path(tempfile.mkdtemp())

        self.init_parameters = init_parameters or {}

        self.simulation_options = simulation_options

        self.output_list = output_list

        if boundary_df is not None:
            self.set_boundaries_df(boundary_df)
            if year is not None:
                warnings.warn(
                    "Simulator year is read from boundary"
                    "DAtaFrame. Argument year is ignored"
                )
        elif year is not None:
            self.year = year
        else:
            self.year = dt.date.today().year

    def set_simulation_options(self, simulation_options):
        self.simulation_options = simulation_options

    def set_boundaries_df(self, df):
        new_bounds_path = self._simulation_path / "boundaries.txt"
        print(new_bounds_path)
        df_to_combitimetable(df, new_bounds_path)

        if self.init_parameters is None:
            self.init_parameters = {}
        self.init_parameters["Boundaries.fileName"] = str(new_bounds_path)

        print(self.init_parameters)

        try:
            self.year = df.index[0].year
        except ValueError:
            raise ValueError(
                "Could not read date from boundary condition. "
                "Please verify that DataFrame index is a datetime."
            )

    def set_param_dict(self, param_dict):
        if self.init_parameters is None:
            self.init_parameters = {}
        self.init_parameters.update(param_dict)

    def simulate(
        self,
        parameter_dict: dict = None,
        simulation_options: dict = None,
        debug_logging=False,
        logger=None,
    ) -> pd.DataFrame:
        if parameter_dict:
            self.set_param_dict(parameter_dict)
        if simulation_options:
            self.set_simulation_options(simulation_options)

        start_time = self.simulation_options.get("startTime", 0)

        result = simulate_fmu(
            filename=self.model_path,
            start_time=self.simulation_options.get("startTime", 0),
            stop_time=self.simulation_options.get("stopTime", 1e6),
            step_size=self.simulation_options.get("stepSize", 3600),
            start_values=self.init_parameters,
            output=self.output_list,
            solver=self.simulation_options.get("solver", "CVode"),
            output_interval=self.simulation_options.get("outputInterval", 3600),
            fmi_type=self.simulation_options.get("fmi_type", "ModelExchange"),
            debug_logging=debug_logging,
            logger=logger,
        )

        df = pd.DataFrame(result, columns=["time"] + self.output_list)
        adjusted_time = df["time"] + start_time
        df.index = seconds_to_datetime(adjusted_time, self.year)

        return df
