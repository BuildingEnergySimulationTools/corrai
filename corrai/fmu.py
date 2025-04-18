import datetime as dt
import shutil
import tempfile
from pathlib import Path

import fmpy
from fmpy import simulate_fmu
import pandas as pd
from sklearn.pipeline import Pipeline

from corrai.base.model import Model


def seconds_index_to_datetime_index(
    index_second: pd.Index, ref_year: int
) -> pd.DatetimeIndex:
    since = dt.datetime(ref_year, 1, 1, tzinfo=dt.timezone.utc)
    diff_seconds = index_second + since.timestamp()
    return pd.DatetimeIndex(pd.to_datetime(diff_seconds, unit="s"))


def datetime_to_second(datetime_in: dt.datetime | pd.Timestamp):
    year = datetime_in.year
    origin = dt.datetime(year, 1, 1)
    return (datetime_in - origin).total_seconds()


def datetime_index_to_seconds_index(index_datetime: pd.DatetimeIndex) -> pd.Index:
    time_start = dt.datetime(index_datetime[0].year, 1, 1, tzinfo=dt.timezone.utc)
    new_index = index_datetime.to_frame().diff().squeeze()
    new_index.iloc[0] = dt.timedelta(
        seconds=index_datetime[0].timestamp() - time_start.timestamp()
    )
    sec_dt = [elmt.total_seconds() for elmt in new_index]
    return pd.Series(sec_dt).cumsum()


def df_to_combitimetable(df: pd.DataFrame, filename):
    """
    Write a text file compatible with modelica Combitimetables object from a
    Pandas DataFrame with a DatetimeIndex. DataFrames with non monotonically increasing
    datetime index will raise a ValueError to prevent bugs
    when file is used in Modelica.
    @param df: DataFrame with DatetimeIndex
    @param filename: string or Path to the output file
    @return: None
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "df DateTimeIndex is not monotonically increasing, this will"
            "cause Modelica to crash."
        )

    df = df.copy()
    with open(filename, "w") as file:
        file.write("#1 \n")
        line = ""
        line += f"double table1({df.shape[0]}, {df.shape[1] + 1})\n"
        line += "\t# Time (s)"
        for i, col in enumerate(df.columns):
            line += f"\t({i + 1}){col}"
        file.write(f"{line} \n")

        if isinstance(df.index, pd.DatetimeIndex):
            df.index = datetime_index_to_seconds_index(df.index)

        file.write(df.to_csv(header=False, sep="\t", lineterminator="\n"))


def get_start_stop_year_tz_from_x(x: pd.DataFrame = None):
    if x is None:
        return None, None, None
    if isinstance(x.index, pd.DatetimeIndex):
        idx = datetime_index_to_seconds_index(x.index)
        year = x.index[0].year
        tz = x.index.tz
    else:
        idx = x.index
        year = None
        tz = None
    return idx.min(), idx.max(), year, tz


class ModelicaFmuModel(Model):
    """
    A class used to wrap an FMU (Functional Mock-up Unit) into the corrai
    Model class formalism.

    Attributes:
        fmu_path (Path): Path to the FMU file.

        simulation_options (dict[str, float | str | int]): Simulation options for
        the FMU. It may include the following keys : startTime, stopTime, stepSize,
        solver, outputInterval, tolerance, fmi_type.

        x (pd.DataFrame): Input boundary data. Will be passed to the model using
        a Combitimetable.

        x_combitimetable_name (to be defines in the modelica model). with a
        txt file.

        output_list (list[str]): List of variables to output from the simulation.

        simulation_dir (Path): Directory for simulation files. will create a temp dir
        if not given

        parameters (dict): Dictionary to store simulation parameters.

        _begin_year (int): The year the simulation starts, extracted from input data.
        Used to retrieve datetime after the simulation

        _x (pd.DataFrame): Stored x data. Is used to prevent unecessary reset of data in
        the txt file.

        _set_x(df: pd.DataFrame):

        _set_x_sim_options(

        _set_simuopt_start_stop_from_x(x: pd.DataFrame):
            Sets the start and stop times for the simulation based on the input data.

        save(file_path: Path):
            Saves the FMU file to the specified location.

        __repr__():
            Returns a string representation of the FMU model, including its parameters.
    """

    def __init__(
        self,
        fmu_path: Path,
        simulation_options: dict[str, float | str | int] = None,
        x: pd.DataFrame = None,
        output_list: list[str] = None,
        x_combitimetable_name: str = None,
        simulation_dir: Path = None,
    ):
        self._x = pd.DataFrame()
        self.simulation_options = {
            "startTime": 0,
            "stopTime": 24 * 3600,
            "stepSize": 60,
            "solver": "CVode",
            "outputInterval": 1,
            "tolerance": 1e-6,
            "fmi_type": "ModelExchange",
        }
        self.model_path = fmu_path
        self.simulation_dir = (
            Path(tempfile.mkdtemp()) if simulation_dir is None else simulation_dir
        )
        self.output_list = output_list
        self.parameters = {}
        self._begin_year = None
        self._tz = None
        self.x_combitimetable_name = (
            x_combitimetable_name if x_combitimetable_name is not None else "Boundaries"
        )
        self._set_x_sim_options(x, simulation_options)

    def set_x(self, df: pd.DataFrame):
        """Sets the input data for the simulation and updates the corresponding file."""

        if not self._x.equals(df):
            new_bounds_path = self.simulation_dir / "boundaries.txt"
            df_to_combitimetable(df, new_bounds_path)
            self.parameters[f"{self.x_combitimetable_name}.fileName"] = (
                new_bounds_path.as_posix()
            )
            self._x = df

            start, stop, year, tz = get_start_stop_year_tz_from_x(df)
            self.simulation_options["startTime"] = start
            self.simulation_options["stopTime"] = stop
            self._begin_year = year
            self._tz = tz

    def _set_x_sim_options(
        self,
        x: pd.DataFrame = None,
        simulation_options: dict[
            str, float | str | int | dt.datetime | pd.Timestamp
        ] = None,
    ):
        """
        Sets the input data and simulation options, updates start and stop times if
        necessary.
        If x is specified, it will set simulationStart and simulationStop based
        on x.index min() and max().  Whether it is datetime or integer (seconds).
        If simulationStart and simulationStop are specified in the simulation_options,
        it will NOT BE TAKEN INTO ACCOUNT. The rest of the simulation_options WILL BE
        written

        If x is not specified, simulationStart and simulationStop will be considered

        :param x (pd.DataFrame): The Dataframe describing the boundary conditions
        :param simulation_options (dict): Simulation options for  the FMU. It may
        include the following keys : startTime, stopTime, stepSize, solver,
        outputInterval, tolerance, fmi_type
        """

        if x is not None:
            self.set_x(x)

        # Get simu options
        if simulation_options is not None:
            # Update all but time
            to_update = {
                key: val
                for key, val in simulation_options.items()
                if key not in ["startTime", "stopTime"]
            }
            self.simulation_options.update(to_update)

            if x is None:
                simo = {}
                for key in ["startTime", "stopTime"]:
                    if key in simulation_options:
                        if isinstance(
                            simulation_options[key], (dt.datetime, pd.Timestamp)
                        ):
                            simo[key] = datetime_to_second(simulation_options[key])
                            if key == "startTime":
                                self._begin_year = simulation_options["startTime"].year
                        else:
                            simo[key] = simulation_options[key]
                    else:
                        simo[key] = self.simulation_options[key]

                self.simulation_options["startTime"] = simo["startTime"]
                self.simulation_options["stopTime"] = simo["stopTime"]

    def simulate(
        self,
        parameter_dict: dict[str, float | int | str] = None,
        simulation_options: dict = None,
        x: pd.DataFrame = None,
        solver_duplicated_keep: str = "last",
        post_process_pipeline: Pipeline = None,
        debug_param: bool = False,
        debug_logging: bool = False,
        logger=None,
    ) -> pd.DataFrame:
        """
        Run FMU simulation for the given parameters and simulation_options.

        :param debug_param: if True, print parameter_dict
        :param solver_duplicated_keep: Some solver will return duplicated index,
        choose the one you want to keep (ex. "last", "first")
        :param x: Input boundary data. Will be passed to the model using a
        Combitimetable
        :param simulation_options: Simulation options for the FMU.
        It may include the following keys : startTime, stopTime, stepSize, solver,
        outputInterval, tolerance, fmi_type.
        :param parameter_dict: Dictionary of parameters values
        :param debug_logging:
        :param logger:
        :return: PandasDataFrame
        """

        if debug_param:
            print(parameter_dict)

        self.parameters.update(parameter_dict or {})
        self._set_x_sim_options(x, simulation_options)

        result = simulate_fmu(
            filename=self.model_path,
            start_time=self.simulation_options["startTime"],
            stop_time=self.simulation_options["stopTime"],
            step_size=self.simulation_options["stepSize"],
            relative_tolerance=self.simulation_options["tolerance"],
            start_values=self.parameters,
            output=self.output_list,
            solver=self.simulation_options["solver"],
            output_interval=self.simulation_options["outputInterval"],
            fmi_type=self.simulation_options["fmi_type"],
            debug_logging=debug_logging,
            logger=logger,
        )

        df = pd.DataFrame(result, columns=["time"] + self.output_list)

        if self._begin_year is not None:
            df.index = seconds_index_to_datetime_index(df["time"], self._begin_year)
            # Weird solver behavior
            df.index = df.index.round("s")
            df = df.tz_localize(self._tz)
            df.index.freq = df.index.inferred_freq
        else:
            # solver can do funny things. Round time
            df.index = round(df["time"], 2)

        df.drop(columns=["time"], inplace=True)

        # First values are often duplicates...
        # For some reason, it appears that values of first timestep is often off
        # A bit dirty, so we let you choose

        df = df.loc[~df.index.duplicated(keep=solver_duplicated_keep)]

        # If all previous operations failed to provide good results
        # You can use you own correction pipeline.
        # Be sure to use pd.DatetimeIndex if you want to resample
        if post_process_pipeline is not None:
            df = post_process_pipeline.fit_transform(df)

        return df

    def save(self, file_path: Path):
        """
        Save the FMU file to the specified location.

        Parameters:
            file_path (Path): The path where the FMU file will be saved.
        """
        shutil.copyfile(self.model_path, file_path)

    def __repr__(self):
        model_description = fmpy.read_model_description(self.model_path.as_posix())

        model_info = f"Model Name: {model_description.modelName}\n"
        model_info += (
            f"Description: {fmpy.read_model_description(self.model_path.as_posix())}\n"
        )
        model_info += f"Version: {model_description.fmiVersion}\n"
        model_info += "Parameters:\n"

        parameters = [
            var
            for var in model_description.modelVariables
            if var.causality == "parameter"
        ]
        if not parameters:
            model_info += "  No parameters available.\n"
        else:
            for param in parameters:
                default_value = (
                    param.start if param.start is not None else "Not specified"
                )
                desc = (
                    param.description
                    if param.description
                    else "No description available."
                )
                model_info += (
                    f"  Name: {param.name}, "
                    f"Default Value: {default_value}, "
                    f"Description: {desc}\n"
                )

        return model_info
