import datetime as dt
import shutil
import tempfile
import warnings
from pathlib import Path
from contextlib import contextmanager

import fmpy
from fmpy import simulate_fmu
import pandas as pd
from sklearn.pipeline import Pipeline

from corrai.base.model import Model

DEFAULT_SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 24 * 3600,
    "stepSize": 60,
    "solver": "CVode",
    "tolerance": 1e-6,
    "fmi_type": "ModelExchange",
}


@contextmanager
def simulation_workspace(fmu_path: Path, boundary_path: Path | None):
    """
    Create an isolated temporary workspace with a copy of the FMU and optional
    boundary file. Cleans up everything automatically at exit.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        local_fmu = tmpdir / fmu_path.name
        shutil.copy(fmu_path, local_fmu)

        local_boundary = None
        if boundary_path is not None:
            local_boundary = tmpdir / boundary_path.name
            shutil.copy(boundary_path, local_boundary)

        yield local_fmu, local_boundary


def parse_simulation_times(start, stop, step, output_int):
    if all(isinstance(elmt, int) for elmt in (start, stop, step, output_int)):
        return start, stop, step, output_int

    elif (
        isinstance(start, (pd.Timestamp, dt.datetime))
        and isinstance(stop, (pd.Timestamp, dt.datetime))
        and isinstance(step, (pd.Timedelta, dt.timedelta))
        and isinstance(output_int, (pd.Timedelta, dt.timedelta))
    ):
        # Handle 2 years with datetime_index_to_seconds_index function
        start_s, stop_s = datetime_index_to_seconds_index(
            pd.date_range(start, stop, periods=2)
        ).astype(int)
        step_s, output_int_s = map(datetime_to_second, (step, output_int))
        return start_s, stop_s, step_s, output_int_s

    raise ValueError("Invalid 'startTime', 'stopTime', 'stepSize', or 'outputInterval")


def seconds_index_to_datetime_index(
    index_second: pd.Index, ref_year: int
) -> pd.DatetimeIndex:
    """
    Convert an index of seconds into a pandas DatetimeIndex.

    Parameters
    ----------
    index_second : pd.Index
        Index representing time in seconds since January 1st of `ref_year`.
    ref_year : int
        The reference year used to compute the origin (January 1st at 00:00).

    Returns
    -------
    pd.DatetimeIndex
        A naive datetime index corresponding to the seconds offset from the reference year.

    Examples
    --------
    >>> import pandas as pd
    >>> seconds = pd.Index([0, 3600, 7200])
    >>> seconds_index_to_datetime_index(seconds, 2020)
    DatetimeIndex(['2020-01-01 00:00:00',
                   '2020-01-01 01:00:00',
                   '2020-01-01 02:00:00'],
                  dtype='datetime64[ns]', freq=None)
    """
    since = dt.datetime(ref_year, 1, 1, tzinfo=dt.timezone.utc)
    diff_seconds = index_second + since.timestamp()
    return pd.DatetimeIndex(pd.to_datetime(diff_seconds, unit="s"))


def datetime_to_second(datetime_in: dt.datetime | pd.Timestamp | pd.Timedelta):
    """
    Convert a datetime or timestamp into the number of seconds since the beginning of its year.

    Parameters
    ----------
    datetime_in : datetime.datetime or pd.Timestamp
        The datetime object to convert.

    Returns
    -------
    float
        Seconds elapsed since January 1st of the same year as `datetime_in`.

    Examples
    --------
    >>> import datetime as dt
    >>> datetime_to_second(dt.datetime(2020, 1, 1, 1, 0, 0))
    3600.0
    """
    if isinstance(datetime_in, (dt.datetime | pd.Timestamp)):
        year = datetime_in.year
        origin = dt.datetime(year, 1, 1, tzinfo=datetime_in.tz)
        return int((datetime_in - origin).total_seconds())
    return int(datetime_in.total_seconds())


def datetime_index_to_seconds_index(index_datetime: pd.DatetimeIndex) -> pd.Index:
    """
    Convert a DatetimeIndex into a cumulative seconds index starting from January 1st.

    Parameters
    ----------
    index_datetime : pd.DatetimeIndex
        The datetime index to convert.

    Returns
    -------
    pd.Index
        An index of seconds relative to the start of the year.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2020-01-01 00:00:00", periods=3, freq="H")
    >>> datetime_index_to_seconds_index(idx)
    0       0.0
    1    3600.0
    2    7200.0
    dtype: float64
    """
    time_start = dt.datetime(index_datetime[0].year, 1, 1, tzinfo=dt.timezone.utc)
    new_index = index_datetime.to_frame().diff().squeeze()
    new_index.iloc[0] = dt.timedelta(
        seconds=index_datetime[0].timestamp() - time_start.timestamp()
    )
    sec_dt = [elmt.total_seconds() for elmt in new_index]
    return pd.Series(sec_dt).cumsum()


def df_to_combitimetable(df: pd.DataFrame, filename):
    """
    Export a pandas DataFrame with a DatetimeIndex into a Modelica-compatible
    CombiTimeTable text file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing boundary conditions with a DatetimeIndex or seconds index.
    filename : str or Path
        Path to the output file.

    Raises
    ------
    ValueError
        If the datetime index is not monotonically increasing.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"val": [1, 2]}, index=pd.date_range("2020-01-01", periods=2, freq="H")
    ... )
    >>> df_to_combitimetable(df, "boundaries.txt")
    # produces a text file compatible with Modelica
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


class ModelicaFmuModel(Model):
    """
    Wrapper for a Modelica FMU (Functional Mock-up Unit) in the corrai ``Model``
    formalism.

    Provides functionality to:
    - Load an FMU and its metadata.
    - Query property initial values.
    - Run simulations with configurable options.
    - Handle boundary conditions using a CombiTimeTable if defined.

    Parameters
    ----------
    fmu_path : Path or str
        Path to the FMU file.
    simulation_dir : Path, optional
        Directory for simulation files. A temporary directory is created if not
        provided.
    output_list : list of str, optional
        Names of FMU variables to record during simulation.
    boundary_table_name : str or None, optional
        Name of the CombiTimeTable object in the FMU used for boundary conditions.
        If provided, boundary data can be passed through
        ``simulation_options["boundary"]`` or ``property_dict["boundary"]``. If
        ``None`` (default), boundaries are ignored. Boundaries specified in
        ``property_dict`` will always override ``simulation_options`` boundaries


    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.fmu import ModelicaFmuModel

    >>> simu = ModelicaFmuModel(
    ...     fmu_path=fmu_path,
    ...     output_list=["Boundaries.y[1]", "Boundaries.y[2]"],
    ...     boundary_table_name="Boundaries",
    ... )

    >>> new_bounds = pd.DataFrame(
    ...     {"Boundaries.y[1]": [1, 2, 3], "Boundaries.y[2]": [3, 4, 5]},
    ...     index=range(3, 6),
    ... )

    >>> res = simu.simulate(
    ...     simulation_options={
    ...         "solver": "CVode",
    ...         "startTime": 3,
    ...         "stopTime": 5,
    ...         "stepSize": 1,
    ...         "boundary": new_bounds,
    ...     },
    ...     solver_duplicated_keep="last",
    ... )

          Boundaries.y[1]  Boundaries.y[2]
    time
    3.0               1.0              3.0
    4.0               2.0              4.0
    5.0               3.0              5.0
    """

    def __init__(
        self,
        fmu_path: Path | str,
        simulation_dir: Path = None,
        output_list: list[str] = None,
        boundary_table_name: str | None = None,
        default_properties: dict[str, float | int | str] = None,
    ):
        super().__init__(is_dynamic=True)
        fmu_path = Path(fmu_path) if isinstance(fmu_path, str) else fmu_path
        if not fmu_path.exists() or not fmu_path.is_file():
            raise FileNotFoundError(f"FMU file not found at {fmu_path}")

        self.fmu_path = fmu_path
        self.simulation_dir = (
            Path(tempfile.mkdtemp()) if simulation_dir is None else simulation_dir
        )
        self.output_list = output_list
        self.boundary_table_name = boundary_table_name
        self.boundary_file_path = None
        if self.boundary_table_name is not None:
            model_description = fmpy.read_model_description(self.fmu_path.as_posix())
            var_map = {var.name: var.start for var in model_description.modelVariables}
            try:
                combi_tab_property_name = f"{self.boundary_table_name}.fileName"
                self.boundary_file_path = Path(rf"{var_map[combi_tab_property_name]}")
            except KeyError:
                warnings.warn(
                    f"Boundary combitimetable '{self.boundary_table_name}' "
                    f"not found in FMU -> ignoring boundary.",
                    UserWarning,
                    stacklevel=2,
                )
        self.default_properties = default_properties or {}

    def get_property_values(
        self, property_list: str | tuple[str, ...] | list[str]
    ) -> list[str | int | float | None]:
        """
        Retrieve initial values of FMU properties.

        Parameters
        ----------
        property_list : str or tuple of str or list of str
            Name(s) of FMU properties to query.

        Returns
        -------
        list
            List of default values or None if unavailable.

        Examples
        --------
        >>> from corrai.fmu import ModelicaFmuModel
        >>> model = ModelicaFmuModel("rosen.fmu", output_list=["res.showNumber"])
        >>> model.get_property_values("x.k")
        ['2.0']
        >>> model.get_property_values(["x.k", "y.k"])
        ['2.0', '2.0']
        """
        if isinstance(property_list, str):
            property_list = (property_list,)

        model_description = fmpy.read_model_description(self.fmu_path.as_posix())
        variable_map = {var.name: var for var in model_description.modelVariables}
        values = []
        for prop in property_list:
            if prop in variable_map:
                try:
                    val = float(variable_map[prop].start)
                except (ValueError, TypeError):
                    val = variable_map[prop].start
                values.append(val)
            else:
                values.append(None)
        return values

    def simulate(
        self,
        property_dict: dict[str, float | int | str] = None,
        simulation_options: dict = None,
        solver_duplicated_keep: str = "last",
        post_process_pipeline: Pipeline = None,
        debug_param: bool = False,
        debug_logging: bool = False,
        logger=None,
    ) -> pd.DataFrame:
        """
        Run an FMU simulation with properties and boundary configuration.

        Parameters
        ----------
        property_dict : dict, optional
            Dictionary of FMU parameter values to set before simulation.
            May include a key ``"boundary"`` with a DataFrame of boundary conditions.
            If both ``property_dict`` and ``simulation_options`` specify boundaries,
            the one in ``property_dict`` takes precedence.
        simulation_options : dict, optional
            Simulation settings. Supported keys include:

            - ``startTime`` : float or pandas.Timestamp
            - ``stopTime`` : float or pandas.Timestamp
            - ``stepSize`` : float or pandas.TimeDelta
            - ``outputInterval`` : float or pandas.TimeDelta. If not provided, it will
                be set equal to ``stepSize``
            - ``solver`` : str
            - ``tolerance`` : float
            - ``fmi_type`` : {"CoSimulation", "ModelExchange"}
            - ``boundary`` : pandas.DataFrame of boundary conditions

        solver_duplicated_keep : {"first", "last"}, default "last"
            Which entry to keep if solver outputs duplicated indices.
        post_process_pipeline : sklearn.Pipeline, optional
            Transformation pipeline applied to simulation results before returning.
        debug_param : bool, default False
            If True, prints the property dictionary before simulation.
        debug_logging : bool, default False
            Enable verbose logging from fmpy.
        logger : callable, optional
            Custom logger for fmpy.

        Returns
        -------
        pandas.DataFrame
            Simulation results indexed by time. If ``startTime`` is a
            :class:`pandas.Timestamp`, the index is a DateTimeIndex; otherwise,
            a numeric index is used.

        Raises
        ------
        ValueError
            If ``startTime`` or ``stopTime`` are outside the boundary DataFrame.

        Notes
        -----
        - Duplicate time indices are resolved using ``solver_duplicated_keep``.

        Examples
        --------
        Run a basic simulation with default options:

        >>> model = ModelicaFmuModel("simple.fmu", output_list=["y"])
        >>> res = model.simulate(
        ...     simulation_options={"startTime": 0, "stopTime": 10, "stepSize": 1}
        ... )
        >>> res.head()
           y
        0.0  0.0
        1.0  1.1
        2.0  2.3
        ...

        Run a simulation with boundary conditions:

        >>> import pandas as pd
        >>> x = pd.DataFrame({"Boundaries.y[1]": [1, 2, 3]}, index=[0, 1, 2])
        >>> model = ModelicaFmuModel(
        ...     "boundary_test.fmu",
        ...     output_list=["Boundaries.y[1]"],
        ...     boundary_table_name="Boundaries",
        ... )
        >>> res = model.simulate(
        ...     simulation_options={
        ...         "boundary": x,
        ...         "startTime": 0,
        ...         "stopTime": 2,
        ...         "stepSize": 1,
        ...     }
        ... )
        >>> res.head()
              Boundaries.y[1]
        time
        0.0               1.0
        1.0               2.0
        2.0               3.0

        """
        property_dict = dict(property_dict or {})
        merged_properties = self.default_properties.copy()
        if property_dict:
            merged_properties.update(property_dict)
        property_dict = merged_properties
        if property_dict and debug_param:
            print(property_dict)

        simulation_options = {
            **DEFAULT_SIMULATION_OPTIONS,
            **(simulation_options or {}),
        }

        start, stop, step, output_int = (
            simulation_options.get(it, None)
            for it in ["startTime", "stopTime", "stepSize", "outputInterval"]
        )

        if output_int is None:
            output_int = step

        start_sec, stop_sec, step_sec, output_int_sec = parse_simulation_times(
            start, stop, step, output_int
        )

        boundary_df = None
        if property_dict:
            boundary_df = property_dict.pop("boundary", boundary_df)

        if simulation_options:
            sim_boundary = simulation_options.pop("boundary", boundary_df)

            if boundary_df is None and sim_boundary is not None:
                boundary_df = sim_boundary
            elif boundary_df is not None and sim_boundary is not None:
                warnings.warn(
                    "Boundary specified in both property_dict and "
                    "simulation_options. The one in property_dict will be used.",
                    UserWarning,
                    stacklevel=2,
                )

        if boundary_df is not None:
            boundary_df = boundary_df.copy()
            if isinstance(boundary_df.index, pd.DatetimeIndex):
                boundary_df.index = datetime_index_to_seconds_index(boundary_df.index)

            if not (
                boundary_df.index[0] <= start_sec <= boundary_df.index[-1]
                and boundary_df.index[0] <= stop_sec <= boundary_df.index[-1]
            ):
                raise ValueError(
                    "'startTime' and 'stopTime' are outside boundary DataFrame"
                )

            self.boundary_file_path = self.simulation_dir / "boundaries.txt"
            df_to_combitimetable(boundary_df, self.boundary_file_path)

        with simulation_workspace(self.fmu_path, self.boundary_file_path) as (
            local_fmu,
            local_boundary,
        ):
            if local_boundary is not None and self.boundary_table_name:
                property_dict[f"{self.boundary_table_name}.fileName"] = (
                    local_boundary.as_posix()
                )

            result = simulate_fmu(
                filename=local_fmu,
                start_time=start_sec,
                stop_time=stop_sec,
                step_size=step_sec,
                relative_tolerance=simulation_options["tolerance"],
                start_values=property_dict,
                output=self.output_list,
                solver=simulation_options["solver"],
                output_interval=output_int_sec,
                fmi_type=simulation_options["fmi_type"],
                debug_logging=debug_logging,
                logger=logger,
            )

        columns = ["time"] + self.output_list if self.output_list else None
        df = pd.DataFrame(result, columns=columns)

        if isinstance(start, (pd.Timestamp, dt.datetime)):
            df.index = seconds_index_to_datetime_index(df["time"], start.year)
            df.index = df.index.round("s")
            df = df.tz_localize(start.tz)
            df.index.freq = df.index.inferred_freq
        else:
            df.index = round(df["time"], 2)

        df.drop(columns=["time"], inplace=True)

        df = df.loc[~df.index.duplicated(keep=solver_duplicated_keep)]

        if post_process_pipeline is not None:
            df = post_process_pipeline.fit_transform(df)

        return df

    def save(self, file_path: Path):
        """
        Save the FMU file to a new location.

        Parameters
        ----------
        file_path : Path
            Destination path.

        """
        shutil.copyfile(self.fmu_path, file_path)

    def __repr__(self):
        """
        Return a string representation of the FMU model.

        Returns
        -------
        str
            A formatted string containing FMU name, description, version,
            and available parameters.

        Examples
        --------
        >>> print(model)
        Model Name: rosen
        Description: ModelDescription(
            fmiVersion='2.0', modelName='rosen',
            coSimulation=CoSimulation(modelIdentifier='rosen'),
            modelExchange=ModelExchange(modelIdentifier='rosen'),
            scheduledExecution=None
        )
        Version: 2.0
        Parameters:
          Name: x.k, Default Value: 2.0, Description: Constant output value
          Name: y.k, Default Value: 2.0, Description: Constant output value
          Name: res.significantDigits, Default Value: 2,
          Description: Number of significant digits to be shown
        """
        model_description = fmpy.read_model_description(self.fmu_path.as_posix())

        model_info = f"Model Name: {model_description.modelName}\n"
        model_info += (
            f"Description: {fmpy.read_model_description(self.fmu_path.as_posix())}\n"
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
