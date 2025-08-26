import datetime as dt
import shutil
import tempfile
from pathlib import Path

import fmpy
from fmpy import simulate_fmu
import pandas as pd
from sklearn.pipeline import Pipeline

from corrai.base.model import Model
from corrai.base.parameter import Parameter


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


def datetime_to_second(datetime_in: dt.datetime | pd.Timestamp):
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
    year = datetime_in.year
    origin = dt.datetime(year, 1, 1)
    return (datetime_in - origin).total_seconds()


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
    >>> df = pd.DataFrame({"val": [1, 2]}, index=pd.date_range("2020-01-01", periods=2, freq="H"))
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


def get_start_stop_year_tz_from_x(x: pd.DataFrame = None):
    """
    Extract simulation time bounds and time zone from boundary condition data.

    Parameters
    ----------
    x : pd.DataFrame, optional
        DataFrame with DatetimeIndex or numeric index.

    Returns
    -------
    tuple
        (start, stop, year, tz):
        - start : float
            Minimum time (in seconds).
        - stop : float
            Maximum time (in seconds).
        - year : int or None
            Reference year if datetime index was used.
        - tz : datetime.tzinfo or None
            Time zone information.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2020-01-01", periods=3, freq="H", tz="UTC")
    >>> x = pd.DataFrame({"val": [1,2,3]}, index=idx)
    >>> get_start_stop_year_tz_from_x(x)
    (0.0, 7200.0, 2020, datetime.timezone.utc)
    """
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
    A wrapper class for FMU (Functional Mock-up Unit) simulations using fmpy.

    This class integrates FMUs into the ``Model`` formalism of corrai and
    provides convenience methods for setting boundary conditions, parameters,
    and simulation options.

    Parameters
    ----------
    fmu_path : Path
        Path to the FMU file.
    simulation_options : dict, optional
        Dictionary of simulation options including:
        ``startTime``, ``stopTime``, ``stepSize``, ``solver``,
        ``outputInterval``, ``tolerance``, ``fmi_type``.
            Can also include key ``x`` with a DataFrame of boundary conditions.
    output_list : list of str, optional
        List of output variable names to record.
    x_combitimetable_name : str, optional
        Name of the CombiTimeTable block in the Modelica model.
    simulation_dir : Path, optional
        Directory for simulation files (temporary dir if not given).

    Attributes
    ----------
    parameters : dict
        Dictionary of FMU parameters to be set before simulation.
    simulation_options : dict
        Dictionary of simulation options.
    output_list : list of str
        Variables to output from the FMU.
    model_path : Path
        Path to the FMU file.
    simulation_dir : Path
        Directory for temporary simulation artifacts.

    Examples
    --------
    >>> from corrai.fmu import ModelicaFmuModel
    >>> model = ModelicaFmuModel(
    ...     fmu_path="rosen.fmu",
    ...     simulation_options={"startTime": 0, "stopTime": 10, "stepSize": 1},
    ...     output_list=["res.showNumber"]
    ... )
    >>> result = model.simulate()
    >>> result.head()
                       res.showNumber
    1970-01-01 00:00:00+00:00        401.0
    1970-01-01 00:00:01+00:00        401.0
    """

    def __init__(
        self,
        fmu_path: Path,
        simulation_options: dict[str, float | str | int] = None,
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
        if simulation_options is not None:
            self._set_x_sim_options(simulation_options=simulation_options)

    def set_x(self, df: pd.DataFrame):
        """
        Set input boundary data for the simulation and update the FMU parameters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DatetimeIndex or seconds index containing boundary conditions.

        Examples
        --------
        >>> import pandas as pd
        >>> x = pd.DataFrame({"u": [1, 2, 3]}, index=[0, 1, 2])
        >>> model.set_x(x)
        """
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
        simulation_options: dict[
            str, float | str | int | dt.datetime | pd.Timestamp
        ] = None,
    ):
        """
        Update simulation options and handle boundary data if passed.

        Parameters
        ----------
        simulation_options : dict, optional
            Dictionary of simulation options. If key ``x`` is present,
            it is treated as boundary condition data (pd.DataFrame).

        Examples
        --------
        >>> import pandas as pd
        >>> from corrai.fmu import ModelicaFmuModel
        >>> model = ModelicaFmuModel(
        ...     fmu_path="boundary_test.fmu",
        ...     output_list=["Boundaries.y[1]", "Boundaries.y[2]"]
        ... )
        >>> x = pd.DataFrame(
        ...     {"Boundaries.y[1]": [1, 2, 3], "Boundaries.y[2]": [3, 4, 5]},
        ...     index=[3, 4, 5]
        ... )
        >>> model._set_x_sim_options({
        ...     "x": x,
        ...     "stepSize": 1,
        ...     "outputInterval": 1,
        ...     "solver": "CVode"
        ... })
        """
        if simulation_options is None:
            return

        if "x" in simulation_options:
            self.set_x(simulation_options["x"])
            simulation_options = {
                k: v for k, v in simulation_options.items() if k != "x"
            }

        to_update = {
            key: val
            for key, val in simulation_options.items()
            if key not in ["startTime", "stopTime"]
        }
        self.simulation_options.update(to_update)

        if "x" not in simulation_options:
            simo = {}
            for key in ["startTime", "stopTime"]:
                if key in simulation_options:
                    if isinstance(simulation_options[key], (dt.datetime, pd.Timestamp)):
                        simo[key] = datetime_to_second(simulation_options[key])
                        if key == "startTime":
                            self._begin_year = simulation_options["startTime"].year
                    else:
                        simo[key] = simulation_options[key]
                else:
                    simo[key] = self.simulation_options[key]
            self.simulation_options["startTime"] = simo["startTime"]
            self.simulation_options["stopTime"] = simo["stopTime"]

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

        model_description = fmpy.read_model_description(self.model_path.as_posix())
        variable_map = {var.name: var for var in model_description.modelVariables}
        values = []
        for prop in property_list:
            if prop in variable_map:
                val = variable_map[prop].start
                values.append(val if val is not None else None)
            else:
                values.append(None)
        return values

    def simulate_parameter(
        self,
        parameter_value_pairs: list[tuple["Parameter", str | int | float]],
        simulation_options: dict = None,
        simulation_kwargs: dict = None,
    ) -> pd.DataFrame:
        """
        Run a simulation by providing parameters as (Parameter, value) pairs.

        Parameters
        ----------
        parameter_value_pairs : list of tuple
            Each element is a (Parameter, value) pair.
        simulation_options : dict, optional
            Simulation options (see ``simulate``).
        simulation_kwargs : dict, optional
            Additional keyword arguments (see ``simulate``).

        Returns
        -------
        pd.DataFrame
            Simulation results.

        Examples
        --------
        >>> from corrai.base.parameter import Parameter
        >>> p = Parameter(name="x", model_property="x.k", interval=(0, 5))
        >>> model.simulate_parameter([(p, 3)])
           res.showNumber
        0            401.0
        1            401.0
        2            401.0
        """
        property_dict = self.get_property_from_param(parameter_value_pairs)
        return self.simulate(property_dict, simulation_options, simulation_kwargs)

    def simulate(
        self,
        property_dict: dict[str, float | int | str] = None,
        simulation_options: dict = None,
        x: pd.DataFrame = None,
        solver_duplicated_keep: str = "last",
        post_process_pipeline: Pipeline = None,
        debug_param: bool = False,
        debug_logging: bool = False,
        logger=None,
    ) -> pd.DataFrame:
        """
        Run an FMU simulation with given parameters, options, and boundary conditions.

        Parameters
        ----------
        property_dict : dict of {str: float or int or str}, optional
            Dictionary of FMU parameter values to override before simulation.
            Keys are FMU variable names (e.g., ``{"x.k": 3, "y.k": 4}``).
        simulation_options : dict, optional
            Dictionary of simulation options. It may include:
            - ``startTime`` : float or datetime
            - ``stopTime`` : float or datetime
            - ``stepSize`` : float
            - ``solver`` : str
            - ``outputInterval`` : float
            - ``tolerance`` : float
            - ``fmi_type`` : {"ModelExchange", "CoSimulation"}
            - ``x`` : pd.DataFrame, optional
                Boundary conditions to inject as a CombiTimeTable
                (alternative to passing via ``simulation_kwargs``).
            - ``x`` : pd.DataFrame
                Boundary conditions.
            - ``solver_duplicated_keep`` : {"first", "last"}, default="last"
                Which duplicated index to keep if solver returns duplicate timesteps.
            - ``post_process_pipeline`` : sklearn.Pipeline, optional
                Pipeline applied to the results (e.g. resampling).
            - ``debug_logging`` : bool, default False
                Enable verbose logging from fmpy.
            - ``logger`` : callable, optional
                Logger function passed to fmpy.

        Returns
        -------
        pd.DataFrame
            Simulation results indexed by time (as `DatetimeIndex` if a year was inferred,
            otherwise numeric seconds). Columns are the variables in ``output_list``.

        Examples
        --------
        Basic simulation with default parameters:

        >>> from corrai.fmu import ModelicaFmuModel
        >>> model = ModelicaFmuModel(
        ...     fmu_path="rosen.fmu",
        ...     simulation_options={"startTime": 0, "stopTime": 2, "stepSize": 1},
        ...     output_list=["res.showNumber"]
        ... )
        >>> result = model.simulate()
        >>> result
           res.showNumber
        0            401.0
        1            401.0
        2            401.0
        """

        if debug_param:
            print(property_dict)

        self.parameters.update(property_dict or {})
        if x is not None:
            self.set_x(x)

        self._set_x_sim_options(simulation_options)

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
        Save the FMU file to a new location.

        Parameters
        ----------
        file_path : Path
            Destination path.

        """
        shutil.copyfile(self.model_path, file_path)

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
