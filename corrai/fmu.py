from fmpy import simulate_fmu
import tempfile
import warnings

from pathlib import Path
import pandas as pd
import datetime as dt
from modelitool.combitabconvert import seconds_to_datetime, df_to_combitimetable

from corrai.base.model import Model


class FmuModel(Model):
    """
    A model class for simulating FMUs (Functional Mock-up Units)
        in a standardized and flexible way,integrating various data
        sources and simulation parameters.

    The `FmuModel` class provides functionalities for setting up and
        running simulations of FMUs, including handling initial parameters,
        boundary conditions, and simulation options.It is designed to
        facilitate easy integration of simulation results into larger workflows,
        with support for converting boundary conditions from Pandas DataFrames
        and adjusting simulation parameters dynamically.


    Methods:

        set_simulation_options(self, simulation_options):
            Updates the simulation options.

        set_boundaries_df(self, df):
            Sets boundary conditions from a Pandas DataFrame and updates
                the model's initial parameters to include these boundaries.

        set_param_dict(self, param_dict):
            Updates the initial parameters of the model with
                the values from `param_dict`.

        simulate: Runs the FMU simulation with the specified parameters and options,
             returning the results as a Pandas DataFrame indexed by datetime.

            Parameters:
                parameter_dict (dict, optional): Parameter values to
                    override or update before simulation. simulation_options:
                    Simulation options to override or update before simulation.
                debug_logging (bool, optional): Enables or disables debug
                    logging for the simulation.
                logger (logging.Logger, optional): A logging.Logger instance
                 for recording simulation logs.

            Returns:
                pd.DataFrame: A DataFrame containing the simulation results,
                    with timestamps as the index.
    """

    def __init__(
        self,
        model_path,
        simulation_options,
        output_list,
        init_parameters=None,
        boundary_df=None,
        year=None,
    ):
        """
        Initialize an instance of the FmuModel class.

        This method sets up the model with necessary paths, simulation options,
            output variables, initial parameters, boundary conditions,
            and the simulation year.

        Parameters:
            model_path (str or Path): The file path to the FMU model.
                Can be a string or a Path object.
            simulation_options (dict): Options for configuring the simulation,
                such as start time, stop time, step size, solver, etc.
            output_list (list of str): A list of output variables
                that the simulation should record.
            init_parameters (dict, optional): Initial values for
                parameters within the FMU model. Defaults to an
                empty dict if None is provided.
            boundary_df (pd.DataFrame, optional): A DataFrame containing
                boundary conditions for the simulation. Must have datetime indices.
                If provided, it will set initial conditions based on this data.
            year (int, optional): The year in which the simulation's
                time series starts. This is important for aligning
                the simulation results with real-world dates.
                Ignored if `boundary_df` is provided since the year
                will be derived from the DataFrame's index.

        Raises:
            Warning: If both `boundary_df` and `year` are provided,
                a warning is raised indicating that `year` will be
                ignored and derived from `boundary_df` instead.
        """
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
        """
        Update the simulation options for the model.

        This method allows for dynamic adjustment of the
            simulation settings, such as changing the solver
            or the time frame of the simulation.

        Parameters:
            simulation_options (dict): A dictionary containing the
                simulation options to be updated. Options might
                include 'startTime', 'stopTime', 'stepSize', 'solver',
                'outputInterval', and 'fmi_type'.
        """
        self.simulation_options = simulation_options

    def set_boundaries_df(self, df):
        """
        Set boundary conditions for the simulation
            from a Pandas DataFrame.

        The DataFrame should contain time series data
            that will be used to generate a boundary
            condition file.The index of the DataFrame
            must be a DateTimeIndex, as it determines
            the simulation's time frame.

        Parameters:
            df (pd.DataFrame): A DataFrame containing the boundary
                conditions with a DateTimeIndex.

        Raises:
            ValueError: If the DataFrame's index is not a DateTimeIndex,
            an error is raised indicating the requirement for datetime indices.
        """
        new_bounds_path = self._simulation_path / "boundaries.txt"
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
        """
        Update the initial parameter values for the model.

        This method allows for the dynamic setting or updating of
            initial parameters before running the simulation.

        Parameters:
            param_dict (dict): A dictionary where keys are parameter
                names and values are the corresponding values
                to be set or updated in the model.
        """
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
        """
        Run the simulation with the current model configuration.

        This method executes the FMU simulation according to the specified
            parameters and options, and returns the simulation
            results as a Pandas DataFrame indexed by datetime.

        Parameters:
            parameter_dict (dict, optional): Additional or overriding
             parameter values for this simulation run.
            simulation_options (dict, optional): Additional or
                overriding   simulation options for this run.
            debug_logging (bool, optional): Whether to enable
                detailed logging of the simulation process. Defaults to False.
            logger (logging.Logger, optional): A logger for capturing
                simulation logs, if debug_logging is True.

        Returns:
            pd.DataFrame: A DataFrame containing the simulation results.
                Columns include 'time' and the specified output variables,
                with the DataFrame indexed by datetime reflecting
                the simulation period.
        """
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
            relative_tolerance=self.simulation_options.get("tolerance", 1e-6),
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
