import pandas as pd

from corrai.base.simulate import run_models_in_parallel
from corrai.math import aggregate_time_series

from SALib.sample import fast_sampler, saltelli, latin
from SALib.sample import morris as morris_sampler
from SALib.analyze import fast, morris, sobol, rbd_fast

import numpy as np

METHOD_SAMPLER_DICT = {
    "FAST": {
        "method": fast,
        "sampling": fast_sampler,
    },
    "Morris": {"method": morris, "sampling": morris_sampler},
    "Sobol": {
        "method": sobol,
        "sampling": saltelli,
    },
    "RBD_fast": {
        "method": rbd_fast,
        "sampling": latin,
    },
}


class SAnalysis:
    """
    This class is designed to perform sensitivity analysis on a given model using
    various sensitivity methods,  Global Sensitivity Analysis (GSA) methods such as
    Sobol and FAST.

    Parameters:
    - parameters_list (list of dict): A list of dictionaries describing the model
        parameters to be analyzed. Each dictionary should contain 'name', 'interval',
        and 'type' keys to define the parameter name, its range (interval),
        and its data type (Real, integer).
    - method (str): The sensitivity analysis method to be used.
        Supported methods are specified in METHOD_SAMPLER_DICT.keys().

    Attributes:
    - method (str): The selected sensitivity analysis method.
    - parameters_list (list of dict): The list of model parameters for analysis.
    - _salib_problem (dict): Problem definition for the sensitivity analysis library
        SAlib.
    - sample (DataFrame): The generated parameter samples.
    - sensitivity_results (dict): The sensitivity analysis results.
    - sample_results (list): List of results obtained from model evaluations.

    Methods:
    - set_parameters_list(parameters_list): Sets the parameters list and updates
        the _salib_problem.
    - draw_sample(n, sampling_kwargs=None): Draws a sample of parameter values for
        sensitivity analysis.
    - evaluate(model, simulation_options, n_cpu=1): Evaluates the model for the
        generated parameter samples.
    - analyze(indicator, agg_method=np.mean, agg_method_kwarg={}, reference_df=None,
        sensitivity_method_kwargs={}): Performs sensitivity analysis on the model
        outputs using the selected method and indicator.

    """

    def __init__(self, parameters_list: [dict], method: str):
        if method not in METHOD_SAMPLER_DICT.keys():
            raise ValueError("Specified sensitivity method is not valid")
        else:
            self.method = method
        self.parameters_list = parameters_list
        self._salib_problem = None
        self.set_parameters_list(parameters_list)
        self.sample = None
        self.sensitivity_results = None
        self.sample_results = []

    def set_parameters_list(self, parameters_list: list):
        """
        Set the list of model parameters and update the _salib_problem definition.

        Parameters:
        - parameters_list (list of dict): A list of dictionaries describing the model
            parameters to be analyzed. Each dictionary should contain 'name',
             'interval', and 'type' keys.
        """
        self._salib_problem = {
            "num_vars": len(parameters_list),
            "names": [p["name"] for p in parameters_list],
            "bounds": list(map(lambda p: p["interval"], parameters_list)),
        }

    def draw_sample(self, n: int, sampling_kwargs: dict = None):
        """
        Samples the parameters for sensitivity analysis.

        Parameters:
        - n (int): The number of samples to generate.
        - sampling_kwargs (dict, optional): Additional keyword arguments for the
            sampling method.
        """
        # Erase previous result to prevent false indicator
        self.sample_results = []

        if sampling_kwargs is None:
            sampling_kwargs = {}

        sampler = METHOD_SAMPLER_DICT[self.method]["sampling"]
        sample_temp = sampler.sample(
            N=n, problem=self._salib_problem, **sampling_kwargs
        )

        for index, param in enumerate(self.parameters_list):
            vtype = param["type"]
            if vtype == "Integer":
                sample_temp[:, index] = np.round(sample_temp[:, index])
                sample_temp = np.unique(sample_temp, axis=0)

        self.sample = pd.DataFrame(sample_temp, columns=self._salib_problem["names"])

    def evaluate(self, model, simulation_options: dict, n_cpu: int = 1):
        """
        Evaluate the model for the generated samples.

        Parameters:
        - model: The model to be evaluated.
        - simulation_options (dict): Options for running the model simulations.
        - n_cpu (int, optional): Number of CPU cores to use for parallel evaluation.
        """
        if self.sample is None:
            raise ValueError(
                "Cannot perform evaluation, no sample was found. "
                "draw sample using draw_sample() method"
            )

        self.sample_results = run_models_in_parallel(
            model=model,
            parameter_samples=self.sample,
            simulation_options=simulation_options,
            n_cpu=n_cpu,
        )

    def analyze(
        self,
        indicator: str,
        agg_method=np.mean,
        agg_method_kwarg: dict = None,
        reference_df: pd.DataFrame = None,
        sensitivity_method_kwargs: dict = None,
    ):
        """
        Perform sensitivity analysis on the model outputs using the selected method
         and indicator.

        Parameters:
        - indicator (str): The model output indicator to analyze.
        - agg_method (function, optional): Aggregation method for time series data.
        - agg_method_kwarg (dict, optional): Additional keyword arguments for the
            aggregation method.
        - reference_df (pd.DataFrame, optional): Reference data to pass to the
            aggregation method. Usually usefull for error function analysis.
            (eg. mean_error(simulation, reference_measurement)
        - sensitivity_method_kwargs (dict, optional): Additional keyword arguments for
            the sensitivity analysis method.
        """

        if sensitivity_method_kwargs is None:
            sensitivity_method_kwargs = {}

        if agg_method_kwarg is None:
            agg_method_kwarg = {}

        if not self.sample_results:
            raise ValueError(
                "No simulation results were found. Use evaluate() method to run "
                "the model."
            )

        if agg_method_kwarg is None:
            agg_method_kwarg = {}

        if indicator not in self.sample_results[0][2].columns:
            raise ValueError("Specified indicator not in computed outputs")

        analyser = METHOD_SAMPLER_DICT[self.method]["method"]

        results_2d = pd.concat(
            [s_res[2][indicator] for s_res in self.sample_results], axis=1
        )
        y_array = np.array(
            aggregate_time_series(
                results_2d, agg_method, agg_method_kwarg, reference_df
            )
        )

        if self.method in ["Sobol", "FAST"]:
            self.sensitivity_results = analyser.analyze(
                problem=self._salib_problem, Y=y_array, **sensitivity_method_kwargs
            )
        elif self.method in ["Morris", "RBD_fast"]:
            self.sensitivity_results = analyser.analyze(
                problem=self._salib_problem,
                X=self.sample,
                Y=y_array,
                **sensitivity_method_kwargs
            )
