import pandas as pd
import numpy as np


class ScikitFunction:
    """
    A class that represents a scikit-learn function.

    Parameters:
    - simulator (object): The simulator object used for the function.
    - surrogate (object): The surrogate object used for prediction.
    - param_list (list): A list of parameter names.
    - indicators (list, optional): A list of indicator names. Defaults to None.

    Attributes:
    - simulator (object): The simulator object used for the function.
    - surrogate (object): The surrogate object used for prediction.
    - param_list (list): A list of parameter names.
    - indicators (list): A list of indicator names.

    Methods:
    - function(x_dict): Calculates the function value for the given input dictionary.

    """

    def __init__(
        self,
        simulator,
        surrogate,
        param_list,
        indicators=None,
    ):
        self.simulator = simulator
        self.surrogate = surrogate
        self.param_list = param_list
        if indicators is None:
            self.indicators = simulator.output_list
        else:
            self.indicators = indicators

    def function(self, x_dict):
        """
        Calculates the function value for the given input dictionary.

        Args:
        - x_dict (dict): A dictionary of input values.

        Returns:
        - res_series (Series): A pandas Series object containing the function values.
        """

        temp_array = np.array(list(x_dict.values()))
        res = self.surrogate.predict(x_array=temp_array)
        res_series = pd.Series(data=res[0, 0], dtype="float64")
        res_series.index = [self.indicators]

        return res_series


class ModelicaFunction:
    """
    A class that defines a function based on a Modelitool Simulator.

    Args:
        simulator (object): A fully configured Modelitool Simulator object.
        param_list (list): A list of parameter defined as dictionaries. At least , each
            parameter dict must have the following keys : "names", "interval".
        indicators (list, optional): A list of indicators to be returned by the
            function. An indicator must be one of the Simulator outputs. If not
            provided, all indicators in the simulator's output list will be returned.
            Default is None.
        agg_methods_dict (dict, optional): A dictionary that maps indicator names to
            aggregation methods. Each aggregation method should be a function that takes
            an array of values and returns a single value. It can also be an error
            function that will return an error indicator between the indicator results
            and a reference array of values defined in reference_df.
            If not provided, the default aggregation method for each indicator is
            numpy.mean. Default is None.
        reference_dict (dict, optional): When using an error function as agg_method, a
            reference_dict must be used to map indicator names to reference indicator
            names. The specified reference name will be used to locate the value in
            reference_df.
            If provided, the function will compute each indicator's deviation from its
            reference indicator using the corresponding aggregation method.
            Default is None.
        reference_df (pandas.DataFrame, optional): A pandas DataFrame containing the
            reference values for each reference indicator specified in reference_dict.
            The DataFrame should have the same length as the simulation results.
            Default is None.

    Returns:
        pandas.Series: A pandas Series containing the function results.
        The index is the indicator names and the values are the aggregated simulation
        results.

    Raises:
        ValueError: If reference_dict and reference_df are not both provided or both
        None.
    """

    def __init__(
        self,
        simulator,
        param_list,
        indicators=None,
        agg_methods_dict=None,
        reference_dict=None,
        reference_df=None,
    ):
        self.simulator = simulator
        self.param_list = param_list
        if indicators is None:
            self.indicators = simulator.output_list
        else:
            self.indicators = indicators
        if agg_methods_dict is None:
            self.agg_methods_dict = {ind: np.mean for ind in self.indicators}
        else:
            self.agg_methods_dict = agg_methods_dict
        if (reference_dict is not None and reference_df is None) or (
            reference_dict is None and reference_df is not None
        ):
            raise ValueError("Both reference_dict and reference_df should be provided")
        self.reference_dict = reference_dict
        self.reference_df = reference_df

    def function(self, x_dict):
        """
        Calculates the function value for the given input dictionary.

        Args:
        - x_dict (dict): A dictionary of input values.

        Returns:
        - res_series (Series): A pandas Series object containing the function values.

        """
        temp_dict = {param["name"]: x_dict[param["name"]] for param in self.param_list}
        self.simulator.set_param_dict(temp_dict)
        self.simulator.simulate()
        res = self.simulator.get_results()

        res_series = pd.Series(dtype="float64")
        solo_ind_names = self.indicators
        if self.reference_dict is not None:
            for k in self.reference_dict.keys():
                res_series[k] = self.agg_methods_dict[k](
                    res[k], self.reference_df[self.reference_dict[k]]
                )

            solo_ind_names = [
                i for i in self.indicators if i not in self.reference_dict.keys()
            ]

        for ind in solo_ind_names:
            res_series[ind] = self.agg_methods_dict[ind](res[ind])
        return res_series
