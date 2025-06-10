from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary

from corrai.base.parameter import Parameter


# TODO Add permutation variables to MyMixedProblem
class MyProblem(ElementwiseProblem):
    """
    A class that represents a single-objective optimization problem
    with one or more constraints. This class inherits from the PyMOO
    ElementwiseProblem class and overrides the _evaluate method.

    Parameters
    ----------
    parameters (list):
        A list of dictionaries, where each dictionary represents a
        parameter and contains its name, interval, and type.
    obj_func_list (list):
        A list of objects that have methods that represent the objective
        functions to be optimized or constraints (example: Modelica,
        EnergyPlus models...). If optimization is subject to inequality
        constraints g, they should be formulated as "less than" constraints
        (i.e., g(x) < 0). Leave empy if none.
    func_list (list):
        A list of objects that have methods that represent the objective
        functions to be optimized or constraints. If optimization is subject
        to inequality constraints g, they should be formulated as "less than"
        constraints (i.e., g(x) < 0). Leave empy if none.
    function_names (list):
        A list of strings that represent the names of the objective functions.
    constraint_names (list):
        A list of strings that represent the names of the constraint. Leave
        empy if none.

    Attributes
    ----------
    n_var (int):
        The number of variables inherited from parameters.
    n_obj (int):
        The number of objective functions inherited from parameters.
    n_ieq_constr (int):
        The number of constraints inherited from parameters.
    xl (numpy.ndarray):
        An array of lower bound of each variable inherited from parameters.
    xu (numpy.ndarray):
        An array of upper bound of each variable inherited from parameters.

    Methods
    -------
    _evaluate(x, out, *args, **kwargs):
        Evaluates the problem for the given variable values and returns
        the objective values out["F"] as a list of NumPy array with length of
        n_obj and the constraints values out["G"] with length of n_ieq_constr
        (if the problem has constraints to be satisfied at all).

    """

    def __init__(
        self,
        parameters: list[dict[Parameter, Any]],
        obj_func_list,
        func_list,
        function_names,
        constraint_names,
    ):
        self.parameters = parameters
        _check_duplicate_params(parameters)
        if len(obj_func_list) == 0 and len(func_list) == 0:
            raise ValueError(
                "At least one of obj_func_list or func_list should be provided"
            )
        self.obj_func_list = obj_func_list
        self.func_list = func_list
        self.function_names = function_names
        self.constraint_names = constraint_names

        super().__init__(
            n_var=len(parameters),
            n_obj=len(function_names),
            n_ieq_constr=len(constraint_names),
            xl=np.array([p[Parameter.INTERVAL][0] for p in parameters]),
            xu=np.array([p[Parameter.INTERVAL][-1] for p in parameters]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        current_param = {
            param[Parameter.NAME]: val for param, val in zip(self.parameters, x)
        }
        res = pd.concat(
            [pd.Series(m.function(current_param)) for m in self.obj_func_list]
            + [pd.Series(pyf(current_param)) for pyf in self.func_list],
            axis=0,
        ).T
        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraint_names])

    def plot_pcp(self, res, ref, bounds=False):
        data_dict = {
            param[Parameter.NAME]: res.X[:, i]
            for param, i in zip(self.parameters, range(res.X.shape[1]))
        }
        data_dict.update(
            {self.function_names[i]: res.F[:, i] for i in range(res.F.shape[1])}
        )

        plot_parcoord(
            data_dict=data_dict,
            bounds=bounds,
            parameters=self.parameters,
            colorby=ref,
            obj_res=res.F,
        )


class MyMixedProblem(ElementwiseProblem):
    """
     A class that represents a mixed-integer optimization problem with one
     or more constraints. This class inherits from the PyMOO ElementwiseProblem
     class and overrides the _evaluate method.

     Parameters
     ----------
     parameters (list):
         A list of dictionaries, where each dictionary represents a parameter
         and contains its name, interval, and type.
     obj_func_list (list):
         A list of objects that have methods that represent the objective
         functions to be optimized. Leave empy if none
     func_list (list):
         A list of PyFunc objects that represent the constraints to be satisfied.
         Leave empy if none.
     function_names (list):
         A list of strings that represent the names of the objective functions.
         Leave empy if none.
     constraint_names (list):
         A list of strings that represent the names of the constraints.
         Leave empy if none.

     Attributes
     ----------
     vars (dict):
         A dictionary of decision variables and their types inherited
         from parameters. The variable types can be any of the following:
         - Real: continuous variable that takes values within a range of real numbers.
         - Integer: variable that takes integer values within a range.
         - Binary: variable that takes binary values (0 or 1).
         - Choice: variable that represents multiple choices from a
         set of discrete values.
     n_var (int):
        The number of variables inherited from parameters.
     n_obj (int):
        The number of objective functions inherited from parameters.
     n_ieq_constr (int):
        The number of constraints inherited from parameters.

     Methods
     -------
    _evaluate(x, out, *args, **kwargs):
        Evaluates the problem for the given variable values and returns
        the objective values out["F"] as a list of NumPy array with length of
        n_obj and the constraints values out["G"] with length of n_ieq_constr
        (if the problem has constraints to be satisfied at all).
    """

    def __init__(
        self,
        parameters: list[dict[Parameter, Any]],
        obj_func_list,
        func_list,
        function_names,
        constraint_names,
    ):
        self.parameters = parameters
        _check_duplicate_params(parameters)
        if len(obj_func_list) == 0 and len(func_list) == 0:
            raise ValueError(
                "At least one of obj_func_list or func_list should be provided"
            )
        self.obj_func_list = obj_func_list
        self.func_list = func_list
        self.function_names = function_names
        self.constraint_names = constraint_names
        variable_string = {}

        for param in parameters:
            name, bounds, vtype = (
                param[Parameter.NAME],
                param[Parameter.INTERVAL],
                param[Parameter.TYPE],
            )
            if vtype == "Integer":
                var = Integer(bounds=bounds)
            elif vtype == "Real":
                var = Real(bounds=bounds)
            elif vtype == "Choice":
                var = Choice(options=bounds)
            elif vtype == "Binary":
                var = Binary()
            else:
                raise ValueError(
                    "Unknown vtype. Choose one of 'Integer', 'Real', 'Choice', "
                    "'Binary'"
                )
            variable_string[name] = var

        super().__init__(
            vars=variable_string,
            n_obj=len(function_names),
            n_ieq_constr=len(constraint_names),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        current_param = {
            param[Parameter.NAME]: val for param, val in zip(self.parameters, x)
        }
        print(current_param)
        res = pd.concat(
            [pd.Series(m.function(current_param)) for m in self.obj_func_list]
            + [pd.Series(pyf(current_param)) for pyf in self.func_list],
            axis=0,
        ).T

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraint_names])

    def plot_pcp(self, res, ref, bounds=False):
        data_dict = {
            param[Parameter.NAME]: res.X[:, i]
            for param, i in zip(self.parameters, range(res.X.shape[1]))
        }
        data_dict.update(
            {self.function_names[i]: res.F[:, i] for i in range(res.F.shape[1])}
        )

        plot_parcoord(
            data_dict=data_dict,
            bounds=bounds,
            parameters=self.parameters,
            colorby=ref,
            obj_res=res.F,
        )


def plot_parcoord(
    data_dict,
    bounds,
    parameters,
    obj_res,
    colorby=None,
    html_file_path=None,
    plot_unselected=True,
):
    # Define the color palette
    color_palette = ["#FFAD85", "#FF8D70", "#ED665A", "#52E0B6", "#479A91"]

    if bounds:
        range_down = [p[Parameter.INTERVAL][0] for i, p in enumerate(parameters)]
        range_down.extend(obj_res.min(axis=0))
        range_up = [p[Parameter.INTERVAL][-1] for i, p in enumerate(parameters)]
        range_up.extend(obj_res.max(axis=0))
        ranges = [[range_down[i], range_up[i]] for i in range(len(range_up))]

    else:
        ranges = [
            [data_dict[par].min(), data_dict[par].max()] for par in data_dict.keys()
        ]

    unselected = dict(line=dict(color="grey", opacity=0.5 if plot_unselected else 0))
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=data_dict[colorby],
                colorscale=color_palette,
                showscale=True,
                cmin=data_dict[colorby].min(),
                cmax=data_dict[colorby].max(),
            ),
            dimensions=[
                {
                    "range": r,
                    "label": par,
                    "values": data_dict[par],
                }
                for par, r in zip(data_dict.keys(), ranges)
            ],
            unselected=unselected,
        )
    )

    fig.show()

    if html_file_path:
        pio.write_html(fig, html_file_path)


def _check_duplicate_params(params):
    """
    Check for duplicate parameter names in the list of parameters.

    Raises
    ------
    ValueError
        If duplicate parameter names are found.
    """
    param_names = set()
    for param in params:
        name = param[Parameter.NAME]
        if name in param_names:
            raise ValueError(f"Duplicate parameter name: {name}")
        param_names.add(name)
