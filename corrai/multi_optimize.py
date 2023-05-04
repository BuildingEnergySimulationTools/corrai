import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary


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
        self, parameters, obj_func_list, func_list, function_names, constraint_names
    ):
        self.parameters = parameters
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
            xl=np.array([p["interval"][0] for p in parameters]),
            xu=np.array([p["interval"][-1] for p in parameters]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        current_param = {param["name"]: val for param, val in zip(self.parameters, x)}
        res = pd.concat(
            [m.function(current_param) for m in self.obj_func_list]
            + [pyf(current_param) for pyf in self.func_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraint_names])


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
         - 'Binary: variable that takes binary values (0 or 1).
         - 'Choice': variable that represents multiple choices from a
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
        self, parameters, obj_func_list, func_list, function_names, constraint_names
    ):
        global var
        self.parameters = parameters
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
            name, bounds, vtype = param["name"], param["interval"], param["type"]
            if vtype == "Integer":
                var = Integer(bounds=bounds)
            elif vtype == "Real":
                var = Real(bounds=bounds)
            elif vtype == "Choice":
                var = Choice(options=bounds)
            elif vtype == "Binary":
                var = Binary()
            variable_string[name] = var

        super().__init__(
            vars=variable_string,
            n_var=len(parameters),
            n_obj=len(function_names),
            n_ieq_constr=len(constraint_names),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        res = pd.concat(
            [m.function(X) for m in self.obj_func_list]
            + [pyf(X) for pyf in self.func_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraint_names])
