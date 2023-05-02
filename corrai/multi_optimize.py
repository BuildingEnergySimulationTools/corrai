import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary


class MyProblem(ElementwiseProblem):
    def __init__(
        self, parameters, obj_func_list, func_list, function_names, constraint_names
    ):
        self.parameters = parameters
        self.obj_func_list = obj_func_list  # objects with methods functions
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
    def __init__(self, parameters, mf_list, pyf_list, function_names, constraint_names):
        global var
        self.parameters = parameters
        self.mf_list = mf_list
        self.pyf_list = pyf_list
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
            xl=np.array([p["interval"][0] for p in parameters]),
            xu=np.array([p["interval"][-1] for p in parameters]),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        res = pd.concat(
            [m.function(X) for m in self.mf_list] + [pyf(X) for pyf in self.pyf_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraint_names])
