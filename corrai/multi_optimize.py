import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary


class ModelicaFunction:
    def __init__(
        self,
        simulator,
        param_dict,
        indicators=None,
        agg_methods_dict=None,
        reference_dict=None,
        reference_df=None,
    ):
        self.simulator = simulator
        self.param_dict = param_dict
        if indicators is None:
            self.indicators = simulator.output_list
        else:
            self.indicators = indicators
        self.agg_methods_dict = agg_methods_dict or [np.mean for _ in self.indicators]
        self.reference_dict = reference_dict
        self.reference_df = reference_df

    def function(self, x_dict):
        temp_dict = {param["name"]: x_dict[param["name"]] for param in self.param_dict}
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
