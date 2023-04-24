import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice


class ModelicaFunction:
    def __init__(self,
                 simulator,
                 param_dict,
                 agg_method=np.mean
                 ):
        self.simulator = simulator
        self.param_dict = param_dict
        self.agg_method = agg_method
        self.indicators = simulator.output_list

    def function(self, x_dict):
        temp_dict = {
            key: x_dict[key] for key in self.param_dict.keys()
        }
        self.simulator.set_param_dict(temp_dict)
        self.simulator.simulate()
        res = self.simulator.get_results()
        return self.agg_method(res, axis=0)


class MyProblem(ElementwiseProblem):
    def __init__(self, parameters, mf_list, pyf_list, function_names, constraint_names):
        self.parameters = parameters
        self.mf_list = mf_list
        self.pyf_list = pyf_list
        self.function_names = function_names
        self.constraints_names = constraint_names
        variable_string = {}

        super().__init__(n_var=len(parameters),
                         n_obj=len(function_names),
                         n_ieq_constr=len(constraint_names),
                         xl=np.array([p['bounds'][0] for p in parameters]),
                         xu=np.array([p['bounds'][-1] for p in parameters]),
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        current_param = {param['name']: val for param, val in zip(self.parameters, x)}
        res = pd.concat(
            [m.function(current_param) for m in self.mf_list] +
            [pyf(current_param) for pyf in self.pyf_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraints_names])


class MyMixedProblem(ElementwiseProblem):
    def __init__(self, parameters, mf_list, pyf_list, function_names, constraint_names):
        global var
        self.parameters = parameters
        self.mf_list = mf_list
        self.pyf_list = pyf_list
        self.function_names = function_names
        self.constraints_names = constraint_names
        variable_string = {}

        for param in parameters:
            name, bounds, vtype = param['name'], param['bounds'], param['type']
            if vtype == "Integer":
                var = Integer(bounds=bounds)
            elif vtype == "Real":
                var = Real(bounds=bounds)
            elif vtype == "Choice":
                var = Choice(options=bounds)
            variable_string[name] = var

        super().__init__(vars=variable_string,
                         n_var=len(parameters),
                         n_obj=len(function_names),
                         n_ieq_constr=len(constraint_names),
                         xl=np.array([p['bounds'][0] for p in parameters]),
                         xu=np.array([p['bounds'][-1] for p in parameters]),
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        res = pd.concat(
            [m.function(X) for m in self.mf_list] +
            [pyf(X) for pyf in self.pyf_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraints_names])
