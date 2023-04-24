import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice


class ModelicaFunction:
    def __init__(self,
                 simulator,
                 param_dict,
                 agg_method=np.mean,r
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


class MyMixedProblem(ElementwiseProblem):
    def __init__(self, parameters, mf_list, pyf_list, function_names, constraints_names):
        global var
        self.parameters = parameters
        self.mf_list = mf_list
        self.pyf_list = pyf_list
        self.function_names = function_names
        self.constraints_names = constraints_names
        variable_string = {}

        for k, v in parameters.items():
            name, bounds, vtype = k, v[:2], v[2]
            if vtype == "Integer":
                var = Integer(bounds=bounds)
            elif vtype == "Real":
                var = Real(bounds=bounds)
            # elif vtype == "Choice":
            #     var = Choice(options=bounds)
            # variable_string[name] = var

        super().__init__(vars=variable_string,
                         n_var=len(parameters),
                         n_obj=len(function_names),
                         n_ieq_constr=len(constraints_names),
                         xl=np.array([val[0] for val in parameters.values()]),
                         xu=np.array([val[1] for val in parameters.values()]),
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        #current_param = {key: val for key, val in zip(self.parameters.keys(), x)}
        res = pd.concat(
            [m.function(X) for m in self.mf_list] +
            [pyf(X) for pyf in self.pyf_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraints_names])

class MyProblem(ElementwiseProblem):
    def __init__(self, parameters, mf_list, pyf_list, function_names, constraints_names):
        self.parameters = parameters
        self.mf_list = mf_list
        self.pyf_list = pyf_list
        self.function_names = function_names
        self.constraints_names = constraints_names
        variable_string = {}



        super().__init__(n_var=len(parameters),
                         n_obj=len(function_names),
                         n_ieq_constr=len(constraints_names),
                         xl=np.array([val[0] for val in parameters.values()]),
                         xu=np.array([val[1] for val in parameters.values()]),
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        current_param = {key: val for key, val in zip(self.parameters.keys(), x)}
        res = pd.concat(
            [m.function(current_param) for m in self.mf_list] +
            [pyf(current_param) for pyf in self.pyf_list]
        )

        out["F"] = list(res[self.function_names])
        out["G"] = list(res[self.constraints_names])

