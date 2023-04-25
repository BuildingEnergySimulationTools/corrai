import pandas as pd
import numpy as np
from corrai.multi_optimize import MyProblem, MyMixedProblem

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real, Choice, Binary


class TestMyProblem(ElementwiseProblem):
    def test_myproblem(self):
        parameters = [
            {"name": "x1", "interval": [0, 10]},
            {"name": "x2", "interval": [-10, 10]},
        ]
        obj_func_list = [lambda x: pd.Series({"f1": x["x1"] ** 2 + x["x2"] ** 2}),
                         lambda x: pd.Series({"f2": (x["x1"] - 1) ** 2 + x["x2"] ** 2}),
                         ]
        func_list = [lambda x: x["x1"] - x["x2"]]
        function_names = ["f1", "f2"]
        constraint_names = ["g1"]
        problem = MyProblem(parameters, obj_func_list, func_list, function_names, constraint_names)
        x = np.array([1, 2])
        out = problem.evaluate(x)
        assert np.allclose(out["F"], [5, 2]), f"Unexpected F value: {out['F']}"
        assert np.allclose(out["G"], [-1]), f"Unexpected G value: {out['G']}"

        # Test with more complex objective functions and constraints
        parameters = [{"name": "x1", "interval": [-5, 5]},
                      {"name": "x2", "interval": [-5, 5]},
                      {"name": "x3", "interval": [-5, 5]},
                      ]
        obj_func_list = [lambda x: pd.Series({"f1": x["x1"] ** 2 + x["x2"] ** 2 + x["x3"] ** 2}),
                         lambda x: pd.Series({"f2": (x["x1"] - 1) ** 2 + (x["x2"] - 1) ** 2 + (x["x3"] - 1) ** 2}),
                         ]
        func_list = [lambda x: x["x1"] + x["x2"] + x["x3"] - 5,
                     lambda x: (x["x1"] - 3) ** 2 + x["x2"] + x["x3"] - 15,
                     ]
        function_names = ["f1", "f2"]
        constraint_names = ["g1", "g2"]
        problem = MyProblem(parameters, obj_func_list, func_list, function_names, constraint_names)
        x = np.array([0, 0, 0])
        out = problem.evaluate(x)
        assert np.allclose(out["F"], [0, 3]), f"Unexpected F value: {out['F']}"
        assert np.allclose(out["G"], [-5, -8]), f"Unexpected G value: {out['G']}"