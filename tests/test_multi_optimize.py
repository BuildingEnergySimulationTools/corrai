import pandas as pd
import numpy as np
from pathlib import Path

import pytest

from corrai.multi_optimize import MyProblem, MyMixedProblem, _check_duplicate_params
from corrai.base.parameter import Parameter

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

PACKAGE_DIR = Path(__file__).parent / "TestLib"


def py_func_rosen(x_dict):
    return pd.Series(
        (1 - x_dict["x"]) ** 2 + 100 * (x_dict["y"] - x_dict["x"] ** 2) ** 2,
        index=["f1"],
    )


class MyObjectBinhandKorn1:
    def function(self, x):
        f1 = 4 * x["x"] ** 2 + 4 * x["y"] ** 2
        f2 = (x["x"] - 5) ** 2 + (x["y"] - 5) ** 2
        g1 = (x["x"] - 5) ** 2 + x["y"] ** 2 - 25
        return {"f1": f1, "f2": f2, "g1": g1}


class MyObjectBinhandKorn2:
    def function(self, x):
        g2 = 7.7 - (x["x"] - 8) ** 2 - (x["y"] + 3) ** 2
        return {"g2": g2}


class MyObjectMixed:
    def function(self, x):
        f1 = x["z"] ** 2 + x["y"] ** 2
        f2 = (x["z"] + 2) ** 2 + (x["y"] - 1) ** 2

        if x["b"]:
            f2 = 100 * f2

        if x["x"] == "multiply":
            f2 = 10 * f2

        return {"f1": f1, "f2": f2}


parameters = [
    {Parameter.NAME: "x", Parameter.INTERVAL: (-2, 10)},
    {Parameter.NAME: "y", Parameter.INTERVAL: (-2, 10)},
]


class TestMyProblem:
    def test_duplicates(self):
        parameters = [
            {Parameter.NAME: "x", Parameter.INTERVAL: (-2, 10)},
            {Parameter.NAME: "y", Parameter.INTERVAL: (-2, 10)},
            {Parameter.NAME: "x", Parameter.INTERVAL: (-2, 12)},
        ]

        with pytest.raises(ValueError) as excinfo:
            _check_duplicate_params(parameters)

        assert "Duplicate parameter name: x" in str(excinfo.value)

    def test_myproblem_simple(self):
        problem = MyProblem(
            parameters=parameters,
            obj_func_list=[],
            func_list=[py_func_rosen],
            function_names=["f1"],
            constraint_names=[],
        )

        algorithm = DE(pop_size=100, sampling=LHS(), CR=0.3, jitter=False)

        res = minimize(problem, algorithm, seed=1, verbose=False)

        np.testing.assert_allclose(res.X, np.array([1, 1]), rtol=0.01)

    def test_myproblem_twoobjectsfunction(self):
        param = [
            {Parameter.NAME: "x", Parameter.INTERVAL: (0, 5)},
            {Parameter.NAME: "y", Parameter.INTERVAL: (0, 3)},
        ]

        obj1 = MyObjectBinhandKorn1()
        obj2 = MyObjectBinhandKorn2()

        problem = MyProblem(
            parameters=param,
            obj_func_list=[obj1, obj2],
            func_list=[],
            function_names=["f1", "f2"],
            constraint_names=["g1", "g2"],
        )

        res = minimize(
            problem, NSGA2(pop_size=10), ("n_gen", 10), seed=1, verbose=False
        )

        assert (
            np.allclose(
                res.X,
                np.array(
                    [
                        [0.08320695, 0.05947538],
                        [3.11192632, 2.89802109],
                        [2.21805962, 2.98362625],
                        [2.43878853, 2.22367131],
                        [0.52530151, 0.08202677],
                        [0.76324568, 0.9977904],
                        [1.36107106, 0.97739205],
                        [1.86374614, 1.50234103],
                        [2.75558726, 2.98367642],
                        [1.68980871, 0.95713564],
                    ]
                ),
            )
            or np.allclose(
                res.X,
                np.array(
                    [
                        [3.0416415, 2.4492151],
                        [0.3985999, 0.2485888],
                        [2.7348734, 2.0720593],
                        [0.7103709, 1.0476442],
                        [1.9838374, 1.0405508],
                        [1.7628016, 2.0258502],
                        [3.0416415, 2.4184421],
                        [1.9694239, 1.5152536],
                        [1.7935434, 2.5076411],
                        [0.8009073, 1.0476442],
                    ]
                ),
            )
            or np.allclose(
                res.X,
                np.array(
                    [
                        [0.18151636138876737, 0.28557238133925045],
                        [2.069443625467415, 2.6572565698098187],
                        [1.9811633819548806, 1.6583069865011475],
                        [0.9873972738138687, 0.9484467702355981],
                        [1.8852248925301685, 2.3568221936620244],
                        [1.0034518809196942, 2.020194592503196],
                        [0.9705035535546149, 1.532438627240462],
                        [2.069443625467415, 2.2702029688309477],
                        [0.9787349877630491, 1.8926611083490412],
                        [0.8579146017708214, 1.6841984531799503],
                    ]
                ),
            )
        )

    def test_myproblem_integers(self):
        problem = MyProblem(
            parameters=parameters,
            obj_func_list=[],
            func_list=[py_func_rosen],
            function_names=["f1"],
            constraint_names=[],
        )

        # for  integer variables only --> no need to specify type integer
        method = GA(
            pop_size=20,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        res = minimize(
            problem, method, termination=("n_gen", 40), seed=1, verbose=False
        )

        assert np.allclose(res.X, np.array([1, 1]), rtol=0, atol=0)

    def test_myproblem_mixed(self):
        param = [
            {Parameter.NAME: "b", Parameter.INTERVAL: (), Parameter.TYPE: "Binary"},
            {
                Parameter.NAME: "x",
                Parameter.INTERVAL: ("nothing", "multiply"),
                Parameter.TYPE: "Choice",
            },
            {
                Parameter.NAME: "y",
                Parameter.INTERVAL: (-2, 2.5),
                Parameter.TYPE: "Integer",
            },
            {Parameter.NAME: "z", Parameter.INTERVAL: (-5, 5), Parameter.TYPE: "Real"},
        ]

        obj = MyObjectMixed()

        problem = MyMixedProblem(
            parameters=param,
            obj_func_list=[obj],
            func_list=[],
            function_names=["f1", "f2"],
            constraint_names=[],
        )

        to_test = problem.evaluate(
            [[False, "nothing", -1, -3.171895287195006]],
            return_as_dictionary=True,
        )

        ref_f = np.array([[11.060919712929891, 5.373338564149866]])
        ref_g = np.array([[]])

        assert np.array_equal(to_test["F"], ref_f)
        assert np.array_equal(to_test["G"], ref_g)

    def test_warning_error(self):
        with pytest.raises(ValueError):
            MyProblem(
                parameters=parameters,
                obj_func_list=[],
                func_list=[],
                function_names=["f1"],
                constraint_names=[],
            )

        with pytest.raises(ValueError):
            MyMixedProblem(
                parameters=parameters,
                obj_func_list=[],
                func_list=[],
                function_names=["f1"],
                constraint_names=[],
            )
