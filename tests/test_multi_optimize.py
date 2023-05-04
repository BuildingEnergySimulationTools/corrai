import pandas as pd
import numpy as np
from pathlib import Path

from corrai.multi_optimize import MyProblem, MyMixedProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA


PACKAGE_DIR = Path(__file__).parent / "TestLib"


def py_func_rosen(x_dict):
    return pd.Series(
        (1 - x_dict['x']) ** 2 + 100 * (x_dict['y'] - x_dict['x'] ** 2) ** 2,
        index=["f1"])


class MyObject_BinhandKorn1:
    def function(self, x):
        f1 = 4 * x['x'] ** 2 + 4 * x['y'] ** 2
        f2 = (x['x'] - 5) ** 2 + (x['y'] - 5) ** 2
        g1 = (x['x'] - 5) ** 2 + x['y'] ** 2 - 25
        return pd.Series([f1, f2, g1], index=["f1", "f2", "g1"])


class MyObject_BinhandKorn2:
    def function(self, x):
        g2 = 7.7 - (x['x'] - 8) ** 2 - (x['y'] + 3) ** 2
        return pd.Series([g2], index=["g2"])

class MyObject_mixed:
    def function(self, x):
        f1 = x['z'] ** 2 + x['y'] ** 2
        f2 = (x['z'] + 2) ** 2 + (x['y']-1) ** 2

        if x['b']:
            f2 = 100 * f2

        if x['x'] == "multiply":
            f2 = 10 * f2

        return pd.Series([f1, f2], index=["f1", "f2"])



parameters = [
    {"name": "x", "interval": (-2, 10)},
    {"name": "y", "interval": (-2, 10)},
]


class TestMyProblem():
    def test_myproblem_simple(self):
        problem = MyProblem(
            parameters=parameters,
            obj_func_list=[],
            func_list=[py_func_rosen],
            function_names=["f1"],
            constraint_names=[]
        )

        algorithm = DE(
            pop_size=100,
            sampling=LHS(),
            CR=0.3,
            jitter=False
        )

        res = minimize(problem,
                       algorithm,
                       seed=1,
                       verbose=False)

        np.testing.assert_allclose(res.X, np.array([1, 1]), rtol=0.01)

    def test_myproblem_twoobjectsfunction(self):
        param = [
            {"name": "x", "interval": (0, 5)},
            {"name": "y", "interval": (0, 3)},
        ]

        obj1 = MyObject_BinhandKorn1()
        obj2 = MyObject_BinhandKorn2()

        problem = MyProblem(
            parameters=param,
            obj_func_list=[obj1, obj2],
            func_list=[],
            function_names=["f1", "f2"],
            constraint_names=["g1", "g2"]
        )

        res = minimize(problem,
                       NSGA2(pop_size=10),
                       ('n_gen', 10),
                       seed=1,
                       verbose=False)

        np.testing.assert_almost_equal(res.X,
                                       np.array([[0.08320695, 0.05947538],
                                                 [3.11192632, 2.89802109],
                                                 [2.21805962, 2.98362625],
                                                 [2.43878853, 2.22367131],
                                                 [0.52530151, 0.08202677],
                                                 [0.76324568, 0.9977904],
                                                 [1.36107106, 0.97739205],
                                                 [1.86374614, 1.50234103],
                                                 [2.75558726, 2.98367642],
                                                 [1.68980871, 0.95713564]]))

    def test_myproblem_integers(self):
        problem = MyProblem(
            parameters=parameters,
            obj_func_list=[],
            func_list=[py_func_rosen],
            function_names=["f1"],
            constraint_names=[]
        )

        # for  integer variables only --> no need to specify type integer
        method = GA(pop_size=20,
                    sampling=IntegerRandomSampling(),
                    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )

        res = minimize(problem,
                       method,
                       termination=('n_gen', 40),
                       seed=1,
                       verbose=False)

        assert np.allclose(res.X, np.array([1, 1]), rtol=0, atol=0)

    def test_myproblem_mixed(self):

        param = [
            {"name": "b", "interval": (), 'type': "Binary"},
            {"name": "x", "interval": ("nothing", "multiply"), 'type': "Choice"},
            {"name": "y", "interval": (-2, 2.5), 'type': "Integer"},
            {"name": "z", "interval": (5, -5), 'type': "Real"},
        ]

        obj = MyObject_mixed()

        problem = MyMixedProblem(
            parameters=param,
            obj_func_list=[obj],
            func_list=[],
            function_names=["f1", "f2"],
            constraint_names=[]
        )

        algorithm = MixedVariableGA(pop_size=10, survival=RankAndCrowdingSurvival())

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 10),
                       seed=1,
                       verbose=False)

        np.array_equal(res.X,
                       np.array([{'b': False, 'x': 'nothing', 'y': 0, 'z': -0.9193284996322444},
                                 {'b': True, 'x': 'nothing', 'y': 0, 'z': -0.031853402754111526},
                                 {'b': False, 'x': 'multiply', 'y': 0, 'z': 0.11304433422108318},
                                 {'b': False, 'x': 'nothing', 'y': 0, 'z': -2.9032762721742804},
                                 {'b': False, 'x': 'nothing', 'y': 0, 'z': -0.5998932514007078},
                                 {'b': True, 'x': 'nothing', 'y': 0, 'z': -0.10011235296152621}])
                       )

