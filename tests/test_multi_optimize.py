import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error

from corrai.multi_optimize import MyProblem, MyMixedProblem
from corrai.metrics import nmbe, cv_rmse
from modelitool.functiongenerator import ModelicaFunction
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from modelitool.simulate import Simulator
# from pymoo.core.variable import Integer, Real, Choice, Binary

PACKAGE_DIR = Path(__file__).parent / "TestLib"


def py_func_rosen(x_dict):
    return pd.Series(
        (1 - x_dict['x.k']) ** 2 + 100 * (x_dict['y.k'] - x_dict['x.k'] ** 2) ** 2,
        index=["f1"])


class MyObject1:
    def function(self, x):
        f1 = 4 * x['x.k'] ** 2 + 4 * x['y.k'] ** 2
        f2 = (x['x.k'] - 5) ** 2 + (x['y.k'] - 5) ** 2
        g1 = (x['x.k'] - 5) ** 2 + x['y.k'] ** 2 - 25
        return pd.Series([f1, f2, g1], index=["f1", "f2", "g1"])


class MyObject2:
    def function(self, x):
        g2 = 7.7 - (x['x.k'] - 8) ** 2 - (x['y.k'] + 3) ** 2
        return pd.Series([g2], index=["g2"])


simu_options = {
    "startTime": 0,
    "stopTime": 1,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl",
}

simu = Simulator(
    model_path="TestLib.BinhandKorn",
    package_path=PACKAGE_DIR / "package.mo",
    simulation_options=simu_options,
    output_list=["f1", "f2", "g1", "g2"],
    simulation_path=None,
    lmodel=["Modelica"],
)

x_dict = {"x.k": 2, "y.k": 2}

parameters = [
    {"name": "x.k", "interval": (-2, 10)},
    {"name": "y.k", "interval": (-2, 10)},
]

algorithm = DE(
    pop_size=100,
    sampling=LHS(),
    CR=0.3,
    jitter=False
)


class TestMyProblem():
    def test_myproblem_simple(self):

        problem = MyProblem(
            parameters=parameters,
            obj_func_list=[],
            func_list=[py_func_rosen],
            function_names=["f1"],
            constraint_names=[]
        )

        res = minimize(problem,
                       algorithm,
                       seed=1,
                       verbose=False)

        np.testing.assert_allclose(res.X, np.array([1, 1]), rtol=0.01)

    def test_myproblem_twoobjectsfunction(self):
        param = [
            {"name": "x.k", "interval": (0, 5)},
            {"name": "y.k", "interval": (0, 3)},
        ]

        obj1 = MyObject1()
        obj2 = MyObject2()

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
                       verbose=True)

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

    def test_myproblem_mixed(self): #à finir

        parameters = [
            {"name": "x", "interval": (-2, 10), 'type': "Integer"},
            {"name": "y", "interval": (-2, 11.5), 'type': "Integer"}
        ]