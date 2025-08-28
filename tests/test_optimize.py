import pandas as pd
import numpy as np
from pathlib import Path

import pytest
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

from corrai.base.model import Model
from corrai.optimize import Problem, check_duplicate_params
from corrai.base.parameter import Parameter

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

from corrai.surrogate import ObjectiveFunction

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
    Parameter(name="x", interval=(-2, 10)),
    Parameter(name="y", interval=(-2, 10)),
]


class TestProblem:
    def test_duplicates(self):
        parameters = [
            Parameter(name="x", interval=(-2, 10)),
            Parameter(name="y", interval=(-2, 10)),
            Parameter(name="x", interval=(-2, 12)),
        ]

        with pytest.raises(ValueError) as excinfo:
            check_duplicate_params(parameters)

        assert "Duplicate parameter name: x" in str(excinfo.value)

    def test_problem_simple(self):
        problem = Problem(
            parameters=parameters,
            evaluators=[py_func_rosen],
            objective_ids=["f1"],
        )
        algorithm = DE(pop_size=100, sampling=LHS(), CR=0.3, jitter=False)
        res = minimize(problem, algorithm, seed=1, verbose=False)

        np.testing.assert_allclose(res.X, np.array([1, 1]), rtol=0.01)

    def test_Problem_twoobjectsfunction(self):
        param = [
            Parameter(name="x", interval=(0, 5)),
            Parameter(name="y", interval=(0, 3)),
        ]

        obj1 = MyObjectBinhandKorn1()
        obj2 = MyObjectBinhandKorn2()

        problem = Problem(
            parameters=param,
            evaluators=[obj1, obj2],
            objective_ids=["f1", "f2"],
            constraint_ids=["g1", "g2"],
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

    parameters = [
        Parameter(name="x", interval=(-2, 10)),
        Parameter(name="y", interval=(-2, 10)),
    ]

    def test_Problem_mixed(self):
        param = [
            Parameter(
                name="b",
                values=(0, 1),
                ptype="Binary",
            ),
            Parameter(name="x", values=("nothing", "multiply"), ptype="Choice"),
            Parameter(name="y", interval=(-2, 2.5), ptype="Integer"),
            Parameter(name="z", interval=(-5, 5), ptype="Real"),
        ]

        obj = MyObjectMixed()

        problem = Problem(
            parameters=param,
            evaluators=[obj],
            objective_ids=["f1", "f2"],
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
            Problem(
                parameters=parameters,
                objective_ids=["f1"],
            )


class IshigamiTwoOutputs(Model):
    def simulate(
        self, property_dict=None, simulation_options=None, **kwargs
    ) -> pd.DataFrame:
        A1 = 7.0
        B1 = 0.1
        A2 = 5.0
        B2 = 0.5

        default_parameters = {
            "x": 1,
            "y": 1,
            "z": 1,
        }

        parameters = {**default_parameters, **property_dict}

        x = parameters["x"]
        y = parameters["y"]
        z = parameters["z"]

        start_date = pd.Timestamp(simulation_options["start"])
        end_date = pd.Timestamp(simulation_options["end"])
        timestep = simulation_options.get("timestep", "h")

        times = pd.date_range(start=start_date, end=end_date, freq=timestep)

        res1 = []
        res2 = []

        for _ in times:
            res1.append(np.sin(x) + A1 * np.sin(y) ** 2 + B1 * z**4 * np.sin(x))
            res2.append(np.sin(x) + A2 * np.sin(y) ** 2 + B2 * z**4 * np.sin(x))

        results_df = pd.DataFrame(
            {"time": times, "res1": res1, "res2": res2}
        ).set_index("time")

        return results_df


class RosenModel(Model):
    def simulate(
        self, property_dict=None, simulation_options=None, simulation_kwargs=None
    ) -> pd.DataFrame:
        default_parameters = {
            "x": 1,
            "y": 1,
        }

        parameters = {**default_parameters, **property_dict}

        x = parameters["x"]
        y = parameters["y"]

        start_date = pd.Timestamp(simulation_options["start"])
        end_date = pd.Timestamp(simulation_options["end"])
        timestep = simulation_options.get("timestep", "h")

        times = pd.date_range(start=start_date, end=end_date, freq=timestep)

        res = []

        for _ in times:
            result = (1 - x) ** 2 + 100 * (y - x**2) ** 2
            res.append(result)

        results_df = pd.DataFrame({"time": times, "res": res}).set_index("time")

        return results_df


PARAMETERS = [
    Parameter(name="x", interval=(0.0, 3.0), init_value=2.0, model_property="x"),
    Parameter(name="y", interval=(0.0, 3.0), init_value=2.0, model_property="y"),
]
X_DICT = {"x": 2, "y": 2}
dataset = pd.DataFrame(
    {
        "meas1": [6, 2],
        "meas2": [14, 1],
    },
    index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
)
SIMU_OPTIONS = {
    "start": "2023-01-01 00:00:00",
    "end": "2023-01-01 00:00:01",
    "timestep": "s",
}


class TestObjectiveFunction:
    def test_function_indicators(self):
        expected_model_res = pd.DataFrame(
            {"res1": [6.79, 6.79], "res2": [5.50, 5.50]},
            index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
        )

        python_model = IshigamiTwoOutputs()
        obj_func = ObjectiveFunction(
            model=python_model,
            simulation_options=SIMU_OPTIONS,
            parameters=PARAMETERS,
            indicators_config={
                "res1": (mean_squared_error, dataset["meas1"]),
                "res2": (mean_absolute_error, dataset["meas2"]),
            },
        )

        res = obj_func.function(X_DICT)

        np.testing.assert_allclose(
            np.array([res["res1"], res["res2"]]),
            (
                mean_squared_error(expected_model_res["res1"], dataset["meas1"]),
                mean_absolute_error(expected_model_res["res2"], dataset["meas2"]),
            ),
            rtol=0.01,
        )

    def test_bounds_and_init_values(self):
        python_model = IshigamiTwoOutputs()
        obj_func = ObjectiveFunction(
            model=python_model,
            simulation_options=SIMU_OPTIONS,
            parameters=PARAMETERS,
            indicators_config={"res1": (np.mean, None), "res2": (np.mean, None)},
        )

        assert obj_func.bounds == [(0.0, 3.0), (0.0, 3.0)]
        assert obj_func.init_values == [2.0, 2.0]

    def test_scipy_obj_function(self):
        rosen = RosenModel()
        obj_func = ObjectiveFunction(
            model=rosen,
            simulation_options=SIMU_OPTIONS,
            parameters=PARAMETERS,
            indicators_config={"res": (np.mean, None)},
        )

        res = minimize(
            obj_func.scipy_obj_function,
            obj_func.init_values,
            method="Nelder-Mead",
            tol=1e-6,
        )

        np.testing.assert_allclose(res.x, np.array([1.0, 1.0]), rtol=0.01)
