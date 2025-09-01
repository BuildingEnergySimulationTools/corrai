import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from corrai.optimize import Problem, check_duplicate_params, ObjectiveFunction
from corrai.base.parameter import Parameter
from corrai.base.model import Model

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.termination import get_termination

from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.optimize import minimize as scipy_minimize


PACKAGE_DIR = Path(__file__).parent / "TestLib"


def py_func_rosen(params: dict[str, float]):
    return {"f1": (1 - params["x"]) ** 2 + 100 * (params["y"] - params["x"] ** 2) ** 2}


class MyObjectBinhandKorn1:
    def function(self, parameter_value_pairs):
        params = {p.name: v for p, v in parameter_value_pairs}
        f1 = 4 * params["x"] ** 2 + 4 * params["y"] ** 2
        f2 = (params["x"] - 5) ** 2 + (params["y"] - 5) ** 2
        g1 = (params["x"] - 5) ** 2 + (params["y"] - 5) ** 2 - 25
        return {"f1": f1, "f2": f2, "g1": g1}


class MyObjectBinhandKorn2:
    def function(self, parameter_value_pairs):
        params = {p.name: v for p, v in parameter_value_pairs}
        g2 = 7.7 - (params["x"] - 8) ** 2 - (params["y"] + 3) ** 2
        return {"g2": g2}


class MyObjectMixed:
    def function(self, parameter_value_pairs):
        params = {p.name: v for p, v in parameter_value_pairs}
        f1 = params["z"] ** 2 + params["y"] ** 2
        f2 = np.sqrt(params["z"] ** 2 + params["y"] ** 2)
        return {"f1": f1, "f2": f2}


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
        parameters = [
            Parameter(name="x", interval=(0, 5)),
            Parameter(name="y", interval=(0, 3)),
        ]
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

        eval1 = MyObjectBinhandKorn1()
        eval2 = MyObjectBinhandKorn2()

        problem = Problem(
            parameters=param,
            evaluators=[eval1, eval2],
            objective_ids=["f1", "f2"],
            constraint_ids=["g1", "g2"],
        )
        res = minimize(
            problem, NSGA2(pop_size=10), ("n_gen", 10), seed=1, verbose=False
        )

        assert res.F.shape[0] == 10
        assert res.F.shape[1] == 2
        assert np.all(res.G <= 1e-6)
        assert np.all(res.F >= 0)

    def test_Problem_mixed(self):
        param = [
            Parameter(name="b", values=(0, 1), ptype="Binary"),
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

        ref_f = np.array([[11.060919712929891, 3.325796]])
        ref_g = np.array([[]])

        np.testing.assert_allclose(to_test["F"], ref_f, rtol=1e-6)
        assert np.array_equal(to_test["G"], ref_g)

    def test_warning_error(self):
        # No evaluators
        parameters = [
            Parameter(name="x", interval=(-2, 10)),
            Parameter(name="y", interval=(-2, 10)),
        ]
        with pytest.raises(ValueError):
            Problem(
                parameters=parameters,
                objective_ids=["f1"],
            )

    def test_Problem_mixed_variable_ga(self):
        param = [
            Parameter(name="b", values=(0, 1), ptype="Binary"),
            Parameter(name="x", values=("nothing", "multiply"), ptype="Choice"),
            Parameter(name="y", interval=(-2, 2.5), ptype="Integer"),
            Parameter(name="z", interval=(-5, 5), ptype="Real"),
        ]

        obj = MyObjectMixed()

        problem = Problem(
            parameters=param,
            evaluators=[obj],
            objective_ids=["f1"],
        )

        algorithm = MixedVariableGA(pop_size=20)
        termination = get_termination("n_gen", 5)
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        assert res.F.shape[0] == 1


PARAMETERS = [
    Parameter(name="x", interval=(0.0, 3.0), init_value=2.0, model_property="x"),
    Parameter(name="y", interval=(0.0, 3.0), init_value=2.0, model_property="y"),
]

X_DICT = {"x": 2, "y": 2}

X_PAIRS = [(p, X_DICT[p.name]) for p in PARAMETERS]

DATASET = pd.DataFrame(
    {"meas1": [6, 2], "meas2": [14, 1]},
    index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
)

SIMU_OPTIONS = {
    "start": "2023-01-01 00:00:00",
    "end": "2023-01-01 00:00:01",
    "timestep": "s",
}


class IshigamiTwoOutputs(Model):
    def simulate(
        self,
        property_dict=None,
        simulation_options=None,
        simulation_kwargs=None,
        **kwargs,
    ) -> pd.DataFrame:
        A1 = 7.0
        B1 = 0.1
        A2 = 5.0
        B2 = 0.5

        default_parameters = {"x": 1, "y": 1, "z": 1}
        parameters = {**default_parameters, **(property_dict or {})}

        x = parameters["x"]
        y = parameters["y"]
        z = parameters["z"]

        start_date = pd.Timestamp(simulation_options["start"])
        end_date = pd.Timestamp(simulation_options["end"])
        timestep = simulation_options.get("timestep", "h")

        times = pd.date_range(start=start_date, end=end_date, freq=timestep)

        res1 = [np.sin(x) + A1 * np.sin(y) ** 2 + B1 * z**4 * np.sin(x) for _ in times]
        res2 = [np.sin(x) + A2 * np.sin(y) ** 2 + B2 * z**4 * np.sin(x) for _ in times]

        return pd.DataFrame({"res1": res1, "res2": res2}, index=times)


class RosenModel(Model):
    def simulate(
        self, property_dict=None, simulation_options=None, simulation_kwargs=None
    ) -> pd.DataFrame:
        default_parameters = {"x": 1, "y": 1}
        parameters = {**default_parameters, **property_dict}

        x, y = parameters["x"], parameters["y"]

        start_date = pd.Timestamp(simulation_options["start"])
        end_date = pd.Timestamp(simulation_options["end"])
        timestep = simulation_options.get("timestep", "h")
        times = pd.date_range(start=start_date, end=end_date, freq=timestep)

        res = [(1 - x) ** 2 + 100 * (y - x**2) ** 2 for _ in times]
        return pd.DataFrame({"res": res}, index=times)


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
                "res1": (mean_squared_error, DATASET["meas1"]),
                "res2": (mean_absolute_error, DATASET["meas2"]),
            },
        )

        res = obj_func.function(parameter_value_pairs=X_PAIRS)

        np.testing.assert_allclose(
            np.array([res["res1"], res["res2"]]),
            (
                mean_squared_error(expected_model_res["res1"], DATASET["meas1"]),
                mean_absolute_error(expected_model_res["res2"], DATASET["meas2"]),
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

        res = scipy_minimize(
            obj_func.scipy_obj_function,
            obj_func.init_values,
            method="Nelder-Mead",
            tol=1e-6,
        )

        np.testing.assert_allclose(res.x, np.array([1.0, 1.0]), rtol=0.01)
