import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

from corrai.base.model import Model
from corrai.base.parameter import Parameter
from corrai.base.objfunctions import ObjectiveFunction


class IshigamiTwoOutputs(Model):
    def simulate(self, property_dict, simulation_options):
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

    def save(self, file_path):
        pass


class RosenModel(Model):
    def simulate(self, property_dict, simulation_options):
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

    def save(self, file_path):
        pass


PARAMETERS = [
    {Parameter.NAME: "x", Parameter.INTERVAL: (0.0, 3.0), Parameter.INIT_VALUE: 2.0},
    {Parameter.NAME: "y", Parameter.INTERVAL: (0.0, 3.0), Parameter.INIT_VALUE: 2.0},
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
            {
                "res1": [6.79, 6.79],
                "res2": [5.50, 5.50],
            },
            index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
        )

        python_model = IshigamiTwoOutputs()
        obj_func = ObjectiveFunction(
            model=python_model,
            simulation_options=SIMU_OPTIONS,
            param_list=PARAMETERS,
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
            param_list=PARAMETERS,
            indicators_config={"res1": (np.mean, None), "res2": (np.mean, None)},
        )

        assert obj_func.bounds == [(0, 3.0), (0, 3.0)]
        assert obj_func.init_values == [2.0, 2.0]

    def test_scipy_obj_function(self):
        rosen = RosenModel()
        obj_func = ObjectiveFunction(
            model=rosen,
            simulation_options=SIMU_OPTIONS,
            param_list=PARAMETERS,
            indicators_config={"res": (np.mean, None)},
        )

        res = minimize(
            obj_func.scipy_obj_function,
            obj_func.init_values,
            method="Nelder-Mead",
            tol=1e-6,
        )

        np.testing.assert_allclose(res.x, np.array([1, 1]), rtol=0.01)
