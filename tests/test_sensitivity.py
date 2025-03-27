import numpy as np
import pytest
import pandas as pd
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error, mean_absolute_error

from corrai.base.model import Model
from corrai.base.parameter import Parameter
from corrai.sensitivity import Method
from corrai.sensitivity import ObjectiveFunction
from corrai.sensitivity import SAnalysis
from corrai.sensitivity import (
    plot_sobol_st_bar,
    plot_morris_st_bar,
    plot_morris_scatter,
    plot_sample,
)

from tests.resources.pymodels import Ishigami

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 05:00:00",
    "timestep": "h",
}

PARAMETER_LIST = [
    {
        Parameter.NAME: "x1",
        Parameter.INTERVAL: (-3.14159265359, 3.14159265359),
        Parameter.TYPE: "Real",
    },
    {
        Parameter.NAME: "x2",
        Parameter.INTERVAL: (-3.14159265359, 3.14159265359),
        Parameter.TYPE: "Real",
    },
    {
        Parameter.NAME: "x3",
        Parameter.INTERVAL: (-3.14159265359, 3.14159265359),
        Parameter.TYPE: "Real",
    },
]


class SobolResult:
    def __init__(self, sobol_dict):
        self.sobol_dict = sobol_dict

    def to_df(self):
        df = [
            pd.DataFrame(
                self.sobol_dict,
                index=[f"param{i + 1}" for i in range(len(self.sobol_dict["ST"]))],
            )
        ]
        return df


def sobol_res_mock():
    sobol_dict = {
        "S1": [-2.5, 0.8, 3],
        "S1_conf": [1e-2, 1e-2, 1e-2],
        "ST": [0.5, 0.7, 0.15],
        "ST_conf": [1e-2, 3e-2, 5e-2],
    }
    return SobolResult(sobol_dict)


class MorrisResult:
    def __init__(self, morris_dict):
        self.morris_dict = morris_dict

    def to_df(self):
        df = pd.DataFrame(
            self.morris_dict,
            index=[f"param{i + 1}" for i in range(len(self.morris_dict["mu"]))],
        )
        return df


def morris_res_mock():
    morris_dict = {
        "mu": [0.2, 0.5, 0.1],
        "mu_star": [0.25, 0.55, 0.15],
        "mu_star_conf": [0.025, 0.055, 0.015],
        "sigma": [0.05, 0.1, 0.02],
        "euclidian distance": [0.5, 0.7, 0.15],
        "normalized euclidian distance": [0.1, 0.2, 0.05],
    }
    return MorrisResult(morris_dict)


class TestSensitivity:
    def test_sanalysis(self):
        model = Ishigami()

        sa_analysis = SAnalysis(parameters_list=PARAMETER_LIST, method=Method.SOBOL)

        sa_analysis.draw_sample(
            1, sampling_kwargs={"calc_second_order": True, "seed": 42}
        )

        sa_analysis.evaluate(model, SIMULATION_OPTIONS, n_cpu=4)

        sa_analysis.analyze(
            "res", sensitivity_method_kwargs={"calc_second_order": True}
        )

        np.testing.assert_almost_equal(
            sa_analysis.sensitivity_results["S1"],
            np.array([1.022, 2.229, -1.412]),
            decimal=3,
        )

        sa_analysis = SAnalysis(parameters_list=PARAMETER_LIST, method=Method.MORRIS)

        sa_analysis.draw_sample(15)

        sa_analysis.evaluate(model, SIMULATION_OPTIONS, n_cpu=4)

        sa_analysis.analyze(indicator="res")

        # Problem is highly non linear and result do not converge.
        # We are just checking the execution

    # @patch("plotly.graph_objects.Figure.show")
    def test_plot_sobol_st_bar(self):
        res = sobol_res_mock()
        fig = plot_sobol_st_bar(res)
        assert fig["layout"]["title"]["text"] == "Sobol Total indices"

    def test_morris_plots(self):
        res = morris_res_mock()
        fig = plot_morris_st_bar(res, distance_metric="absolute")
        assert (
            fig["layout"]["title"]["text"]
            == "Morris Sensitivity Analysis - euclidian Distance"
        )
        fig = plot_morris_st_bar(res)
        assert (
            fig["layout"]["title"]["text"]
            == "Morris Sensitivity Analysis - Normalized euclidian Distance"
        )
        with pytest.raises(
            ValueError,
            match="Distance metric must be either 'absolute' or 'normalized'",
        ):
            plot_morris_st_bar(res, distance_metric="invalid_metric")
        fig = plot_morris_scatter(res)
        assert fig["layout"]["title"]["text"] == "Morris Sensitivity Analysis"

    def test_plot_sample(self):
        results = pd.DataFrame(
            {"variable1": [0.5, 0.3], "variable2": [0.1, 0.05]},
            index=pd.date_range("2009-07-13 00:00:00", periods=2, freq="h"),
        )

        sa_res1 = ({"param1": 1, "param2": 2}, {"simulation_options": 0}, results)
        sa_res2 = ({"param1": 12, "param2": 22}, {"simulation_options": 0}, results)
        sa_res = [sa_res1, sa_res2]
        fig = plot_sample(sa_res, indicator="variable1")
        assert len(fig.data) == 2

        fig_with_options = plot_sample(
            sa_res,
            indicator="variable1",
            title="Test Title",
            y_label="Y Axis",
            x_label="X Axis",
            show_legends=True,
        )
        assert fig_with_options.layout.title.text == "Test Title"
        assert fig_with_options.layout.xaxis.title.text == "X Axis"
        assert fig_with_options.layout.yaxis.title.text == "Y Axis"


class IshigamiTwoOutputs(Model):
    def simulate(self, parameter_dict, simulation_options):
        A1 = 7.0
        B1 = 0.1
        A2 = 5.0
        B2 = 0.5

        default_parameters = {
            "x": 1,
            "y": 1,
            "z": 1,
        }

        parameters = {**default_parameters, **parameter_dict}

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
    def simulate(self, parameter_dict, simulation_options):
        default_parameters = {
            "x": 1,
            "y": 1,
        }

        parameters = {**default_parameters, **parameter_dict}

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
