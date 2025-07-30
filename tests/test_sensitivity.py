import numpy as np
import pytest
import pandas as pd
import copy
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error, mean_absolute_error

from corrai.base.model import Model
from corrai.base.parameter import Parameter
from corrai.sensitivity import Method, SobolSanalysis, MorrisSanalysis
from corrai.sensitivity import ObjectiveFunction
from corrai.sensitivity import SAnalysisLegacy
from corrai.sensitivity import (
    # plot_sobol_st_bar,
    # plot_morris_st_bar,
    # plot_morris_scatter,
    plot_sample,
)

from tests.resources.pymodels import Ishigami

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 05:00:00",
    "timestep": "h",
}

PARAMETER_LIST = [
    Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
    Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
    Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
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
    def test_sanalysis_sobol(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )

        sobol_analysis.add_sample(N=1, n_cpu=1, calc_second_order=True, seed=42)

        res = sobol_analysis.analyze("res", calc_second_order=True)
        np.testing.assert_almost_equal(
            res["mean_res"]["S1"], np.array([1.02156668, 2.22878092, -1.41228784])
        )

        res = sobol_analysis.analyze("res", freq="h", calc_second_order=True)
        assert res.index.tolist() == [
            pd.Timestamp("2009-01-01 00:00:00"),
            pd.Timestamp("2009-01-01 01:00:00"),
            pd.Timestamp("2009-01-01 02:00:00"),
            pd.Timestamp("2009-01-01 03:00:00"),
            pd.Timestamp("2009-01-01 04:00:00"),
            pd.Timestamp("2009-01-01 05:00:00"),
        ]

        np.testing.assert_almost_equal(
            res["2009-01-01 00:00:00"]["S1"],
            np.array([1.02156668, 2.22878092, -1.41228784]),
            decimal=3,
        )

    def test_sanalysis_morris(self):
        morris_analysis = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )
        morris_analysis.add_sample(N=2, n_cpu=1, seed=42)
        res = morris_analysis.analyze("res")
        np.testing.assert_almost_equal(
            res["mean_res"]["mu"], np.array([1.45525800, 1.77635683e-15, 6.24879610])
        )
        assert len(res["mean_res"]["mu_star"]) == len(PARAMETER_LIST)

        res = morris_analysis.analyze("res", freq="h")
        assert res.index.tolist() == [
            pd.Timestamp("2009-01-01 00:00:00"),
            pd.Timestamp("2009-01-01 01:00:00"),
            pd.Timestamp("2009-01-01 02:00:00"),
            pd.Timestamp("2009-01-01 03:00:00"),
            pd.Timestamp("2009-01-01 04:00:00"),
            pd.Timestamp("2009-01-01 05:00:00"),
        ]
        np.testing.assert_almost_equal(
            res["2009-01-01 00:00:00"]["mu"],
            np.array([1.45525800, 1.77635683e-15, 6.24879610]),
            decimal=3,
        )


class TestPlots:
    def test_morris_plots(self):
        morris_analysis = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )
        morris_analysis_2 = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )

        morris_analysis.add_sample(N=2, n_cpu=1, seed=42)
        fig1 = morris_analysis.plot_scatter()
        assert fig1["layout"]["title"]["text"] == "Morris Sensitivity Analysis"

        morris_analysis_2.add_sample(N=2, n_cpu=1, seed=42)
        res = morris_analysis_2.analyze("res", freq="h")
        fig2 = morris_analysis_2.plot_scatter()

        x1 = fig1.data[0].x
        y1 = fig1.data[0].y
        x2 = fig2.data[0].x
        y2 = fig2.data[0].y

        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(
            y1,
            y2,
        )
        assert list(fig1.data[0].text) == list(fig2.data[0].text)

        fig3 = morris_analysis_2.plot_bar(distance_metric="absolute")
        assert (
            fig3["layout"]["title"]["text"]
            == "Morris Sensitivity Analysis – Absolute Euclidian distance"
        )

        fig4 = morris_analysis_2.plot_bar(distance_metric="normalized")
        normalized_values = fig4["data"][0]["y"]
        assert (
            fig4["layout"]["title"]["text"]
            == "Morris Sensitivity Analysis – Normalized Euclidian distance"
        )
        assert all(val <= 1 for val in normalized_values)

    # def test_dynamic_analysis_and_absolute(self):
    #     model = Ishigami()
    #
    #     sa_analysis = SAnalysisLegacy(
    #         parameters_list=PARAMETER_LIST, method=Method.SOBOL
    #     )
    #
    #     sa_analysis.draw_sample(
    #         100, sampling_kwargs={"calc_second_order": False, "seed": 42}
    #     )
    #
    #     sa_analysis.evaluate(
    #         model=model, simulation_options=SIMULATION_OPTIONS, n_cpu=1
    #     )
    #
    #     sa_analysis.analyze(
    #         indicator="res",
    #         freq="3h",
    #         absolute=True,
    #         sensitivity_method_kwargs={"calc_second_order": False},
    #     )
    #
    #     assert isinstance(sa_analysis.sensitivity_dynamic_results, dict)
    #     assert len(sa_analysis.sensitivity_dynamic_results) > 0
    #
    #     for key, result in sa_analysis.sensitivity_dynamic_results.items():
    #         assert "ST" in result
    #         assert "names" in result
    #         assert "_absolute" in result
    #
    #     indicators = sa_analysis.calculate_sensitivity_indicators()
    #     assert sa_analysis.static is False
    #
    #     assert "ST" in indicators
    #     assert isinstance(indicators["ST"], pd.Series)
    #
    #     expected_st = {
    #         pd.Timestamp("2009-01-01 00:00:00"): np.array(
    #             [8.5377964, 6.88499214, 3.90531498]
    #         ),
    #         pd.Timestamp("2009-01-01 03:00:00"): np.array(
    #             [8.5377964, 6.88499214, 3.90531498]
    #         ),
    #     }
    #
    #     for timestamp, expected in expected_st.items():
    #         assert timestamp in sa_analysis.sensitivity_dynamic_results
    #         result = sa_analysis.sensitivity_dynamic_results[timestamp]["ST"]
    #         np.testing.assert_allclose(result, expected, rtol=0.05)
    #
    #     fig = plot_sobol_st_bar(sa_analysis.sensitivity_dynamic_results)
    #     assert fig.layout.title.text == "Sobol ST indices (dynamic)"

    # def test_sobol_st_bar_normalize(self):
    #     sobol_dict_dynamic = {
    #         "time1": {"ST": [0.5, 0.7, 0.15], "names": ["param1", "param2", "param3"]},
    #         "time2": {"ST": [0.6, 0.8, 0.2], "names": ["param1", "param2", "param3"]},
    #     }
    #     fig = plot_sobol_st_bar(sobol_dict_dynamic, normalize_dynamic=True)
    #     # fig.show()
    #     assert fig.layout.yaxis.title.text == "Cumulative percentage [0-1]"
    #     assert fig.layout.title.text == "Sobol ST indices (dynamic)"
    #
    #     df_to_plot = pd.DataFrame(
    #         {
    #             t: pd.Series(res["ST"], index=res["names"])
    #             for t, res in sobol_dict_dynamic.items()
    #         }
    #     ).T
    #     normalized_values = df_to_plot.div(df_to_plot.sum(axis=1), axis=0)
    #
    #     for bar in fig.data:
    #         param_name = bar["name"]  # Récupère le nom du paramètre (ex: 'param1')
    #         expected_values = normalized_values[
    #             param_name
    #         ]  # Récupère les valeurs normalisées pour ce paramètre
    #         np.testing.assert_allclose(bar["y"], expected_values.values, rtol=0.05)

    # @patch("plotly.graph_objects.Figure.show")
    # def test_plot_sobol_st_bar(self):
    #     res = sobol_res_mock()
    #     fig = plot_sobol_st_bar(res)
    #     assert fig["layout"]["title"]["text"] == "Sobol Total indices"

    # def test_plot_sample(self):
    #     results = pd.DataFrame(
    #         {"variable1": [0.5, 0.3], "variable2": [0.1, 0.05]},
    #         index=pd.date_range("2009-07-13 00:00:00", periods=2, freq="h"),
    #     )
    #
    #     sa_res1 = ({"param1": 1, "param2": 2}, {"simulation_options": 0}, results)
    #     sa_res2 = ({"param1": 12, "param2": 22}, {"simulation_options": 0}, results)
    #     sa_res = [sa_res1, sa_res2]
    #     fig = plot_sample(sa_res, indicator="variable1")
    #     assert len(fig.data) == 2
    #
    #     fig_with_options = plot_sample(
    #         sa_res,
    #         indicator="variable1",
    #         title="Test Title",
    #         y_label="Y Axis",
    #         x_label="X Axis",
    #         show_legends=True,
    #     )
    #     assert fig_with_options.layout.title.text == "Test Title"
    #     assert fig_with_options.layout.xaxis.title.text == "X Axis"
    #     assert fig_with_options.layout.yaxis.title.text == "Y Axis"


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


# PARAMETERS = [
#     {Parameter.NAME: "x", Parameter.INTERVAL: (0.0, 3.0), Parameter.INIT_VALUE: 2.0},
#     {Parameter.NAME: "y", Parameter.INTERVAL: (0.0, 3.0), Parameter.INIT_VALUE: 2.0},
# ]


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
