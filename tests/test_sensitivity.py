import numpy as np
import pytest
import pandas as pd

from corrai.base.parameter import Parameter
from corrai.sensitivity import Method
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
