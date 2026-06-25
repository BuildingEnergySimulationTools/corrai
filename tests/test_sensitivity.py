import warnings

import numpy as np
import pandas as pd
import pytest

from corrai.base.parameter import Parameter
from corrai.base.model import IshigamiDynamic, Ishigami
from corrai.sensitivity import (
    SobolSanalysis,
    MorrisSanalysis,
    FASTSanalysis,
    RBDFASTSanalysis,
    plot_bars,
    plot_dynamic_metric,
    plot_morris_scatter,
    plot_s2_matrix,
)


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


class TestSensitivity:
    def test_sanalysis_sobol_static(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            calc_second_order=True,
        )

        sobol_analysis.add_sample(N=1000, n_cpu=1, seed=42)
        res = sobol_analysis.analyze("res", seed=42)

        np.testing.assert_almost_equal(
            res["mean_res"]["S1"],
            np.array([0.33080399, 0.44206835, 0.00946747]),
        )

    def test_sanalysis_sobol_dynamic(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
            calc_second_order=True,
        )

        sobol_analysis.add_sample(N=1000, n_cpu=1, seed=42)
        res = sobol_analysis.analyze("res", seed=42)

        np.testing.assert_almost_equal(
            res["mean_res"]["S1"],
            np.array([0.33080399, 0.44206835, 0.00946747]),
        )

        res_freq = sobol_analysis.analyze("res", freq="h", seed=42)
        assert res_freq.index.tolist() == [
            pd.Timestamp("2009-01-01 00:00:00"),
            pd.Timestamp("2009-01-01 01:00:00"),
            pd.Timestamp("2009-01-01 02:00:00"),
            pd.Timestamp("2009-01-01 03:00:00"),
            pd.Timestamp("2009-01-01 04:00:00"),
            pd.Timestamp("2009-01-01 05:00:00"),
        ]
        np.testing.assert_almost_equal(
            res_freq["2009-01-01 00:00:00"]["S1"],
            np.array([0.33080399, 0.44206835, 0.00946747]),
            decimal=3,
        )

    def test_sobol_incremental_sampling_raises(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
        )
        sobol_analysis.add_sample(N=64, n_cpu=1, seed=42)
        with pytest.raises(ValueError, match="does not support incremental sampling"):
            sobol_analysis.add_sample(N=64, n_cpu=1, seed=0)

    def test_sobol_plot_s2_matrix_raises_when_no_second_order(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            calc_second_order=False,
        )
        sobol_analysis.add_sample(N=64, n_cpu=1, seed=42)
        with pytest.raises(ValueError, match="requires second-order indices"):
            sobol_analysis.plot_s2_matrix()

    def test_sobol_static_model_no_spurious_warning(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
        )
        sobol_analysis.add_sample(N=64, n_cpu=1, seed=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sobol_analysis.analyze("res", seed=42)
        our_warnings = [w for w in caught if "sensitivity" in str(w.filename)]
        assert (
            len(our_warnings) == 0
        ), f"Unexpected warnings from sensitivity.py: {our_warnings}"

    def test_sanalysis_morris(self):
        morris_analysis = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        morris_analysis.add_sample(N=1000, n_cpu=1, seed=42)
        agg_res = morris_analysis.sampler.sample.get_aggregated_time_series("res")

        pd.testing.assert_frame_equal(
            agg_res.loc[0:7],
            pd.DataFrame(
                {
                    "aggregated_res": {
                        0: 5.250000000000186,
                        1: 4.2798279944936946,
                        2: -0.970172005506723,
                        3: -9.301900143286739,
                        4: 0.9701720055067234,
                        5: 2.316952686764885e-13,
                        6: 5.250000000000647,
                        7: 5.250000000002636,
                    }
                }
            ),
        )

        res = morris_analysis.analyze("res")
        np.testing.assert_almost_equal(
            res["mean_res"]["mu"],
            np.array([7.654063742766665, -0.3150000000000245, 0.37492776620012525]),
        )
        np.testing.assert_almost_equal(
            res["mean_res"]["euclidian_distance"],
            np.array([9.882749087179025, 11.135259466146298, 10.443154164442882]),
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
            np.array([7.654063742766665, -0.3150000000000245, 0.37492776620012525]),
            decimal=3,
        )

    def test_sanalysis_fast(self):
        fast_analysis = FASTSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        # N = max(N, 4 * M**2 + 1)
        fast_analysis.add_sample(N=65, n_cpu=1, seed=42)
        res_array = np.array(
            [0.36517142701698435, 0.6669670030829337, 0.022211236308415948]
        )
        res = fast_analysis.analyze("res")
        np.testing.assert_almost_equal(res["mean_res"]["S1"], res_array)
        assert len(res["mean_res"]["S1_conf"]) == len(PARAMETER_LIST)

        res = fast_analysis.analyze("res", freq="h")
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
            res_array,
            decimal=3,
        )

    def test_sanalysis_rbdfast(self):
        rbdfast_analysis = RBDFASTSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )

        # N = max(N, 2 * M + 1)
        rbdfast_analysis.add_sample(N=100, n_cpu=1, seed=42)
        res_array = np.array(
            [0.2497000361318199, 0.5276937277925962, 0.12398148477945364]
        )
        res = rbdfast_analysis.analyze("res")
        np.testing.assert_almost_equal(res["mean_res"]["S1"], res_array)

        assert len(res["mean_res"]["S1_conf"]) == len(PARAMETER_LIST)

        res = rbdfast_analysis.analyze("res", freq="h")
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
            res_array,
            decimal=3,
        )


class TestPlots:
    def test_sobol_s2_matrix(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
            calc_second_order=True,
        )
        sobol_analysis.add_sample(N=2**2, n_cpu=1)
        fig_matrix = sobol_analysis.plot_s2_matrix()
        assert fig_matrix["layout"]["title"]["text"] == (
            "Sobol mean res " "- 2nd order interactions"
        )

    def test_morris_plots(self):
        morris_analysis = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        morris_analysis_2 = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )

        morris_analysis.add_sample(N=2, n_cpu=1, seed=42)
        fig_scatter = morris_analysis.plot_scatter()
        assert fig_scatter["layout"]["title"]["text"] == "Morris Sensitivity Analysis"

        morris_analysis_2.add_sample(N=2, n_cpu=1, seed=42)
        fig_scatter2 = morris_analysis_2.plot_scatter()

        x1 = fig_scatter.data[0].x
        y1 = fig_scatter.data[0].y
        x2 = fig_scatter2.data[0].x
        y2 = fig_scatter2.data[0].y

        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(
            y1,
            y2,
        )
        assert list(fig_scatter.data[0].text) == list(fig_scatter2.data[0].text)

        fig_bar = morris_analysis_2.plot_bar(sensitivity_metric="euclidian_distance")
        assert (
            fig_bar["layout"]["title"]["text"] == "Morris euclidian_distance mean res"
        )

        fig_data = fig_bar.data[0]
        assert list(fig_data.x) == ["par_x1", "par_x3", "par_x2"]
        np.testing.assert_allclose(
            fig_data.y, [1.455258008259737, 10.823232337117245, 13.63990010960599]
        )

        fig_dyn = morris_analysis.plot_dynamic_metric(
            indicator="res",
            sensitivity_metric="euclidian_distance",
            freq="h",
            method="mean",
        )
        assert (
            fig_dyn["layout"]["title"]["text"]
            == "Morris dynamic euclidian_distance mean res"
        )

    def test_plot_bars_plot_kwargs(self):
        s = pd.Series([0.3, 0.5, 0.2], index=["p1", "p2", "p3"], name="ST")
        fig = plot_bars(s, title="My Title", paper_bgcolor="red")
        assert fig["layout"]["title"]["text"] == "My Title"
        assert fig["layout"]["paper_bgcolor"] == "red"

    def test_plot_dynamic_metric_plot_kwargs(self):
        metrics = pd.DataFrame(
            {"p1": [0.1, 0.2], "p2": [0.3, 0.4]},
            index=pd.date_range("2009-01-01", periods=2, freq="h"),
        )
        fig = plot_dynamic_metric(metrics, font={"size": 20}, paper_bgcolor="blue")
        assert fig["layout"]["font"]["size"] == 20
        assert fig["layout"]["paper_bgcolor"] == "blue"

    def test_plot_s2_matrix_plot_kwargs(self):
        result = {"S2": np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.2], [0.0, 0.2, 0.0]])}
        fig = plot_s2_matrix(
            result, ["p1", "p2", "p3"], title="Custom S2", paper_bgcolor="green"
        )
        assert fig["layout"]["paper_bgcolor"] == "green"
        assert fig["layout"]["title"]["text"] == "Custom S2"

    def test_plot_s2_matrix_fixed_range(self):
        result = {"S2": np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.2], [0.0, 0.2, 0.0]])}
        param_names = ["p1", "p2", "p3"]

        fig_fixed = plot_s2_matrix(
            result, param_names, fixed_range=True, colorscale="Blues"
        )
        assert fig_fixed.data[0].zmin == -1
        assert fig_fixed.data[0].zmax == 1

        fig_auto = plot_s2_matrix(result, param_names, fixed_range=False)
        assert fig_auto.data[0].zmin == 0.0
        assert fig_auto.data[0].zmax == 0.2

    def test_plot_s2_matrix_trace_kwargs(self):
        result = {"S2": np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.2], [0.0, 0.2, 0.0]])}
        fig = plot_s2_matrix(result, ["p1", "p2", "p3"], zmin=-0.5, zmax=0.5)
        assert fig.data[0].zmin == -0.5
        assert fig.data[0].zmax == 0.5

    def test_plot_morris_scatter_plot_kwargs(self):
        morris_df = pd.DataFrame(
            {"mu_star": [1.0, 2.0], "sigma": [0.5, 0.8], "mu_star_conf": [0.1, 0.2]},
            index=["p1", "p2"],
        )
        fig = plot_morris_scatter(
            morris_df, title="Custom Morris", paper_bgcolor="yellow"
        )
        assert fig["layout"]["title"]["text"] == "Custom Morris"
        assert fig["layout"]["paper_bgcolor"] == "yellow"

    def test_sobol_plot_bar_plot_kwargs(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
        )
        sobol_analysis.add_sample(N=2**4, n_cpu=1, seed=42)
        fig = sobol_analysis.plot_bar(
            plot_kwargs={
                "title": "My Custom Title",
                "font": {"family": "Arial", "size": 14},
            },
        )
        assert fig["layout"]["title"]["text"] == "My Custom Title"
        assert fig["layout"]["font"]["family"] == "Arial"
        assert fig["layout"]["font"]["size"] == 14
