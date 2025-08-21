import numpy as np
import pandas as pd

from corrai.base.parameter import Parameter
from corrai.sensitivity import (
    SobolSanalysis,
    MorrisSanalysis,
    FASTSanalysis,
    RBDFASTSanalysis,
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


class TestSensitivity:
    def test_sanalysis_sobol(self):
        sobol_analysis = SobolSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )

        sobol_analysis.add_sample(N=1000, n_cpu=1, calc_second_order=True)
        res = sobol_analysis.analyze("res", calc_second_order=True, seed=42)

        np.testing.assert_almost_equal(
            res["mean_res"]["S1"],
            np.array([0.31234581102948833, 0.4429936089111491, 0.009030856634835067]),
        )

        res = sobol_analysis.analyze("res", freq="h", calc_second_order=True, seed=42)
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
            np.array([0.31234581102948833, 0.4429936089111491, 0.009030856634835067]),
            decimal=3,
        )

    def test_sanalysis_morris(self):
        morris_analysis = MorrisSanalysis(
            parameters=PARAMETER_LIST,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )
        morris_analysis.add_sample(N=1000, n_cpu=1, seed=42)
        agg_res = morris_analysis.sampler.sample.get_aggregate_time_series("res")

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
            model=Ishigami(),
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
            model=Ishigami(),
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
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sobol_analysis.add_sample(N=2**2, n_cpu=1, calc_second_order=True)
        fig_matrix = sobol_analysis.plot_s2_matrix()
        assert fig_matrix["layout"]["title"]["text"] == (
            "Sobol mean res " "- 2nd order interactions"
        )

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

        res_euclidian = morris_analysis.analyze("res")[f"mean_res"]
        fig_data = fig_bar.data[0]
        expected_x = [p.name for p in PARAMETER_LIST]
        expected_y = res_euclidian["euclidian_distance"].tolist()
        assert list(fig_data.x) == expected_x
        np.testing.assert_allclose(fig_data.y, expected_y)

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

        # assert isinstance(fig_dyn, go.Figure)

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
