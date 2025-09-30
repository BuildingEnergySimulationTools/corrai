from pathlib import Path
import numpy as np
import pandas as pd

from corrai.base.parameter import Parameter
from corrai.sampling import (
    LHSSampler,
    MorrisSampler,
    SobolSampler,
    Sample,
)

from corrai.base.model import (
    IshigamiDynamic,
    Ishigami,
    PymodelDynamic,
    PymodelStatic,
    Sine,
)

import pytest

import plotly.graph_objects as go


FILES_PATH = Path(__file__).parent / "resources"

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 00:00:00",
    "timestep": "h",
}

REAL_PARAM = [
    Parameter("param_1", (0, 10), relabs="Absolute", model_property="prop_1"),
    Parameter("param_2", (0.8, 1.2), relabs="Relative", model_property="prop_2"),
    Parameter("param_3", (0, 100), relabs="Absolute", model_property="prop_3"),
]

ISHIGAMI_PARAMETERS = [
    Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
    Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
    Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
]


class TestSample:
    def test_sample_methods(self):
        # Dynamic sample
        sample = Sample(REAL_PARAM)
        assert sample.values.shape == (0, 3)
        pd.testing.assert_series_equal(sample.results, pd.Series())

        sample.add_samples(
            np.array([[1, 0.9, 10], [3, 0.85, 20]]),
            [
                pd.DataFrame(),
                pd.DataFrame(
                    {"res": [1, 2]}, index=pd.date_range("2009", freq="h", periods=2)
                ),
            ],
        )

        assert sample.get_pending_index().tolist() == [True, False]
        pd.testing.assert_frame_equal(
            sample.values,
            pd.DataFrame(
                {
                    "param_1": {0: 1.0, 1: 3.0},
                    "param_2": {0: 0.9, 1: 0.85},
                    "param_3": {0: 10.0, 1: 20.0},
                }
            ),
        )

        assert sample.get_parameters_intervals().tolist() == [
            [0.0, 10.0],
            [0.8, 1.2],
            [0.0, 100.0],
        ]
        assert sample.get_list_parameter_value_pairs(sample.get_pending_index()) == [
            [(REAL_PARAM[0], 1.0), (REAL_PARAM[1], 0.9), (REAL_PARAM[2], 10.0)],
        ]

        assert len(sample) == 2

        item = sample[1]
        assert isinstance(item, dict)
        assert np.allclose(item["values"], [3.0, 0.85, 20.0])
        pd.testing.assert_frame_equal(item["results"], sample.results.iloc[1])

        new_result = pd.DataFrame({"res": [42]}, index=pd.date_range("2009", periods=1))
        sample[1] = {"results": new_result}
        pd.testing.assert_frame_equal(sample.results.iloc[1], new_result)

        sample[0] = {
            "values": np.array([9.9, 1.1, 88]),
            "results": pd.DataFrame({"res": [123]}, index=[pd.Timestamp("2009-01-01")]),
        }
        pd.testing.assert_series_equal(
            sample.values.loc[0],
            pd.Series({"param_1": 9.9, "param_2": 1.1, "param_3": 88.0}, name=0),
        )
        assert not sample.results.iloc[0].empty

        dimless_val = sample.get_dimension_less_values()
        pd.testing.assert_frame_equal(
            dimless_val,
            pd.DataFrame(
                {
                    "param_1": {0: 0.99, 1: 0.3},
                    "param_2": {0: 0.7500000000000003, 1: 0.12499999999999986},
                    "param_3": {0: 0.88, 1: 0.2},
                }
            ),
        )

        pd.testing.assert_frame_equal(
            sample.get_aggregated_time_series("res"),
            pd.DataFrame([123.0, 42.0], [0, 1], columns=["aggregated_res"]),
        )

        fig = sample.plot_hist("res")
        assert fig.layout.title["text"] == "Sample distribution of mean res"
        assert fig.layout.xaxis.title["text"] == "mean res "

        fig = sample.plot_sample("res")
        assert fig

        ref = pd.Series(
            [123, 456],
            index=pd.date_range("2009-01-02", periods=2, freq="d"),
            name="ref",
        )

        sample.results[0].loc[pd.Timestamp("2009-01-02"), "res"] = 456
        sample.results[1].loc[pd.Timestamp("2009-01-02"), "res"] = 43

        score_res = sample.get_score_df("res", ref)

        pd.testing.assert_frame_equal(
            score_res,
            pd.DataFrame(
                {
                    "r2_score": {0: 1.0, 1: -2.19},
                    "nmbe": {0: -57.51, 1: 94.11},
                    "cv_rmse": {0: 115.02, 1: 188.23},
                    "mean_absolute_error": {0: 0.0, 1: 247.0},
                    "root_mean_squared_error": {0: 0.0, 1: 297.59},
                    "max_error": {0: 0.0, 1: 413.0},
                }
            ).rename_axis(ref.index.freq),
            atol=0.1,
        )

        sample._validate()

        # Static Sample
        sample_static = Sample(REAL_PARAM, is_dynamic=False)

        sample_static.add_samples(
            np.array([[1, 0.9, 10], [3, 0.85, 20]]),
            [pd.Series(), pd.Series({"res": 2})],
        )

        pd.testing.assert_frame_equal(
            sample_static.get_static_results_as_df(),
            pd.DataFrame({"res": {0: np.nan, 1: 2.0}}),
        )

    def test_plot_sample(self):
        t = pd.date_range("2025-01-01 00:00:00", periods=2, freq="h")
        df1 = pd.DataFrame({"res": [1.0, 2.0]}, index=t)
        df2 = pd.DataFrame({"res": [3.0, 4.0]}, index=t)
        df3 = pd.DataFrame({"res": [5.0, 6.0]}, index=t)

        ref = pd.Series([2.0, 2.0], index=t)

        sample = Sample(
            parameters=[
                Parameter("p1", interval=(0, 10)),
                Parameter("p2", interval=(0, 10)),
            ]
        )

        sample.add_samples(
            np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]), [df1, df2, df3]
        )

        fig = sample.plot_sample(
            indicator="res",
            reference_timeseries=ref,
            title="test",
            x_label="time",
            y_label="value",
            alpha=0.3,
            show_legends=True,
            type_graph="scatter",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

        np.testing.assert_allclose(fig.data[0]["y"], df1["res"].to_numpy())
        np.testing.assert_allclose(fig.data[-1]["y"], ref.to_numpy())

        assert fig.data[0].name == "p1: 1.1, p2: 2.2"
        assert fig.data[1].name == "p1: 3.3, p2: 4.4"
        assert fig.data[2].name == "p1: 5.5, p2: 6.6"

        # Partial simulation
        empty_df = pd.DataFrame({"res": []})
        sample[:] = {"results": [empty_df, df1, empty_df]}

        fig_partial = sample.plot_sample(indicator="res", type_graph="scatter")

        # Only 1 non-empty sample
        assert len(fig_partial.data) == 1
        np.testing.assert_allclose(fig_partial.data[0]["y"], df1["res"].to_numpy())

        # All results empty and no reference
        sample[:] = {"results": [empty_df] * 3}

        with pytest.raises(ValueError, match="No simulated data available to plot."):
            sample.plot_sample(indicator="res", type_graph="scatter")

        # All results empty but with reference
        fig_ref_only = sample.plot_sample(
            indicator="res",
            reference_timeseries=ref,
            type_graph="scatter",
        )
        assert len(fig_ref_only.data) == 1
        np.testing.assert_allclose(fig_ref_only.data[0]["y"], ref.to_numpy())

        t = pd.date_range("2025-01-01 00:00:00", periods=3, freq="h")
        df1 = pd.DataFrame({"res": [1.0, 2.0, 3.0]}, index=t)
        df2 = pd.DataFrame({"res": [2.0, 3.0, 4.0]}, index=t)
        df3 = pd.DataFrame({"res": [3.0, 4.0, 5.0]}, index=t)
        ref = pd.Series([2.0, 2.5, 3.0], index=t)

        sample[:] = {"results": [df1, df2, df3]}

        fig = sample.plot_sample(
            indicator="res",
            reference_timeseries=ref,
            type_graph="area",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 6
        np.testing.assert_allclose(fig.data[-1]["y"], ref.to_numpy())
        assert fig.data[-1].name == "Reference"

        names = [tr.name for tr in fig.data]
        assert "Median" in names
        assert "Quantiles" in names

        fig = sample.plot_sample(
            indicator="res",
            reference_timeseries=ref,
            show_legends=False,
            type_graph="scatter",
        )
        assert len(fig.data) == 4
        assert fig.data[0].mode == "markers"
        np.testing.assert_allclose(np.array(fig.data[-1].y), ref.to_numpy())
        assert fig.data[-1].mode == "lines"

    def test_plot_hist(self):
        # Dynamic
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=PymodelDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42, simulate=True)

        fig = sampler.sample.plot_hist(
            indicator="res",
            method="mean",
            unit="J",
            bins=10,
            colors="orange",
            reference_value=70,
            show_rug=True,
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Sample distribution of mean res"

        hist_traces = [tr for tr in fig.data if tr.type == "histogram"]
        assert len(hist_traces) == 1
        hist = hist_traces[0]
        assert len(hist.x) == len(sampler.results)

        # Static
        sampler = LHSSampler(
            parameters=ISHIGAMI_PARAMETERS,
            model=Ishigami(),
            simulation_options=SIMULATION_OPTIONS,
        )

        sampler.add_sample(3, 42)

        fig = sampler.sample.plot_hist(
            indicator="res",
            unit="J",
            bins=10,
            colors="orange",
            reference_value=70,
            show_rug=True,
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Sample distribution of res"

        hist_traces = [tr for tr in fig.data if tr.type == "histogram"]
        assert len(hist_traces) == 1
        hist = hist_traces[0]
        assert len(hist.x) == len(sampler.results)

    def test_plot_pcp(self):
        # Dynamic
        lhs = LHSSampler(
            parameters=[
                Parameter("omega", (2, 8), model_property="omega"),
                Parameter("amplitude", (1, 3), model_property="omega"),
            ],
            model=Sine(),
            simulation_options={},
        )

        lhs.add_sample(15)

        lhs.sample.plot_pcp([("res", "mean"), ("res", "sum")])

        # Static
        lhs = LHSSampler(
            parameters=[
                Parameter("x1", (-3.14, 3.14), model_property="x"),
                Parameter("x2", (-3.14, 3.14), model_property="x2"),
                Parameter("x3", (-3.14, 3.14), model_property="x3"),
            ],
            model=Ishigami(),
            simulation_options={},
        )

        lhs.add_sample(100)

        lhs.sample.plot_pcp(["res"])

    def test_plot_pcp_in_sampler(self):
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=PymodelDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42, simulate=True)

        fig = sampler.plot_pcp(
            indicators_configs=[("res", "mean")],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_lhs_sampler(self):
        # Dynamic
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=PymodelDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42, False)
        np.testing.assert_allclose(
            sampler.sample.values,
            np.array(
                [
                    [6.9441, 1.0785, 70.7802],
                    [5.6356, 0.9393, 27.4968],
                    [0.0112, 0.8330, 61.6539],
                ]
            ),
            rtol=0.01,
        )

        sampler.simulate_pending()

        expected = {
            0: np.array([[85.75934698790918]]),
            1: np.array([[38.08478803524709]]),
            2: np.array([[61.67268698504139]]),
        }

        for k, arr in sampler.results.items():
            np.testing.assert_allclose(arr.values, expected[k], rtol=0.05)

        sampler.add_sample(3, rng=42, simulate=False)
        assert sampler.values.shape == (6, 3)
        assert all(df.empty for df in sampler.results[-3:].values)

        sampler.simulate_at(4)
        assert [df.empty for df in sampler.results[-3:].values] == [True, False, True]

        sampler.simulate_at([3, 5])
        for k, arr in sampler.results[-3:].items():
            np.testing.assert_allclose(
                arr.values,
                expected[k % 3],  # reuse expected values modulo cycle
                rtol=0.05,
            )

        sampler.add_sample(3, rng=42, simulate=False)

        sampler.simulate_at(slice(4, 7))
        assert [df.empty for df in sampler.results[-3:].values] == [False, False, True]

        sampler.add_sample(3, rng=42, simulate=False)
        sampler.simulate_at(slice(10, None))
        assert [df.empty for df in sampler.results[-3:].values] == [True, False, False]

        sampler.add_sample(3, rng=42, simulate=True)
        assert all(not df.empty for df in sampler.results[-3:])

        sampler.simulate_pending()
        to_test = sampler.get_sample_aggregated_time_series("res")
        pd.testing.assert_frame_equal(
            to_test,
            pd.DataFrame(
                {
                    "aggregated_res": {
                        0: 85.75934698790918,
                        1: 38.08478803524709,
                        2: 61.67268698504139,
                        3: 85.75934698790918,
                        4: 38.08478803524709,
                        5: 61.67268698504139,
                        6: 85.75934698790918,
                        7: 38.08478803524709,
                        8: 61.67268698504139,
                        9: 85.75934698790918,
                        10: 38.08478803524709,
                        11: 61.67268698504139,
                        12: 85.75934698790918,
                        13: 38.08478803524709,
                        14: 61.67268698504139,
                    }
                }
            ),
            check_exact=False,
        )

        # Static
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=PymodelStatic(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42)
        np.testing.assert_allclose(
            sampler.sample.values,
            np.array(
                [
                    [6.9441, 1.0785, 70.7802],
                    [5.6356, 0.9393, 27.4968],
                    [0.0112, 0.8330, 61.6539],
                ]
            ),
            rtol=0.01,
        )

        pd.testing.assert_frame_equal(
            sampler.sample.get_static_results_as_df(),
            pd.DataFrame({"res": {0: 85.759, 1: 38.084, 2: 61.672}}),
            atol=0.01,
        )

        assert True

    def test_morris_sampler(self):
        sampler = MorrisSampler(
            parameters=ISHIGAMI_PARAMETERS,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )

        sampler.add_sample(N=1, **{"seed": 42})
        np.testing.assert_allclose(
            sampler.values,
            np.array(
                [
                    [3.14159265, 1.04719755, 1.04719755],
                    [-1.04719755, 1.04719755, 1.04719755],
                    [-1.04719755, -3.14159265, 1.04719755],
                    [-1.04719755, -3.14159265, -3.14159265],
                ]
            ),
        )

    def test_sobol_sampler(self):
        sampler = SobolSampler(
            parameters=ISHIGAMI_PARAMETERS,
            model=IshigamiDynamic(),
            simulation_options=SIMULATION_OPTIONS,
        )

        sampler.add_sample(N=1, **{"seed": 42})
        np.testing.assert_allclose(
            sampler.values,
            np.array(
                [
                    [-0.43335459, 1.97523222, 1.92524819],
                    [-2.8057144, 1.97523222, 1.92524819],
                    [-0.43335459, -1.26958525, 1.92524819],
                    [-0.43335459, 1.97523222, 2.13551644],
                    [-0.43335459, -1.26958525, 2.13551644],
                    [-2.8057144, 1.97523222, 2.13551644],
                    [-2.8057144, -1.26958525, 1.92524819],
                    [-2.8057144, -1.26958525, 2.13551644],
                ]
            ),
        )
