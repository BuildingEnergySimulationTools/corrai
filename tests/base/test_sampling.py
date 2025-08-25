import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from corrai.base.model import Model
from corrai.base.parameter import Parameter
from corrai.base.sampling import (
    # VariantSubSampler,
    Sampler,
    plot_sample,
    plot_pcp,
    # get_mapped_bounds,
    LHSSampler,
    MorrisSampler,
    SobolSampler,
    Sample,
)
from corrai.variant import VariantKeys, get_combined_variants, get_modifier_dict
from corrai.base.math import aggregate_time_series

from tests.resources.pymodels import VariantModel, Pymodel, Ishigami
import pytest

import plotly.graph_objects as go


FILES_PATH = Path(__file__).parent / "resources"

# parameters = [
#     {Parameter.NAME: "Parameter1", Parameter.INTERVAL: (0, 6), Parameter.TYPE: "Real"},
#     {
#         Parameter.NAME: "Parameter2",
#         Parameter.INTERVAL: (1, 4),
#         Parameter.TYPE: "Integer",
#     },
#     {
#         Parameter.NAME: "Parameter3",
#         Parameter.INTERVAL: (0.02, 0.04, 0.2),
#         Parameter.TYPE: "Choice",
#     },
#     {
#         Parameter.NAME: "Parameter4",
#         Parameter.INTERVAL: (0, 1),
#         Parameter.TYPE: "Binary",
#     },
# ]
#
# parameters_to_expand = [
#     {
#         Parameter.NAME: "options_expanded",
#         Parameter.INTERVAL: [2, 6],
#         Parameter.TYPE: "Choice",
#     },
#     {
#         Parameter.NAME: "Parameter3",
#         Parameter.INTERVAL: [0.5, 0.7],
#         Parameter.TYPE: "Choice",
#     },
#     {
#         Parameter.NAME: "options_expanded_bis",
#         Parameter.INTERVAL: ["bad", "good"],
#         Parameter.TYPE: "Choice",
#     },
# ]
#
# discrete_parameters = [
#     {
#         Parameter.NAME: "discrete1",
#         Parameter.INTERVAL: (0, 2),
#         Parameter.TYPE: "Choice",
#     },
#     {
#         Parameter.NAME: "discrete2",
#         Parameter.INTERVAL: (1, 2),
#         Parameter.TYPE: "Choice",
#     },
#     {
#         Parameter.NAME: "discrete3",
#         Parameter.INTERVAL: (4, 6),
#         Parameter.TYPE: "Choice",
#     },
#     {
#         Parameter.NAME: "discrete4",
#         Parameter.INTERVAL: (0, 1),
#         Parameter.TYPE: "Choice",
#     },
# ]
#
# params_mappings = {
#     "options_expanded": [
#         "Parameter1",
#         "Parameter2",
#     ],
#     "options_expanded_bis": {
#         "bad": {
#             "Parameter4": 25,
#         },
#         "good": {
#             "Parameter4": 3,
#         },
#     },
# }
#
SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 00:00:00",
    "timestep": "h",
}
#
# simulation_options_for_param = {
#     "start": "2025-01-01 00:00:00",
#     "end": "2025-01-01 23:00:00",
#     "timestep": "h",
# }

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
    def test_plot_hist(self):
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=Pymodel(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42, simulate=True)

        fig = sampler.sample.plot_hist(
            indicator="res",
            method="mean",
            unit="J",
            bin_size=0.5,
            colors="orange",
            show_rug=True,
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Sample distribution of mean res"

        hist_traces = [tr for tr in fig.data if tr.type == "histogram"]
        assert len(hist_traces) == 1
        hist = hist_traces[0]
        assert len(hist.x) == len(sampler.results)

    def test_plot_pcp(self):
        t = pd.date_range("2025-01-01 00:00:00", periods=2, freq="h")
        df1 = pd.DataFrame({"res": [1.0, 2.0]}, index=t)
        df2 = pd.DataFrame({"res": [3.0, 4.0]}, index=t)
        df3 = pd.DataFrame({"res": [5.0, 6.0]}, index=t)
        results = pd.Series([df1, df2, df3])

        param_names = ["p1", "p2"]
        param_values = np.array(
            [
                [1.1, 2.2],
                [3.3, 4.4],
                [5.5, 6.6],
            ]
        )

        agg_sum = aggregate_time_series(
            results, indicator="res", method="sum", prefix="sum"
        )
        agg_mean = aggregate_time_series(
            results, indicator="res", method="mean", prefix="mean"
        )
        aggregated = pd.concat([agg_sum, agg_mean], axis=1)

        fig = plot_pcp(
            parameter_values=param_values,
            parameter_names=param_names,
            aggregated_results=aggregated,
            color_by="sum_res",
            title="Parallel Coordinates — Samples",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        pc = fig.data[0]
        np.testing.assert_allclose(pc.dimensions[0]["values"], [1.1, 3.3, 5.5])  # p1

    def test_plot_pcp_in_sampler(self):
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=Pymodel(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42, simulate=True)

        fig = sampler.plot_pcp(
            indicator="res",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_plot_sample(self):
        t = pd.date_range("2025-01-01 00:00:00", periods=2, freq="h")
        df1 = pd.DataFrame({"res": [1.0, 2.0]}, index=t)
        df2 = pd.DataFrame({"res": [3.0, 4.0]}, index=t)
        df3 = pd.DataFrame({"res": [5.0, 6.0]}, index=t)
        results = pd.Series([df1, df2, df3])
        ref = pd.Series([2.0, 2.0], index=t)

        param_names = ["p1", "p2"]
        param_values = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])

        fig = plot_sample(
            results=results,
            reference_timeseries=ref,
            title="test",
            x_label="time",
            y_label="value",
            alpha=0.3,
            show_legends=True,
            parameter_values=param_values,
            parameter_names=param_names,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4
        np.testing.assert_allclose(fig.data[0]["y"], df1["res"].to_numpy())
        np.testing.assert_allclose(fig.data[-1]["y"], ref.to_numpy())

        assert fig.data[0].name == "p1: 1.1, p2: 2.2"
        assert fig.data[1].name == "p1: 3.3, p2: 4.4"
        assert fig.data[2].name == "p1: 5.5, p2: 6.6"

        df_multi = pd.concat(
            [df1.rename(columns={"res": "a"}), df2.rename(columns={"res": "b"})], axis=1
        )
        results_multi = pd.Series([df_multi])
        with pytest.raises(ValueError, match="Provide `indicator`: multiple columns"):
            plot_sample(results=results_multi)
        fig_a = plot_sample(results=results_multi, indicator="a")
        y_values = np.array(fig_a.data[0].y)
        np.testing.assert_allclose(y_values, df_multi["a"].to_numpy())

        # empty Series
        with pytest.raises(ValueError):
            plot_sample(results=pd.Series(dtype=object))

        # Partial simulation
        empty_df = pd.DataFrame({"res": []})
        results_partial = pd.Series([empty_df, df1, empty_df])
        fig_partial = plot_sample(results=results_partial, indicator="res")

        # Only 1 non-empty sample
        assert len(fig_partial.data) == 1
        np.testing.assert_allclose(fig_partial.data[0]["y"], df1["res"].to_numpy())

        # All results empty and no reference
        results_all_empty = pd.Series([empty_df, empty_df])
        with pytest.raises(ValueError, match="No simulated data available to plot."):
            plot_sample(results_all_empty, indicator="res")

        # All results empty but with reference
        fig_ref_only = plot_sample(
            results_all_empty, indicator="res", reference_timeseries=ref
        )
        assert len(fig_ref_only.data) == 1
        np.testing.assert_allclose(fig_ref_only.data[0]["y"], ref.to_numpy())

    def test_plot_sample_in_sampler(self):
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=Pymodel(),
            simulation_options=SIMULATION_OPTIONS,
        )
        sampler.add_sample(3, 42, simulate=True)

        fig = sampler.plot_sample(
            indicator="res",
            reference_timeseries=None,
            title="test",
            x_label="time",
            y_label="value",
            alpha=0.3,
            show_legends=True,
        )
        fig.show()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
        assert fig.layout.title.text == "test"

    def test_plot_sample_infer_indicator_from_reference_name(self):
        t = pd.date_range("2025-01-01 00:00:00", periods=3, freq="h")
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}, index=t)
        results = pd.Series([df])

        ref = pd.Series([0.0, 0.0, 0.0], index=t, name="a")  # <- nom = "a"

        fig = plot_sample(
            results=results,
            reference_timeseries=ref,
            show_legends=False,
        )
        assert len(fig.data) == 2

        np.testing.assert_allclose(np.array(fig.data[0].y), df["a"].to_numpy())
        assert fig.data[0].mode == "markers"
        np.testing.assert_allclose(np.array(fig.data[1].y), ref.to_numpy())
        assert fig.data[1].mode == "lines"

    def test_sample(self):
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
        assert sample.values.tolist() == [[1.0, 0.9, 10.0], [3.0, 0.85, 20.0]]
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
        np.testing.assert_allclose(sample.values[0], [9.9, 1.1, 88])
        assert not sample.results.iloc[0].empty

        dimless_val = sample.get_dimension_less_values()
        np.testing.assert_allclose(
            dimless_val, np.array([[0.99, 0.75, 0.88], [0.3, 0.125, 0.2]])
        )

        pd.testing.assert_frame_equal(
            sample.get_aggregate_time_series("res"),
            pd.DataFrame([123.0, 42.0], [0, 1], columns=["aggregated_res"]),
        )

        fig = sample.plot_hist("res")
        assert fig.layout.title["text"] == "Sample distribution of mean res"
        assert fig.layout.xaxis.title["text"] == "mean res "

        fig = sample.plot("res")
        assert fig

        sample._validate()

    def test_lhs_sampler(self):
        sampler = LHSSampler(
            parameters=REAL_PARAM,
            model=Pymodel(),
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

        assert {k: i.values.tolist() for k, i in sampler.results.items()} == {
            0: [[85.75934698790918]],
            1: [[38.08478803524709]],
            2: [[61.67268698504139]],
        }

        sampler.add_sample(3, rng=42, simulate=False)
        assert sampler.values.shape == (6, 3)
        assert all(df.empty for df in sampler.results[-3:].values)

        sampler.simulate_at(4)
        assert [df.empty for df in sampler.results[-3:].values] == [True, False, True]

        sampler.simulate_at([3, 5])
        assert {k: i.values.tolist() for k, i in sampler.results[-3:].items()} == {
            3: [[85.75934698790918]],
            4: [[38.08478803524709]],
            5: [[61.67268698504139]],
        }

        sampler.add_sample(3, rng=42, simulate=False)

        sampler.simulate_at(slice(4, 7))
        assert [df.empty for df in sampler.results[-3:].values] == [False, True, True]

        sampler.add_sample(3, simulate=False)
        sampler.simulate_at(slice(10, None))
        assert [df.empty for df in sampler.results[-3:].values] == [True, False, False]

        sampler.add_sample(3, simulate=True)
        assert all(not df.empty for df in sampler.results[-3:])

    def test_morris_sampler(self):
        sampler = MorrisSampler(
            parameters=ISHIGAMI_PARAMETERS,
            model=Ishigami(),
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
            model=Ishigami(),
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


class ModelVariant(Model):
    def __init__(self):
        self.y1 = 1
        self.z1 = 2
        self.multiplier = 1

    def simulate(
        self, property_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        if property_dict is None:
            property_dict = {"discrete1": 1, "discrete2": 2}

        result = (
            self.y1 * property_dict["discrete1"] + self.z1 * property_dict["discrete2"]
        ) * self.multiplier

        df = pd.DataFrame(
            {"result": [result]},
            index=pd.date_range(
                simulation_options["start"],
                simulation_options["end"],
                freq=simulation_options["timestep"],
            ),
        )

        return df

    def save(self, file_path: Path):
        pass


class Simul(Model):
    def simulate(
        self, property_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        Parameter1, Parameter2, Parameter3, Parameter4 = map(
            float, property_dict.values()
        )
        df = Parameter1 + Parameter2 - Parameter3 + 2 * Parameter4
        df_out = pd.DataFrame({"df": [df]})
        return df_out

    def save(self, file_path: Path):
        pass


def modifier_1(model, description, multiplier=None):
    model.y1 = description["y1"]
    if multiplier is not None:
        model.multiplier = multiplier


def modifier_2(model, description):
    model.z1 = description["z1"]


class TestVariantSubSampler:
    def setup_method(self):
        self.variant_dict = {
            "EXISTING_mod1": {
                VariantKeys.MODIFIER: "mod1",
                VariantKeys.ARGUMENTS: {"multiplier": 1},
                VariantKeys.DESCRIPTION: {"y1": 1},
            },
            "Variant_1": {
                VariantKeys.MODIFIER: "mod1",
                VariantKeys.ARGUMENTS: {"multiplier": 2},
                VariantKeys.DESCRIPTION: {"y1": 20},
            },
            "EXISTING_mod1_bis": {
                VariantKeys.MODIFIER: "mod1_bis",
                VariantKeys.ARGUMENTS: {"multiplier": 1},
                VariantKeys.DESCRIPTION: {"y1": 1},
            },
            "Variant_2": {
                VariantKeys.MODIFIER: "mod1_bis",
                VariantKeys.ARGUMENTS: {},
                VariantKeys.DESCRIPTION: {"y1": 30},
            },
            "EXISTING_mod2": {
                VariantKeys.MODIFIER: "mod2",
                VariantKeys.ARGUMENTS: {},
                VariantKeys.DESCRIPTION: {"z1": 2},
            },
            "Variant_3": {
                VariantKeys.MODIFIER: "mod2",
                VariantKeys.ARGUMENTS: {},
                VariantKeys.DESCRIPTION: {"z1": 40},
            },
        }

        self.combinations = get_combined_variants(self.variant_dict)
        self.mod_map = {"mod1": modifier_1, "mod1_bis": modifier_1, "mod2": modifier_2}

        self.sampler = VariantSubSampler(
            model=VariantModel(),
            custom_combination=self.combinations,
            variant_dict=self.variant_dict,
            modifier_map=self.mod_map,
            simulation_options=SIMULATION_OPTIONS,
        )

        self.sampler_no_custom = VariantSubSampler(
            model=VariantModel(),
            variant_dict=self.variant_dict,
            modifier_map=self.mod_map,
            simulation_options=SIMULATION_OPTIONS,
        )

        self.variant_dict_for_param = {
            "Variant_1": {
                VariantKeys.MODIFIER: "mod1",
                VariantKeys.DESCRIPTION: {"y1": 20},
                VariantKeys.ARGUMENTS: {"multiplier": 2},
            },
            "Variant_2": {
                VariantKeys.MODIFIER: "mod2",
                VariantKeys.DESCRIPTION: {"z1": 40},
                VariantKeys.ARGUMENTS: {},
            },
            "Variant_3": {
                VariantKeys.MODIFIER: "mod1",
                VariantKeys.DESCRIPTION: {"y1": 10},
                VariantKeys.ARGUMENTS: {"multiplier": 3},
            },
        }

        self.modifier_map_for_param = {
            "mod1": modifier_1,
            "mod2": modifier_2,
        }

    def test_add_sample_with_no_custom_combinations(self):
        initial_sample_size = 10
        self.sampler_no_custom.add_sample(
            initial_sample_size,
            n_cpu=1,
        )

        assert len(self.sampler_no_custom.sample_results) > 0

        expected_list = [5, 81, 34, 110, 5, 81, 68, 220]
        actual_results = []
        for df in self.sampler_no_custom.sample_results:
            actual_results.extend(df["res"].tolist())

        assert set(actual_results) == set(expected_list)

    def test_add_sample_with_simulation(self):
        initial_sample_size = 10
        self.sampler.add_sample(
            initial_sample_size,
            n_cpu=1,
        )

        assert len(self.sampler.sample_results) > 0

        expected_list = [5, 81, 34, 110, 5, 81, 68, 220]
        actual_results = []
        for df in self.sampler.sample_results:
            actual_results.extend(df["res"].tolist())

        assert set(actual_results) == set(expected_list)

    def test_clear_sample(self):
        self.sampler.add_sample(1, simulate=False, ensure_full_coverage=True, n_cpu=1)
        self.sampler.clear_sample()
        assert len(self.sampler.sample) == 0
        assert len(self.sampler.sample_results) == 0
        assert len(self.sampler.not_simulated_combinations) == 0

    def test_dump_sample(self):
        n_sim = 1
        n_non_sim = 4
        self.sampler.add_sample(
            n_sim, simulate=True, ensure_full_coverage=False, n_cpu=1
        )
        self.sampler.draw_sample(n_non_sim)
        assert len(self.sampler.sample) == n_sim + n_non_sim
        assert len(self.sampler.simulated_samples) == n_sim
        self.sampler.dump_sample()
        assert len(self.sampler.sample) == n_sim
        assert len(self.sampler.not_simulated_combinations) == 0

    def test_draw_sample(self):
        sample_size = 2
        self.sampler.draw_sample(sample_size, ensure_full_coverage=True)

        covered_variants = set(itertools.chain(*self.sampler.sample))
        all_variants = set(itertools.chain(*self.combinations))
        assert covered_variants == all_variants
        assert len(self.sampler.sample) >= sample_size

    def test_add_sample_without_simulation(self):
        self.sampler.clear_sample()

        initial_sample_size = 2
        self.sampler.add_sample(initial_sample_size, simulate=False)
        assert len(self.sampler.sample) >= initial_sample_size
        assert len(self.sampler.not_simulated_combinations) == initial_sample_size

    def test_draw_sample_then_simulate(self):
        self.sampler.clear_sample()
        sample_size = 2
        self.sampler.draw_sample(sample_size, ensure_full_coverage=False, seed=42)
        self.sampler.simulate_combinations(n_cpu=1)
        assert len(self.sampler.sample) == sample_size
        expected_list = [5, 110]

        actual_results = []
        for df in self.sampler.sample_results:
            actual_results.extend(df["res"].tolist())

        assert set(actual_results) == set(expected_list)

    def test_warning_errors(self):
        sampler_warning = VariantSubSampler(
            model=VariantModel(),
            variant_dict=self.variant_dict,
            modifier_map=self.mod_map,
        )
        with pytest.raises(ValueError):
            sampler_warning.add_sample(sample_size=2, n_cpu=1)

        sampler_warning.add_sample(sample_size=2, simulate=False, simulation_options={})
        with pytest.raises(ValueError):
            sampler_warning.simulate_combinations(n_cpu=1)

    def test_seed_consistency(self):
        sampler1 = VariantSubSampler(
            model=Simul(), simulation_options={}, custom_combination=self.combinations
        )
        sampler2 = VariantSubSampler(
            model=Simul(), simulation_options={}, custom_combination=self.combinations
        )
        seed = 42
        sample_size = 2

        sampler1.add_sample(sample_size, seed=seed, simulate=False)
        sampler2.add_sample(sample_size, seed=seed, simulate=False)
        expected_list = [
            ("EXISTING_mod1", "Variant_2", "Variant_3"),
            ("Variant_1", "EXISTING_mod1_bis", "EXISTING_mod2"),
        ]
        assert list(sampler1.sample) == expected_list
        assert list(sampler1.sample) == list(sampler2.sample)

        sampler1.clear_sample()
        sampler2.clear_sample()

        new_seed = 43
        sampler1.add_sample(sample_size, seed=seed, simulate=False)
        sampler2.add_sample(sample_size, seed=new_seed, simulate=False)
        assert sampler1.sample != sampler2.sample

    def test_ensure_full_coverage(self):
        n_sample = 3
        self.sampler.add_sample(
            n_sample,
            simulate=False,
            ensure_full_coverage=True,
            n_cpu=1,
        )
        covered_variants = set(itertools.chain(*self.sampler.sample))
        all_variants = set(itertools.chain(*self.combinations))
        assert covered_variants == all_variants
        assert len(self.sampler.sample) >= n_sample

    def test_simulate_combined_variants_with_custom_combi(self):
        custom_combinations = [("Variant_1", "Variant_2"), ("Variant_3", "Variant_2")]
        sampler = VariantSubSampler(
            model=ModelVariant(),
            variant_dict=self.variant_dict_for_param,
            modifier_map=self.modifier_map_for_param,
            custom_combination=custom_combinations,
            simulation_options=simulation_options_for_param,
        )

        sampler.simulate_all_variants_and_parameters(
            parameter=discrete_parameters, param_mapping=params_mappings
        )

        expected_number_of_simulations = pow(2, len(discrete_parameters)) * len(
            custom_combinations
        )
        assert len(sampler.sample_results) == expected_number_of_simulations

        expected_first_result = (
            20 * 0 + 40 * 1
        ) * 2  # y1×discrete1+z1×discrete2)×multiplier

        first_result = sampler.sample_results[0].iloc[0, 0]
        assert first_result == expected_first_result

    def test_simulate_combined_variants(self):
        sampler = VariantSubSampler(
            model=ModelVariant(),
            variant_dict=self.variant_dict_for_param,
            modifier_map=self.modifier_map_for_param,
            simulation_options=simulation_options_for_param,
        )
        sampler.simulate_all_variants_and_parameters(
            parameter=discrete_parameters, param_mapping=params_mappings
        )

        mod_dict = get_modifier_dict(self.variant_dict_for_param)
        total_combinations = 1
        for key, values in mod_dict.items():
            total_combinations *= len(values)
        num_discrete_combinations = pow(2, len(discrete_parameters))
        expected_number_of_simulations = num_discrete_combinations * total_combinations

        assert len(sampler.sample_results) == expected_number_of_simulations

        expected_first_result = (
            20 * 0 + 40 * 1
        ) * 2  # (y1 * discrete1 + z1 * discrete2) * multiplier

        first_result = sampler.sample_results[0].iloc[0, 0]
        assert first_result == expected_first_result


class TestSimulationSampler:
    def test_draw_sample_only(self):
        Simulator = Simul()
        sampler = Sampler(parameters, model=Simulator)
        sampler.draw_sample(sample_size=5, seed=42)

        assert sampler.sample.shape[0] >= 5
        assert len(sampler.sample_results) == 0

    def test_draw_sample_then_simulate_drawn_samples(self):
        Simulator = Simul()
        sampler = Sampler(parameters, model=Simulator)
        sampler.draw_sample(sample_size=3, seed=0)

        assert len(sampler.sample_results) == 0

        sampler.simulate_drawn_samples()
        assert len(sampler.sample_results) == sampler.sample.shape[0]

    def test_get_boundary_sample(self):
        sampler = Sampler(parameters, model=None)

        boundary_sample = sampler.get_boundary_sample()

        assert isinstance(boundary_sample, np.ndarray)
        assert boundary_sample.shape == (len(parameters), 2)

        for i, param in enumerate(parameters):
            lower_bound = param[Parameter.INTERVAL][0]
            if param[Parameter.TYPE] == "Choice":
                lower_bound = min(param[Parameter.INTERVAL])
            upper_bound = param[Parameter.INTERVAL][-1]
            if param[Parameter.TYPE] == "Choice":
                upper_bound = max(param[Parameter.INTERVAL])
            for j in range(2):  # Loop through lower and upper bounds
                assert lower_bound <= boundary_sample[i, j] <= upper_bound

    def test_add_sample(self):
        Simulator = Simul()
        sampler = Sampler(parameters, model=Simulator)
        sample_size = 10
        sampler.add_sample(sample_size)

        assert len(sampler.sample_results) == sample_size + 2

        new_sample_size = 5
        sampler.add_sample(new_sample_size)

        assert len(sampler.sample_results) == sample_size + new_sample_size + 2

        for sample_values in sampler.sample:
            for par, val in zip(parameters, sample_values):
                interval = par[Parameter.INTERVAL]
                if par[Parameter.TYPE] == "Real":
                    assert interval[0] <= val <= interval[1]
                elif par[Parameter.TYPE] == "Integer":
                    assert interval[0] <= val <= interval[1]
                    assert val.is_integer()
                elif par[Parameter.TYPE] == "Choice":
                    assert val in interval
                elif par[Parameter.TYPE] == "Binary":
                    assert val in [0, 1]

        sampler.clear_sample()
        sampler.add_sample(1, seed=42)

        expected_sample = np.array(
            [[1.35626, 2, 0.2, 1], [0, 1, 0.02, 0], [6, 4, 0.2, 1]]
        )
        np.testing.assert_allclose(
            sampler.sample, expected_sample, atol=1e-8, rtol=1e-5
        )

        expected_result = [
            pd.DataFrame([5.156264], columns=["df"]),
            pd.DataFrame([0.98], columns=["df"]),
            pd.DataFrame([11.8], columns=["df"]),
        ]
        np.testing.assert_allclose(
            sampler.sample_results, expected_result, atol=1e-8, rtol=1e-5
        )

    def test_clear_sample(self):
        Simulator = Simul()
        sampler = Sampler(parameters, model=Simulator)
        sampler.add_sample(1)
        sampler.clear_sample()

        assert np.array_equal(sampler.sample, np.empty(shape=(0, len(parameters))))

    def test_simulate_all_combinations(self):
        sampler = Sampler(discrete_parameters, model=Simul())

        sampler.simulate_all_combinations()

        assert len(sampler.sample_results) == pow(2, len(discrete_parameters))


class TestPlotPCP:
    def setup_method(self):
        self.sample_results = [
            pd.DataFrame({"HEATING_Energy_[J]": [100, 200, 300]}),
            pd.DataFrame({"HEATING_Energy_[J]": [150, 250, 350]}),
        ]
        self.param_sample = np.array([[0.5, "bad"], [0.3, "good"]])

        self.parameters = [
            {
                Parameter.NAME: "param1",
                Parameter.INTERVAL: [0, 1],
                Parameter.TYPE: "Real",
            },
            {
                Parameter.NAME: "param2",
                Parameter.INTERVAL: ["bad", "good"],
                Parameter.TYPE: "Real",
            },
        ]

        self.indicators = ["HEATING_Energy_[J]"]

    def test_plot_pcp(self):
        fig = plot_pcp(
            sample_results=self.sample_results,
            param_sample=self.param_sample,
            parameters=self.parameters,
            indicators=self.indicators,
        )

        fig.show()

        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].dimensions) == len(self.parameters) + len(
            self.indicators
        )
