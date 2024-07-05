import itertools
from pathlib import Path
from corrai.base.model import Model
import numpy as np
import pandas as pd
from corrai.learning.sampling import ModelSampler
from corrai.base.parameter import Parameter
from corrai.learning.sampling import VariantSubSampler
from corrai.variant import VariantKeys, get_combined_variants
from tests.resources.pymodels import VariantModel
import pytest

FILES_PATH = Path(__file__).parent / "resources"

parameters = [
    {Parameter.NAME: "Parameter1", Parameter.INTERVAL: (0, 6), Parameter.TYPE: "Real"},
    {
        Parameter.NAME: "Parameter2",
        Parameter.INTERVAL: (1, 4),
        Parameter.TYPE: "Integer",
    },
    {
        Parameter.NAME: "Parameter3",
        Parameter.INTERVAL: (0.02, 0.04, 0.2),
        Parameter.TYPE: "Choice",
    },
    {
        Parameter.NAME: "Parameter4",
        Parameter.INTERVAL: (0, 1),
        Parameter.TYPE: "Binary",
    },
]

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 00:00:00",
    "timestep": "h",
}


class Simul(Model):
    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        Parameter1, Parameter2, Parameter3, Parameter4 = parameter_dict.values()
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
            combinations=self.combinations,
            variant_dict=self.variant_dict,
            modifier_map=self.mod_map,
            simulation_options=SIMULATION_OPTIONS,
        )

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
            combinations=self.combinations,
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
            model=Simul(), simulation_options={}, combinations=self.combinations
        )
        sampler2 = VariantSubSampler(
            model=Simul(), simulation_options={}, combinations=self.combinations
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


class TestSimulationSampler:
    def test_get_boundary_sample(self):
        sampler = ModelSampler(parameters, model=None)

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
        sampler = ModelSampler(parameters, model=Simulator)
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
        sampler = ModelSampler(parameters, model=Simulator)
        sampler.add_sample(1)
        sampler.clear_sample()

        assert np.array_equal(sampler.sample, np.empty(shape=(0, len(parameters))))
