import os
import shutil
import tempfile
from pathlib import Path

from corrai.variant import (
    simulate_variants,
    VariantKeys,
    get_combined_variants,
    get_modifier_dict,
)
from tests.resources.pymodels import VariantModel


def modifier_1(model, description, multiplier=None):
    model.y1 = description["y1"]
    if multiplier is not None:
        model.multiplier = multiplier


def modifier_2(model, description):
    model.z1 = description["z1"]


VARIANT_DICT_true = {
    "Variant_1": {
        VariantKeys.MODIFIER: "mod1",
        VariantKeys.ARGUMENTS: {"multiplier": 2},
        VariantKeys.DESCRIPTION: {"y1": 20},
    },
    "Variant_2": {
        VariantKeys.MODIFIER: "mod1_bis",
        VariantKeys.ARGUMENTS: {},
        VariantKeys.DESCRIPTION: {"y1": 30},
    },
    "Variant_3": {
        VariantKeys.MODIFIER: "mod2",
        VariantKeys.ARGUMENTS: {},
        VariantKeys.DESCRIPTION: {"z1": 40},
    },
}

VARIANT_DICT_false = {
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

MODIFIER_MAP = {"mod1": modifier_1, "mod1_bis": modifier_1, "mod2": modifier_2}

SIMULATION_OPTIONS = {
    "start": "2009-01-01 00:00:00",
    "end": "2009-01-01 00:00:00",
    "timestep": "h",
}


class TestVariant:
    def test_variant(self):
        modifier_dict_true = get_modifier_dict(VARIANT_DICT_true, add_existing=True)
        modifier_dict_false = get_modifier_dict(VARIANT_DICT_false, add_existing=False)
        expected_dict_modifiers = {
            "mod1": ["EXISTING_mod1", "Variant_1"],
            "mod1_bis": ["EXISTING_mod1_bis", "Variant_2"],
            "mod2": ["EXISTING_mod2", "Variant_3"],
        }
        assert modifier_dict_false == expected_dict_modifiers
        assert modifier_dict_true == expected_dict_modifiers

        variant_list_false = get_combined_variants(
            VARIANT_DICT_false, add_existing=False
        )
        variant_list_true = get_combined_variants(VARIANT_DICT_true, add_existing=True)

        expected_variant_list = [
            ("EXISTING_mod1", "EXISTING_mod1_bis", "EXISTING_mod2"),  # 5
            ("EXISTING_mod1", "EXISTING_mod1_bis", "Variant_3"),  # 81
            ("EXISTING_mod1", "Variant_2", "EXISTING_mod2"),  # 34
            ("EXISTING_mod1", "Variant_2", "Variant_3"),  # 110
            ("Variant_1", "EXISTING_mod1_bis", "EXISTING_mod2"),  # 5 # 48
            ("Variant_1", "EXISTING_mod1_bis", "Variant_3"),  # 81  # 200
            ("Variant_1", "Variant_2", "EXISTING_mod2"),  # 68
            ("Variant_1", "Variant_2", "Variant_3"),  # 220
        ]

        assert variant_list_true == expected_variant_list
        assert variant_list_false == expected_variant_list

        model = VariantModel()

        expected_list = [5, 81, 34, 110, 5, 81, 68, 220]

        # Sequential
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT_false,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=1,
            add_existing=False,
        )
        actual_results_f = []
        for df in res:
            actual_results_f.extend(df["res"].tolist())
        assert actual_results_f == expected_list

        # Parallel
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT_false,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=-1,
        )

        actual_results = []
        for df in res:
            actual_results.extend(df["res"].tolist())
        assert actual_results == expected_list
        # for combinations with conflictual values (several y1),
        # the last one erases the previous ones

        # Add existing True
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT_true,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=1,
            add_existing=True,
        )

        # different from previous one as existing are not
        # taken into account when applying modifiers
        expected_list = [5, 81, 34, 110, 48, 200, 68, 220]

        actual_results_t = []
        for df in res:
            actual_results_t.extend(df["res"].tolist())
        assert actual_results_t == expected_list

    def test_custom_combination(self):
        model = VariantModel()

        custom_combination = [
            ("EXISTING_mod1", "Variant_2", "Variant_3"),  # 110
            ("Variant_1", "EXISTING_mod1_bis", "EXISTING_mod2"),  # 5
        ]

        expected_list = [110, 5]

        # Sequential
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT_false,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=1,
            custom_combinations=custom_combination,
            add_existing=False,
        )

        actual_results_t = []
        for df in res:
            actual_results_t.extend(df["res"].tolist())

        assert actual_results_t == expected_list

    def test_save_path(self):
        model = VariantModel()
        variant_dict = {
            "Variant_1": {
                VariantKeys.MODIFIER: "mod1",
                VariantKeys.ARGUMENTS: {"multiplier": 2},
                VariantKeys.DESCRIPTION: {"y1": 20},
            }
        }

        modifier_map = {"mod1": modifier_1}
        simulation_options = {
            "start": "2009-01-01 00:00:00",
            "end": "2009-01-01 02:00:00",
            "timestep": "h",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir)

            simulate_variants(
                model=model,
                variant_dict=variant_dict,
                modifier_map=modifier_map,
                simulation_options=simulation_options,
                save_dir=save_path,
            )

            assert os.path.exists(save_path)
            assert os.path.exists(save_path / "Model_1.txt")
            shutil.rmtree(save_path)
