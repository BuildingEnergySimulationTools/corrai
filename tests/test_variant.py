from corrai.variant import (
    simulate_variants,
    VariantKeys,
    get_combined_variants,
    get_modifier_dict,
)

import pandas as pd
from tests.resources.pymodels import VariantModel


def modifier_1(model, description, multiplier=None):
    model.y1 = description["y1"]
    if multiplier is not None:
        model.multiplier = multiplier


def modifier_2(model, description):
    model.z1 = description["z1"]


VARIANT_DICT = {
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
    "end": "2009-01-01 02:00:00",
    "timestep": "H",
}


class TestVariant:
    def test_variant(self):
        modifier_dict = get_modifier_dict(VARIANT_DICT)
        assert modifier_dict == {
            "mod1": ["EXISTING_mod1", "Variant_1"],
            "mod1_bis": ["EXISTING_mod1_bis", "Variant_2"],
            "mod2": ["EXISTING_mod2", "Variant_3"],
        }

        variant_list = get_combined_variants(VARIANT_DICT)
        assert variant_list == [
            ("EXISTING_mod1", "EXISTING_mod1_bis", "EXISTING_mod2"),
            ("EXISTING_mod1", "EXISTING_mod1_bis", "Variant_3"),
            ("EXISTING_mod1", "Variant_2", "EXISTING_mod2"),
            ("EXISTING_mod1", "Variant_2", "Variant_3"),
            ("Variant_1", "EXISTING_mod1_bis", "EXISTING_mod2"),
            ("Variant_1", "EXISTING_mod1_bis", "Variant_3"),
            ("Variant_1", "Variant_2", "EXISTING_mod2"),
            ("Variant_1", "Variant_2", "Variant_3"),
        ]

        model = VariantModel()

        # Sequential
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=1,
        )

        # Parallel
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=-1,
        )

        assert list(pd.concat(res, axis=1).max()) == [5, 81, 34, 110, 5, 81, 68, 220]
        # for combinations with conflictual values (several y1),
        # the last one erases the previous ones
