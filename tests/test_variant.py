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

        assert list(pd.concat(res, axis=1).max()) == [5, 81, 34, 110, 48, 200, 68, 220]

        # Parallel
        res = simulate_variants(
            model=model,
            variant_dict=VARIANT_DICT,
            modifier_map=MODIFIER_MAP,
            simulation_options=SIMULATION_OPTIONS,
            n_cpu=-1,
        )

        assert list(pd.concat(res, axis=1).max()) == [5, 81, 34, 110, 48, 200, 68, 220]


# VARIANT_DICT = {
#     "EEM1_Wall_int_insulation": {
#         VariantKeys.MODIFIER: "walls",
#         VariantKeys.ARGUMENTS: {"boundaries": "external"},
#         VariantKeys.DESCRIPTION: [
#             {
#                 "Name": "Project medium concrete block_.2",
#                 "Thickness": 0.2,
#                 "Conductivity": 0.51,
#                 "Density": 1400,
#                 "Specific_Heat": 1000,
#             },
#             {
#                 "Name": "Laine_15cm",
#                 "Thickness": 0.15,
#                 "Conductivity": 0.032,
#                 "Density": 40,
#                 "Specific_Heat": 1000,
#             },
#         ],
#     },
#     "EEM2_Wall_ext_insulation": {
#         VariantKeys.MODIFIER: "wall",
#         VariantKeys.ARGUMENTS: {"names": "Ext_South"},
#         VariantKeys.DESCRIPTION: [
#             # Outside Layer
#             {
#                 "Name": "Coating",
#                 "Thickness": 0.01,
#                 "Conductivity": 0.1,
#                 "Density": 400,
#                 "Specific_Heat": 1200,
#             },
#             {
#                 "Name": "Laine_30cm",
#                 "Thickness": 0.30,
#                 "Conductivity": 0.032,
#                 "Density": 40,
#                 "Specific_Heat": 1000,
#             },
#             {
#                 "Name": "Project medium concrete block_.2",
#                 "Thickness": 0.2,
#                 "Conductivity": 0.51,
#                 "Density": 1400,
#                 "Specific_Heat": 1000,
#             },
#         ],
#     },
#     "EEM3_Double_glazing": {
#         VariantKeys.MODIFIER: "windows",
#         VariantKeys.ARGUMENTS: {},
#         VariantKeys.DESCRIPTION: {
#             "Name": "Double_glazing",
#             "UFactor": 1.1,
#             "Solar_Heat_Gain_Coefficient": 0.41,
#             "Visible_Transmittance": 0.71,
#         },
#     },
# }
