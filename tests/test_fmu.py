import pandas as pd
import numpy as np
from pathlib import Path
import pytest

import platform

from corrai.fmu import FmuModel

system = platform.system()

if system == "Windows":
    PACKAGE_DIR = Path(__file__).parent / "resources/fmu_Windows"
elif system == "Linux":
    PACKAGE_DIR = Path(__file__).parent / "resources/fmu_Linux"
else:
    raise NotImplementedError(f"Unsupported operating system: {system}")


@pytest.fixture()
def simul():
    simul_options = {
        "startTime": 0,
        "stopTime": 2,
        "solver": "CVode",
        "outputInterval": 1,
        "tolerance": 1e-6,
    }

    outputs = ["res.showNumber"]

    simu = FmuModel(
        model_path=PACKAGE_DIR / "rosen.fmu",
        simulation_options=simul_options,
        output_list=outputs,
    )
    return simu


simul_options = {
    "startTime": 16675200,
    "stopTime": 16682400,
    "outputInterval": 3600,
    "solver": "Euler",
    "stepSize": 3600,
    "tolerance": 1e-6,
    "fmi_type": "ModelExchange",
}

outputs = ["Boundaries.y[1]", "Boundaries.y[2]"]


@pytest.fixture()
def simul_boundaries_int():
    simu = FmuModel(
        model_path=PACKAGE_DIR / "TestLib.boundary_test.fmu",
        simulation_options=simul_options,
        output_list=outputs,
    )
    return simu


@pytest.fixture()
def simul_boundaries():
    simu = FmuModel(
        model_path=PACKAGE_DIR / "boundary_test.fmu",
        simulation_options=simul_options,
        output_list=outputs,
    )
    return simu


class TestFmu:
    def test_set_param_dict(self, simul):
        test_dict = {
            "x.k": 2.0,
            "y.k": 3.0,
        }
        simul.set_param_dict(test_dict)
        modified_params = simul.check_parameter_modifications()
        assert test_dict == modified_params

    def test_results(self, simul):
        res = simul.simulate()
        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        assert np.allclose(res["res.showNumber"].values, ref["res.showNumber"].values)

    if system == "Windows":
        # Because issues with relatives filepaths exporting FMUs from OM.
        def test_set_boundaries_df(self, simul_boundaries, simul_boundaries_int):
            new_bounds = pd.DataFrame(
                {"Boundaries.y[1]": [1, 2, 3], "Boundaries.y[2]": [3, 4, 5]},
                index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="H"),
            )
            new_bounds.index.freq = None
            new_bounds = new_bounds.astype(float)

            res1 = simul_boundaries_int.simulate()

            simul_boundaries.set_boundaries_df(new_bounds)
            res2 = simul_boundaries.simulate()

            assert np.allclose(res1.values, res2.values)

    def test_set_boundaries_value_error(self, simul_boundaries):
        invalid_df = pd.DataFrame(
            {"Boundaries.y[1]": [1, 2, 3], "Boundaries.y[2]": [3, 4, 5]},
            index=pd.Index([1, 2, 3]),
        )
        invalid_df = invalid_df.astype(float)

        with pytest.raises(ValueError):
            simul_boundaries.set_boundaries_df(invalid_df)
