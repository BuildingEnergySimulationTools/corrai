import datetime as dt
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from corrai.fmu import ModelicaFmuModel

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
        "stepSize": 1,
        "solver": "CVode",
        "outputInterval": 1,
        "tolerance": 1e-6,
    }

    outputs = ["res.showNumber"]

    simu = ModelicaFmuModel(
        fmu_path=PACKAGE_DIR / "rosen.fmu",
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
    simu = ModelicaFmuModel(
        fmu_path=PACKAGE_DIR / "TestLib.boundary_test.fmu",
        simulation_options=simul_options,
        output_list=outputs,
    )
    return simu


@pytest.fixture()
def simul_boundaries():
    simu = ModelicaFmuModel(
        fmu_path=PACKAGE_DIR / "boundary_test.fmu",
        simulation_options=simul_options,
        output_list=outputs,
    )
    return simu


class TestFmu:
    def test_results(self, simul):
        res = simul.simulate()
        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        assert np.allclose(res["res.showNumber"].values, ref["res.showNumber"].values)

    if system == "Windows":
        # Because issues with relatives filepaths exporting FMUs from OM.
        def test_simulate(self, simul_boundaries, simul_boundaries_int):
            new_bounds = pd.DataFrame(
                {"Boundaries.y[1]": [1, 2, 3], "Boundaries.y[2]": [3, 4, 5]},
                index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
            )
            new_bounds.index.freq = None
            new_bounds = new_bounds.astype(float)

            res1 = simul_boundaries_int.simulate()
            res2 = simul_boundaries.simulate(x=new_bounds)

            assert np.allclose(res1.values, res2.values)

            x_datetime = pd.DataFrame(
                {
                    "Boundaries.y[1]": [1, 2, 3, 4, 5],
                    "Boundaries.y[2]": [3, 4, 5, 6, 7],
                },
                index=pd.date_range("2009-07-13 00:00:00", periods=5, freq="h"),
            )

            simul_boundaries.simulate(
                x=x_datetime,
                simulation_options={
                    "startTime": dt.datetime(
                        year=2009, month=7, day=13, hour=1, minute=0, second=0
                    ),
                    "stopTime": dt.datetime(
                        year=2009, month=7, day=13, hour=2, minute=0, second=0
                    ),
                },
            )

            with pytest.raises(TypeError):
                simul_boundaries.simulate(
                    x=x_datetime, simulation_options={"startTime": 1, "stopTime": 2.0}
                )

    def test_save(self, simul):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_model.fmu")

            simul.save(Path(file_path))
            assert os.path.exists(temp_dir)
            assert "test_model.fmu" in os.listdir(temp_dir)

            # os.remove(temp_dir)
