import datetime as dt
import os
import platform
import tempfile
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from corrai.fmu import ModelicaFmuModel
from corrai.base.parameter import Parameter

system = platform.system()

if system == "Windows":
    PACKAGE_DIR = Path(__file__).parent / "resources/fmu_Windows"
elif system == "Linux":
    PACKAGE_DIR = Path(__file__).parent / "resources/fmu_Linux"
else:
    raise NotImplementedError(f"Unsupported operating system: {system}")


class TestFmu:
    def test_results(self):
        simu = ModelicaFmuModel(
            fmu_path=PACKAGE_DIR / "rosen.fmu",
            simulation_options={
                "startTime": 0,
                "stopTime": 2,
                "stepSize": 1,
                "solver": "CVode",
                "outputInterval": 1,
                "tolerance": 1e-6,
            },
            output_list=["res.showNumber"],
        )

        res = simu.simulate()
        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        assert np.allclose(res["res.showNumber"].values, ref["res.showNumber"].values)

    if system == "Windows":
        # Because issues with relatives filepaths exporting FMUs from OM.
        def test_simulate(self):
            simu = ModelicaFmuModel(
                fmu_path=PACKAGE_DIR / "boundary_test.fmu",
                output_list=["Boundaries.y[1]", "Boundaries.y[2]"],
                boundary_table="Boundaries",
            )

            new_bounds = pd.DataFrame(
                {"Boundaries.y[1]": [1, 2, 3], "Boundaries.y[2]": [3, 4, 5]},
                index=range(3, 6),
            )

            new_bounds.index.freq = None
            new_bounds = new_bounds.astype(float)

            res = simu.simulate(
                simulation_options={
                    "solver": "CVode",
                    "outputInterval": 1,
                    "stepSize": 1,
                    "boundary": new_bounds,
                },
                solver_duplicated_keep="last",
            )

            assert res.to_dict() == {
                "Boundaries.y[1]": {
                    3.0: 1.0000000000000333,
                    4.0: 2.0000000000000453,
                    5.0: 3.0,
                },
                "Boundaries.y[2]": {
                    3.0: 3.0000000000000333,
                    4.0: 4.000000000000045,
                    5.0: 5.0,
                },
            }

            x_datetime = pd.DataFrame(
                {
                    "Boundaries.y[1]": [1, 2, 3, 4, 5],
                    "Boundaries.y[2]": [3, 4, 5, 6, 7],
                },
                index=pd.date_range("2009-07-13 00:00:00", periods=5, freq="h"),
            )
            x_datetime.index.name = "time"
            x_datetime.index.freq = None
            x_datetime = x_datetime.astype(float)

            res = simu.simulate(
                simulation_options={
                    "outputInterval": 3600,
                    "stepSize": 3600,
                    "boundary": x_datetime,
                },
                solver_duplicated_keep="last",
            )

            pd.testing.assert_frame_equal(res, x_datetime)

            res = simu.simulate(
                simulation_options={
                    "startTime": dt.datetime(2009, 7, 13, 0, 0, 0),
                    "stopTime": dt.datetime(2009, 7, 13, 3, 0, 0),
                    "outputInterval": 3600,
                },
                solver_duplicated_keep="last",
            )

            pd.testing.assert_frame_equal(res, x_datetime.loc[:"2009-7-13 03:00:00", :])

    def test_save(self):
        simu = ModelicaFmuModel(
            fmu_path=PACKAGE_DIR / "rosen.fmu",
            output_list=["res.showNumber"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_model.fmu")

            simu.save(Path(file_path))
            assert os.path.exists(temp_dir)
            assert "test_model.fmu" in os.listdir(temp_dir)

    def test_get_property_values(self):
        simu = ModelicaFmuModel(
            fmu_path=PACKAGE_DIR / "rosen.fmu",
            output_list=["res.showNumber"],
        )
        values = simu.get_property_values(("res.showNumber",))
        assert isinstance(values, list)
        assert len(values) == 1

        vals = simu.get_property_values("x.k")
        assert vals == ["2.0"]

        vals = simu.get_property_values(("x.k",))
        assert vals == ["2.0"]

        vals = simu.get_property_values(["x.k", "y.k"])
        assert vals == ["2.0", "2.0"]

    def test_simulate_parameter(self):
        simu = ModelicaFmuModel(
            fmu_path=PACKAGE_DIR / "rosen.fmu",
            simulation_options={"startTime": 0, "stopTime": 2, "stepSize": 1},
            output_list=["res.showNumber"],
        )

        param = [
            Parameter(name="x", model_property="x.k", interval=(0, 5), init_value=2),
            Parameter(
                name="y",
                model_property="y.k",
                interval=(0, 5),
                init_value=2,
            ),
        ]

        res1 = simu.simulate({"y.k": 4, "x.k": 3})
        res2 = simu.simulate_parameter(
            [
                (param[0], 3),
                (param[1], 4),
            ]
        )
        assert res1["res.showNumber"].equals(res2["res.showNumber"])

    def test_boundary_warning(self):
        simu = ModelicaFmuModel(
            fmu_path=PACKAGE_DIR / "rosen.fmu",
            output_list=["res.showNumber"],
            boundary_table="Boundaries",
        )

        fake_boundary = pd.DataFrame({"u": [1, 2, 3]}, index=[0, 1, 2])

        with pytest.warns(
            UserWarning,
            match="Boundary combitimetable 'Boundaries' "
            "not found in FMU -> ignoring boundary.",
        ):
            res = simu.simulate(
                simulation_options={
                    "startTime": 0,
                    "stopTime": 2,
                    "stepSize": 1,
                    "boundary": fake_boundary,
                }
            )

        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        np.testing.assert_allclose(
            res["res.showNumber"].values, ref["res.showNumber"].values
        )
