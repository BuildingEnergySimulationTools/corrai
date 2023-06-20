import numpy as np
import pandas as pd
from pathlib import Path

import pytest

from sklearn.metrics import mean_squared_error, mean_absolute_error
from modelitool.simulate import Simulator
from modelitool.surrogate import SimulationSampler
from modelitool.surrogate import SurrogateModel

from corrai.connector import ModelicaFunction


PACKAGE_DIR = Path(__file__).parent / "TestLib"

outputs = ["res1.showNumber", "res2.showNumber"]

parameters = [
    {"name": "x.k", "interval": (1.0, 3.0)},
    {"name": "y.k", "interval": (1.0, 3.0)},
]

agg_methods_dict = {
    "res1.showNumber": mean_squared_error,
    "res2.showNumber": mean_absolute_error,
}

reference_dict = {"res1.showNumber": "meas1", "res2.showNumber": "meas2"}

simu_options = {
    "startTime": 0,
    "stopTime": 1,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl",
    "outputFormat": "csv",
}

simu = Simulator(
    model_path="TestLib.ishigami_two_outputs",
    package_path=PACKAGE_DIR / "package.mo",
    simulation_options=simu_options,
    output_list=outputs,
    simulation_path=None,
    lmodel=["Modelica"],
)

x_dict = {"x.k": 2, "y.k": 2}

dataset = pd.DataFrame(
    {
        "meas1": [6, 2],
        "meas2": [14, 1],
    },
    index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
)

expected_res = pd.DataFrame(
    {
        "meas1": [8.15, 8.15],
        "meas2": [12.31, 12.31],
    },
    index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
)


class TestModelicaFunction:
    def test_function_indicators(self):
        mf = ModelicaFunction(
            simulator=simu,
            param_list=parameters,
            agg_methods_dict=agg_methods_dict,
            indicators=["res1.showNumber", "res2.showNumber"],
            reference_df=dataset,
            reference_dict=reference_dict,
        )

        res = mf.function(x_dict)

        np.testing.assert_allclose(
            np.array([res["res1.showNumber"], res["res2.showNumber"]]),
            np.array(
                [
                    mean_squared_error(expected_res["meas1"], dataset["meas1"]),
                    mean_absolute_error(expected_res["meas2"], dataset["meas2"]),
                ]
            ),
            rtol=0.01,
        )

    def test_function_no_indicators(self):
        mf = ModelicaFunction(
            simulator=simu,
            param_list=parameters,
            agg_methods_dict=None,
            indicators=None,
            reference_df=None,
            reference_dict=None,
        )

        res = mf.function(x_dict)

        np.testing.assert_allclose(
            np.array([res["res1.showNumber"], res["res2.showNumber"]]),
            np.array([np.mean(expected_res["meas1"]), np.mean(expected_res["meas2"])]),
            rtol=0.01,
        )

    def test_warning_error(self):
        # reference_df is not provided
        with pytest.raises(ValueError):
            ModelicaFunction(
                simulator=simu,
                param_list=parameters,
                reference_df=None,
                reference_dict=dataset,
            )

        # reference_dict is not provided
        with pytest.raises(ValueError):
            ModelicaFunction(
                simulator=simu,
                param_list=parameters,
                reference_df=dataset,
                reference_dict=None,
            )


class TestScikitFunction:
    def test_function(self):
        surrogate = SurrogateModel(
            simulation_sampler=SimulationSampler(
                simulator=simu,
                parameters=parameters,
            )
        )

        surrogate.add_samples(100, seed=42)
        surrogate.fit_sample(
            indicator="res1.showNumber",
            aggregation_method=np.mean,
        )

        res = surrogate.predict(np.array([2, 2]))
        np.testing.assert_almost_equal(res, 8.15, decimal=1)
