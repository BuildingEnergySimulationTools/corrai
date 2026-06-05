import pandas as pd
import pytest

from corrai.base.model import Ishigami, PyModel
from corrai.base.parameter import Parameter
from corrai.optimize import SciOptimizer
from corrai.sampling import LHSSampler
from corrai.sensitivity import SobolSanalysis
from corrai.store import (
    OptimizationStore,
    SamplingStore,
    SensitivityAnalysisStore,
    deserialize_parameter,
    load_parameters,
    save_parameters,
    serialize_parameter,
    _pack_simulation_options,
    _unpack_simulation_options,
)

PARAMETERS = [
    Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
    Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
    Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
]

MULTI_PROP_PARAM = Parameter(
    "par_multi",
    (0.0, 1.0),
    model_property=("x1", "x2"),
    init_value=0.5,
)


class StaticSquare(PyModel):
    """Simple static model: returns x1^2."""

    def __init__(self):
        super().__init__(is_dynamic=False)
        self.x1 = 1.0
        self.x2 = 1.0
        self.x3 = 1.0

    def simulate(self, property_dict=None, simulation_options=None, **kw):
        if property_dict:
            self.set_property_values(property_dict)
        return pd.Series({"res": self.x1**2 + self.x2**2})


class TestSimulationOptionsPacking:
    def test_file_path_is_copied_and_restored(self, tmp_path):
        scenario = tmp_path / "scenario.csv"
        scenario.write_text("a,b\n1,2\n")

        bundle = tmp_path / "bundle"
        bundle.mkdir()

        opts = {"startTime": 0, "stopTime": 3600, "scenario_file": scenario}
        packed = _pack_simulation_options(opts, bundle)

        assert "__bundle_file__" in packed["scenario_file"]
        assert (bundle / "simulation_files" / "scenario.csv").exists()
        assert packed["startTime"] == 0

        unpacked = _unpack_simulation_options(packed, bundle)
        assert unpacked["scenario_file"] == bundle / "simulation_files" / "scenario.csv"
        assert unpacked["startTime"] == 0

    def test_string_path_is_detected(self, tmp_path):
        ray_file = tmp_path / "weather.ray"
        ray_file.write_text("data")

        bundle = tmp_path / "bundle"
        bundle.mkdir()

        opts = {"ray_file": str(ray_file)}
        packed = _pack_simulation_options(opts, bundle)

        assert "__bundle_file__" in packed["ray_file"]

    def test_name_collision_is_handled(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "data.csv").write_text("1")
        (dir_b / "data.csv").write_text("2")

        bundle = tmp_path / "bundle"
        bundle.mkdir()

        opts = {"file_a": dir_a / "data.csv", "file_b": dir_b / "data.csv"}
        packed = _pack_simulation_options(opts, bundle)

        files = list((bundle / "simulation_files").iterdir())
        assert len(files) == 2

    def test_nonexistent_path_string_not_treated_as_file(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        opts = {"solver": "CVode", "label": "/not/a/real/file.txt"}
        packed = _pack_simulation_options(opts, bundle)

        assert packed["solver"] == "CVode"
        assert packed["label"] == "/not/a/real/file.txt"
        assert not (bundle / "simulation_files").exists()

    def test_simulation_files_survive_full_store_roundtrip(self, tmp_path):
        scenario = tmp_path / "scenario.csv"
        scenario.write_text("col\n1\n2\n")

        params = [Parameter("x1", (0.0, 2.0), model_property="x1")]
        sampler = LHSSampler(
            params,
            StaticSquare(),
            simulation_options={"scenario_file": scenario, "startTime": 0},
        )

        store = SamplingStore(sampler)
        store.save(tmp_path / "bundle")

        loaded = SamplingStore.load(tmp_path / "bundle", model=StaticSquare())
        restored_path = loaded._simulation_options["scenario_file"]
        assert isinstance(restored_path, __import__("pathlib").Path)
        assert restored_path.exists()
        assert restored_path.read_text() == "col\n1\n2\n"


class TestParameterSerialization:
    def test_roundtrip_basic(self):
        p = PARAMETERS[0]
        d = serialize_parameter(p)
        p2 = deserialize_parameter(d)
        assert p2.name == p.name
        assert p2.interval == p.interval
        assert p2.ptype == p.ptype
        assert p2.model_property == p.model_property

    def test_roundtrip_tuple_model_property(self):
        p = MULTI_PROP_PARAM
        d = serialize_parameter(p)
        p2 = deserialize_parameter(d)
        assert p2.model_property == ("x1", "x2")
        assert p2.interval == (0.0, 1.0)

    def test_roundtrip_choice_param(self):
        p = Parameter(
            "algo", values=("A", "B", "C"), ptype="Choice", model_property="m"
        )
        d = serialize_parameter(p)
        p2 = deserialize_parameter(d)
        assert p2.values == ("A", "B", "C")

    def test_save_load_parameters(self, tmp_path):
        path = tmp_path / "params.json"
        save_parameters(PARAMETERS, path)
        loaded = load_parameters(path)
        assert len(loaded) == len(PARAMETERS)
        for orig, restored in zip(PARAMETERS, loaded):
            assert restored.name == orig.name
            assert restored.interval == orig.interval
            assert restored.model_property == orig.model_property


class TestSensitivityAnalysisStore:
    def test_config_only_save_load(self, tmp_path):
        store = SensitivityAnalysisStore.from_config(
            "SobolSanalysis", PARAMETERS, Ishigami()
        )
        store.save(tmp_path / "sa_config")

        loaded = SensitivityAnalysisStore.load(tmp_path / "sa_config", model=Ishigami())
        assert loaded._study_class_name == "SobolSanalysis"
        assert len(loaded._parameters) == 3
        assert loaded._sample is None or loaded._sample.values.empty

    def test_config_to_study_and_simulate(self, tmp_path):
        store = SensitivityAnalysisStore.from_config(
            "SobolSanalysis", PARAMETERS, Ishigami()
        )
        store.save(tmp_path / "sa_config")

        loaded = SensitivityAnalysisStore.load(tmp_path / "sa_config", model=Ishigami())
        sa = loaded.to_study()
        sa.add_sample(N=64, simulate=True)
        assert len(sa.sample) > 0
        assert not sa.sample.results.empty

    def test_full_cycle_with_results(self, tmp_path):
        sa = SobolSanalysis(PARAMETERS, Ishigami(), calc_second_order=False)
        sa.add_sample(N=64, simulate=True)

        store = SensitivityAnalysisStore(sa)
        store.save(tmp_path / "sa_full")

        # manifest should have has_results=True
        import json

        manifest = json.loads((tmp_path / "sa_full" / "manifest.json").read_text())
        assert manifest["has_results"] is True
        assert manifest["n_samples"] == len(sa.sample)
        assert "indicators" in manifest
        assert manifest["indicators"] == ["res"]

        # reload and check sample fidelity
        loaded = SensitivityAnalysisStore.load(tmp_path / "sa_full", model=Ishigami())
        assert loaded._sample is not None
        pd.testing.assert_frame_equal(
            loaded._sample.values.reset_index(drop=True),
            sa.sample.values.reset_index(drop=True),
        )
        assert len(loaded._sample.results) == len(sa.sample.results)

    def test_to_study_without_model_raises(self, tmp_path):
        sa = SobolSanalysis(PARAMETERS, StaticSquare())
        store = SensitivityAnalysisStore(sa)
        store.save(tmp_path / "sa_no_model")

        loaded = SensitivityAnalysisStore.load(tmp_path / "sa_no_model")
        with pytest.raises(ValueError, match="Model could not be restored"):
            loaded.to_study()


class TestOptimizationStore:
    def test_config_only_save_load(self, tmp_path):
        store = OptimizationStore.from_config(PARAMETERS, StaticSquare())
        store.save(tmp_path / "opt_config")

        loaded = OptimizationStore.load(tmp_path / "opt_config", model=StaticSquare())
        assert loaded._study_class_name == "SciOptimizer"
        assert len(loaded._parameters) == 3

    def test_full_cycle_with_results(self, tmp_path):
        optimizer = SciOptimizer(PARAMETERS, Ishigami())
        result = optimizer.diff_evo_minimize("res", rng=42, maxiter=5)

        store = OptimizationStore(
            optimizer,
            optimize_result=result,
            algorithm_params={"method": "differential_evolution"},
        )
        store.save(tmp_path / "opt_full")

        import json

        manifest = json.loads((tmp_path / "opt_full" / "manifest.json").read_text())
        assert manifest["has_results"] is True
        assert "optimize_result" in manifest
        assert "x" in manifest["optimize_result"]

        loaded = OptimizationStore.load(tmp_path / "opt_full", model=Ishigami())
        assert loaded._optimize_result is not None
        assert loaded._algorithm_params == {"method": "differential_evolution"}

        # Resume: reconstruct and verify sample
        opt2 = loaded.to_study()
        assert not opt2.sample.values.empty

    def test_to_study_without_model_raises(self, tmp_path):
        optimizer = SciOptimizer(PARAMETERS, StaticSquare())
        store = OptimizationStore(optimizer)
        store.save(tmp_path / "opt_no_model")

        loaded = OptimizationStore.load(tmp_path / "opt_no_model")
        with pytest.raises(ValueError, match="Model could not be restored"):
            loaded.to_study()


class TestSamplingStore:
    def test_config_only_save_load(self, tmp_path):
        store = SamplingStore.from_config("LHSSampler", PARAMETERS, Ishigami())
        store.save(tmp_path / "samp_config")

        loaded = SamplingStore.load(tmp_path / "samp_config", model=Ishigami())
        assert loaded._study_class_name == "LHSSampler"
        assert len(loaded._parameters) == 3

    def test_full_cycle_static(self, tmp_path):
        params = [
            Parameter("x1", (0.0, 2.0), model_property="x1"),
            Parameter("x2", (0.0, 2.0), model_property="x2"),
        ]
        sampler = LHSSampler(params, StaticSquare())
        sampler.add_sample(n=20, rng=0, simulate=True)

        store = SamplingStore(sampler)
        store.save(tmp_path / "samp_static")

        import json

        manifest = json.loads((tmp_path / "samp_static" / "manifest.json").read_text())
        assert manifest["has_results"] is True
        assert "output_statistics" in manifest
        assert "res" in manifest["output_statistics"]
        assert "mean" in manifest["output_statistics"]["res"]

        loaded = SamplingStore.load(tmp_path / "samp_static", model=StaticSquare())
        sampler2 = loaded.to_study()

        pd.testing.assert_frame_equal(
            sampler2.sample.values.reset_index(drop=True),
            sampler.sample.values.reset_index(drop=True),
        )
        assert len(sampler2.sample.results) == 20

        # Resume: add more samples
        sampler2.add_sample(n=5, rng=1, simulate=True)
        assert len(sampler2.sample) == 25
