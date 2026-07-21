import dataclasses
import importlib
import json
import shutil
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from corrai.base.distribution import Distribution
from corrai.base.model import Model
from corrai.base.parameter import Parameter
from corrai.sampling import Sample

_CLASS_REGISTRY = {
    "SobolSanalysis": "corrai.sensitivity",
    "MorrisSanalysis": "corrai.sensitivity",
    "FASTSanalysis": "corrai.sensitivity",
    "RBDFASTSanalysis": "corrai.sensitivity",
    "SciOptimizer": "corrai.optimize",
    "LHSSampler": "corrai.sampling",
    "SobolSampler": "corrai.sampling",
    "MorrisSampler": "corrai.sampling",
    "FASTSampler": "corrai.sampling",
    "RBDFASTSampler": "corrai.sampling",
}


# ─── JSON helpers ─────────────────────────────────────────────────────────────


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _import_class(class_name: str):
    if class_name not in _CLASS_REGISTRY:
        raise ValueError(
            f"Unknown study class: {class_name!r}. "
            f"Supported: {list(_CLASS_REGISTRY)}"
        )
    module = importlib.import_module(_CLASS_REGISTRY[class_name])
    return getattr(module, class_name)


# ─── Parameter serialization ──────────────────────────────────────────────────


def serialize_parameter(param: Parameter) -> dict:
    return dataclasses.asdict(param)


def deserialize_parameter(d: dict) -> Parameter:
    if d.get("interval") is not None:
        d["interval"] = tuple(d["interval"])
    if d.get("values") is not None:
        d["values"] = tuple(d["values"])
    if isinstance(d.get("model_property"), list):
        d["model_property"] = tuple(d["model_property"])
    mmv = d.get("min_max_interval")
    if mmv is not None:
        if isinstance(mmv, list) and len(mmv) > 0 and isinstance(mmv[0], list):
            d["min_max_interval"] = [tuple(t) for t in mmv]
        elif isinstance(mmv, list):
            d["min_max_interval"] = tuple(mmv)
    if isinstance(d.get("distribution"), dict):
        d["distribution"] = Distribution(**d["distribution"])
    return Parameter(**d)


def save_parameters(parameters: list[Parameter], path: str | Path) -> None:
    data = [serialize_parameter(p) for p in parameters]
    Path(path).write_text(json.dumps(data, indent=2))


def load_parameters(path: str | Path) -> list[Parameter]:
    data = json.loads(Path(path).read_text())
    return [deserialize_parameter(d) for d in data]


# ─── Simulation options serialization ────────────────────────────────────────

_BUNDLE_FILE_KEY = "__bundle_file__"


def _pack_simulation_options(opts: dict, bundle_path: Path) -> dict:
    """Copy referenced files into the bundle and replace their values with markers."""
    if not opts:
        return opts

    files_dir = bundle_path / "simulation_files"
    packed = {}
    name_counts: dict[str, int] = {}

    for key, value in opts.items():
        p = None
        if isinstance(value, Path):
            p = value
        elif isinstance(value, str):
            candidate = Path(value)
            if candidate.is_file():
                p = candidate

        if p is not None and p.is_file():
            stem = p.stem
            suffix = p.suffix
            count = name_counts.get(p.name, 0)
            dest_name = p.name if count == 0 else f"{stem}_{count}{suffix}"
            name_counts[p.name] = count + 1

            files_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, files_dir / dest_name)
            packed[key] = {_BUNDLE_FILE_KEY: f"simulation_files/{dest_name}"}
        else:
            packed[key] = value

    return packed


def _unpack_simulation_options(opts_data: dict, bundle_path: Path) -> dict:
    """Restore file paths from bundle markers to absolute Path objects."""
    unpacked = {}
    for key, value in opts_data.items():
        if isinstance(value, dict) and _BUNDLE_FILE_KEY in value:
            unpacked[key] = bundle_path / value[_BUNDLE_FILE_KEY]
        else:
            unpacked[key] = value
    return unpacked


# ─── Sample serialization ─────────────────────────────────────────────────────


def _save_sample(sample: Sample, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sample.values.to_parquet(path / "values.parquet")

    results_dir = path / "results"
    results_dir.mkdir(exist_ok=True)

    pending_indices = []
    saved_indices = []
    for i in sample.results.index:
        result = sample.results[i]
        is_empty = isinstance(result, pd.DataFrame) and result.empty
        if is_empty:
            pending_indices.append(int(i))
        elif isinstance(result, pd.DataFrame):
            result.to_parquet(results_dir / f"{i}.parquet")
            saved_indices.append(int(i))
        elif isinstance(result, pd.Series):
            result.to_frame(name="value").to_parquet(results_dir / f"{i}.parquet")
            saved_indices.append(int(i))

    metadata = {
        "is_dynamic": sample.is_dynamic,
        "n_samples": len(sample),
        "pending_indices": pending_indices,
        "saved_indices": saved_indices,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _load_sample(sample_dir: Path, parameters: list[Parameter]) -> Sample:
    metadata = json.loads((sample_dir / "metadata.json").read_text())
    is_dynamic = metadata["is_dynamic"]
    pending_indices = set(metadata["pending_indices"])
    n_samples = metadata["n_samples"]

    values = pd.read_parquet(sample_dir / "values.parquet")

    results_dir = sample_dir / "results"
    results_list = []
    for i in range(n_samples):
        if i in pending_indices:
            results_list.append(pd.DataFrame())
        else:
            parquet_path = results_dir / f"{i}.parquet"
            if is_dynamic:
                results_list.append(pd.read_parquet(parquet_path))
            else:
                df = pd.read_parquet(parquet_path)
                results_list.append(df.iloc[:, 0])

    sample = Sample(parameters, is_dynamic=is_dynamic)
    sample.values = values.reset_index(drop=True)
    sample.results = pd.Series(results_list, dtype=object)
    return sample


# ─── Model serialization ──────────────────────────────────────────────────────


def _save_model(model: Model, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_type = type(model).__name__
    meta = {
        "model_type": model_type,
        "model_module": type(model).__module__,
        "serializable": False,
    }

    if model_type == "ModelicaFmuModel":
        fmu_dest = model_dir / "model.fmu"
        shutil.copyfile(model.fmu_path, fmu_dest)
        meta.update(
            {
                "serializable": True,
                "fmu_path": "model.fmu",
                "output_list": model.output_list,
                "boundary_table_name": model.boundary_table_name,
            }
        )
    elif model_type == "StaticScikitModel":
        pkl_path = model_dir / "model.pkl"
        joblib.dump(model.scikit_model, pkl_path)
        meta.update(
            {
                "serializable": True,
                "model_pkl": "model.pkl",
                "target_name": model.target_name,
            }
        )
    else:
        try:
            model.save(model_dir / "model")
            meta["serializable"] = True
            meta["saved_path"] = "model"
        except NotImplementedError:
            pass

    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))


def _load_model(model_dir: Path, user_model: Model | None = None) -> Model | None:
    if user_model is not None:
        return user_model

    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    if not meta.get("serializable", False):
        return None

    model_type = meta["model_type"]

    if model_type == "ModelicaFmuModel":
        from corrai.fmu import ModelicaFmuModel

        return ModelicaFmuModel(
            fmu_path=model_dir / meta["fmu_path"],
            output_list=meta.get("output_list"),
            boundary_table_name=meta.get("boundary_table_name"),
        )

    if model_type == "StaticScikitModel":
        from corrai.surrogate import StaticScikitModel

        scikit_model = joblib.load(model_dir / meta["model_pkl"])
        return StaticScikitModel(scikit_model, target_name=meta.get("target_name"))

    if meta.get("saved_path") and meta.get("model_module"):
        import importlib

        mod = importlib.import_module(meta["model_module"])
        cls = getattr(mod, model_type)
        if hasattr(cls, "load"):
            return cls.load(model_dir / meta["saved_path"])

    return None


# ─── Base class ───────────────────────────────────────────────────────────────


class BaseStudyStore(ABC):
    """
    Abstract base for study bundles that persist parameters, model,
    simulation options, and results to a directory.
    """

    _study_class_name: str
    _parameters: list[Parameter]
    _model: Model | None
    _simulation_options: dict | None
    _sample: Sample | None

    def save(self, path: str | Path) -> None:
        """Save the study bundle to *path* (directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        params_data = [serialize_parameter(p) for p in self._parameters]
        (path / "parameters.json").write_text(json.dumps(params_data, indent=2))

        if self._simulation_options:
            packed_opts = _pack_simulation_options(self._simulation_options, path)
            (path / "simulation_options.json").write_text(
                json.dumps(packed_opts, indent=2, default=str)
            )

        (path / "method.json").write_text(
            json.dumps({"study_class": self._study_class_name}, indent=2)
        )

        _save_model(self._model, path / "model")

        has_results = (
            self._sample is not None
            and not self._sample.results.empty
            and not all(
                isinstance(r, pd.DataFrame) and r.empty for r in self._sample.results
            )
        )
        has_values = self._sample is not None and not self._sample.values.empty
        if has_values or has_results:
            _save_sample(self._sample, path / "sample")

        self._save_extra(path, has_results)

        manifest = self._build_manifest(has_results, path)
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default)
        )

    @classmethod
    def load(cls, path: str | Path, model: Model | None = None) -> "BaseStudyStore":
        """Load a study bundle from *path*.

        Parameters
        ----------
        path : str or Path
            Directory containing the saved bundle.
        model : Model, optional
            Provide the model instance if the bundle could not serialise it
            (e.g. custom PyModel subclasses).
        """
        path = Path(path)
        params = [
            deserialize_parameter(d)
            for d in json.loads((path / "parameters.json").read_text())
        ]
        sim_opts = None
        if (path / "simulation_options.json").exists():
            raw_opts = json.loads((path / "simulation_options.json").read_text())
            sim_opts = _unpack_simulation_options(raw_opts, path)

        loaded_model = _load_model(path / "model", model)

        sample = None
        if (path / "sample").exists():
            sample = _load_sample(path / "sample", params)

        method = json.loads((path / "method.json").read_text())["study_class"]

        return cls._from_parts(
            study_class_name=method,
            parameters=params,
            model=loaded_model,
            simulation_options=sim_opts,
            sample=sample,
            bundle_path=path,
        )

    @classmethod
    @abstractmethod
    def _from_parts(
        cls,
        *,
        study_class_name: str,
        parameters: list[Parameter],
        model: Model | None,
        simulation_options: dict | None,
        sample: Sample | None,
        bundle_path: Path,
    ) -> "BaseStudyStore":
        """Reconstruct store from loaded pieces (called by load())."""

    @abstractmethod
    def to_study(self):
        """Return the study object (Sanalysis / SciOptimizer / Sampler) ready to use."""

    def _save_extra(self, path: Path, has_results: bool) -> None:
        """Override to save additional files (e.g. SA indices)."""

    @abstractmethod
    def _build_manifest(self, has_results: bool, path: Path) -> dict:
        """Return the manifest dict."""

    def _get_indicators(self) -> list[str] | None:
        if self._sample is None:
            return None
        results = self._sample.results.dropna()
        if results.empty:
            return None
        first = results.iloc[0]
        if isinstance(first, pd.DataFrame):
            return list(first.columns) if not first.empty else None
        if isinstance(first, pd.Series):
            return list(first.index) if not first.empty else None
        return None

    def _base_manifest(self, study_type: str, has_results: bool) -> dict:
        sim_opts = self._simulation_options or {}
        manifest = {
            "study_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "study_type": study_type,
            "method": self._study_class_name,
            "n_samples": len(self._sample) if self._sample else 0,
            "n_parameters": len(self._parameters),
            "parameter_names": [p.name for p in self._parameters],
            "has_results": has_results,
            "simulation_start": sim_opts.get("startTime", sim_opts.get("start")),
            "simulation_stop": sim_opts.get(
                "stopTime", sim_opts.get("stop", sim_opts.get("end"))
            ),
        }
        indicators = self._get_indicators()
        if indicators is not None:
            manifest["indicators"] = indicators
        return manifest

    def _require_model(self):
        if self._model is None:
            raise ValueError(
                "Model could not be restored from the bundle. "
                "Pass the model explicitly: load(path, model=your_model)."
            )


# ─── Sensitivity Analysis Store ───────────────────────────────────────────────


class SensitivityAnalysisStore(BaseStudyStore):
    """
    Bundle store for sensitivity analysis studies (Sobol, Morris, FAST, RBD-FAST).

    Examples
    --------
    Save after a study::

        sa = SobolSanalysis(params, model, opts)
        sa.add_sample(N=256, simulate=True)
        store = SensitivityAnalysisStore(sa)
        store.save("my_study/")

    Save a configuration before simulating::

        store = SensitivityAnalysisStore.from_config("SobolSanalysis", params, model, opts)
        store.save("my_study/")

    Load and resume::

        store = SensitivityAnalysisStore.load("my_study/")
        sa = store.to_study()
        sa.add_sample(N=256, simulate=True)
        SensitivityAnalysisStore(sa).save("my_study/")
    """

    def __init__(self, sanalysis):
        self._study_class_name = type(sanalysis).__name__
        self._parameters = sanalysis.sampler.parameters
        self._model = sanalysis.sampler.model
        self._simulation_options = sanalysis.sampler.simulation_options or {}
        self._sample = sanalysis.sample
        self._sanalysis = sanalysis
        self._study_options = {}
        if hasattr(sanalysis, "_calc_second_order"):
            self._study_options["calc_second_order"] = sanalysis._calc_second_order

    @classmethod
    def from_config(
        cls,
        method: str,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict | None = None,
        study_options: dict | None = None,
    ) -> "SensitivityAnalysisStore":
        """Create a store from a configuration (no results yet)."""
        study_cls = _import_class(method)
        sanalysis = study_cls(
            parameters, model, simulation_options, **(study_options or {})
        )
        return cls(sanalysis)

    @classmethod
    def _from_parts(
        cls,
        *,
        study_class_name,
        parameters,
        model,
        simulation_options,
        sample,
        bundle_path,
    ):
        method_data = json.loads((bundle_path / "method.json").read_text())
        study_options = method_data.get("study_options", {})
        study_cls = _import_class(study_class_name)
        dummy_model = model or _DummyModel()
        sanalysis = study_cls(
            parameters, dummy_model, simulation_options, **study_options
        )
        if sample is not None:
            sanalysis.sampler.sample = sample
        store = object.__new__(cls)
        store._study_class_name = study_class_name
        store._parameters = parameters
        store._model = model
        store._simulation_options = simulation_options
        store._sample = sample
        store._sanalysis = sanalysis
        store._study_options = study_options
        return store

    def to_study(self):
        """Return a ready-to-use Sanalysis object."""
        self._require_model()
        study_cls = _import_class(self._study_class_name)
        sanalysis = study_cls(
            self._parameters,
            self._model,
            self._simulation_options,
            **self._study_options,
        )
        if self._sample is not None and not self._sample.values.empty:
            sanalysis.sampler.sample = self._sample
        return sanalysis

    def _save_extra(self, path: Path, has_results: bool) -> None:
        method_data = {
            "study_class": self._study_class_name,
            "study_options": self._study_options,
        }
        (path / "method.json").write_text(json.dumps(method_data, indent=2))

        if not has_results:
            return
        indicators = self._get_indicators()
        if not indicators:
            return
        sensitivity_indices = {}
        for indicator in indicators:
            try:
                res = self._sanalysis.analyze(indicator)
                sensitivity_indices[indicator] = {
                    key: (v.tolist() if hasattr(v, "tolist") else v)
                    for r in res
                    for key, v in r.items()
                }
            except Exception:
                pass

        if sensitivity_indices:
            (path / "analysis.json").write_text(
                json.dumps(sensitivity_indices, indent=2, default=_json_default)
            )

    def _build_manifest(self, has_results: bool, path: Path) -> dict:
        manifest = self._base_manifest("sensitivity_analysis", has_results)
        if (path / "analysis.json").exists():
            manifest["sensitivity_indices"] = json.loads(
                (path / "analysis.json").read_text()
            )
        return manifest


# ─── Optimization Store ───────────────────────────────────────────────────────


class OptimizationStore(BaseStudyStore):
    """
    Bundle store for optimization studies (SciOptimizer).

    Examples
    --------
    ::

        optimizer = SciOptimizer(params, model)
        result = optimizer.minimize(indicator_config, simulation_options=opts)
        store = OptimizationStore(
            optimizer, optimize_result=result, algorithm_params={"method": "L-BFGS-B"}
        )
        store.save("optim_study/")

        store = OptimizationStore.load("optim_study/")
        optimizer = store.to_study()
    """

    def __init__(self, optimizer, optimize_result=None, algorithm_params=None):
        if hasattr(optimizer, "model_evaluator"):
            self._study_class_name = "SciOptimizer"
            self._parameters = optimizer.model_evaluator.parameters
            self._model = optimizer.model_evaluator.model
            self._simulation_options = None
            self._sample = optimizer.model_evaluator.sample
        else:
            raise TypeError(
                f"Unsupported optimizer type: {type(optimizer).__name__}. "
                "Currently only SciOptimizer is supported."
            )
        self._optimize_result = optimize_result
        self._algorithm_params = algorithm_params or {}

    @classmethod
    def from_config(
        cls,
        parameters: list[Parameter],
        model: Model,
        algorithm_params: dict | None = None,
    ) -> "OptimizationStore":
        """Create a store from a configuration (no results yet)."""
        from corrai.optimize import SciOptimizer

        optimizer = SciOptimizer(parameters, model)
        return cls(optimizer, algorithm_params=algorithm_params)

    @classmethod
    def _from_parts(
        cls,
        *,
        study_class_name,
        parameters,
        model,
        simulation_options,
        sample,
        bundle_path,
    ):
        store = object.__new__(cls)
        store._study_class_name = study_class_name
        store._parameters = parameters
        store._model = model
        store._simulation_options = simulation_options
        store._sample = sample

        opt_result_path = bundle_path / "optimize_result.json"
        store._optimize_result = (
            json.loads(opt_result_path.read_text())
            if opt_result_path.exists()
            else None
        )
        method_data = json.loads((bundle_path / "method.json").read_text())
        store._algorithm_params = method_data.get("algorithm_params", {})
        return store

    def to_study(self):
        """Return a ready-to-use SciOptimizer."""
        self._require_model()
        from corrai.optimize import SciOptimizer

        optimizer = SciOptimizer(self._parameters, self._model)
        if self._sample is not None and not self._sample.values.empty:
            optimizer.model_evaluator.sample = self._sample
        return optimizer

    def _save_extra(self, path: Path, has_results: bool) -> None:
        if self._optimize_result is not None:
            result_dict = {}
            for k, v in dict(self._optimize_result).items():
                if hasattr(v, "tolist"):
                    result_dict[k] = v.tolist()
                elif isinstance(v, (np.integer,)):
                    result_dict[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    result_dict[k] = float(v)
                else:
                    result_dict[k] = v
            (path / "optimize_result.json").write_text(
                json.dumps(result_dict, indent=2, default=_json_default)
            )

        method_data = {
            "study_class": self._study_class_name,
            "algorithm_params": self._algorithm_params,
        }
        (path / "method.json").write_text(json.dumps(method_data, indent=2))

    def _build_manifest(self, has_results: bool, path: Path) -> dict:
        manifest = self._base_manifest("optimization", has_results)
        manifest["algorithm_params"] = self._algorithm_params
        if self._optimize_result is not None:
            result_dict = {}
            for k, v in dict(self._optimize_result).items():
                if hasattr(v, "tolist"):
                    result_dict[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    result_dict[k] = v.item()
                else:
                    result_dict[k] = v
            manifest["optimize_result"] = result_dict
        return manifest


# ─── Sampling Store ───────────────────────────────────────────────────────────


class SamplingStore(BaseStudyStore):
    """
    Bundle store for sampling / uncertainty propagation studies.

    Examples
    --------
    ::

        sampler = LHSSampler(params, model, opts)
        sampler.add_sample(n=200, simulate=True)
        store = SamplingStore(sampler)
        store.save("sampling_study/")

        store = SamplingStore.load("sampling_study/")
        sampler = store.to_study()
        sampler.add_sample(n=50, simulate=True)
        SamplingStore(sampler).save("sampling_study/")
    """

    def __init__(self, sampler):
        self._study_class_name = type(sampler).__name__
        self._parameters = sampler.parameters
        self._model = sampler.model
        self._simulation_options = sampler.simulation_options or {}
        self._sample = sampler.sample

    @classmethod
    def from_config(
        cls,
        method: str,
        parameters: list[Parameter],
        model: Model,
        simulation_options: dict | None = None,
    ) -> "SamplingStore":
        """Create a store from a configuration (no results yet)."""
        sampler_cls = _import_class(method)
        sampler = sampler_cls(parameters, model, simulation_options)
        return cls(sampler)

    @classmethod
    def _from_parts(
        cls,
        *,
        study_class_name,
        parameters,
        model,
        simulation_options,
        sample,
        bundle_path,
    ):
        store = object.__new__(cls)
        store._study_class_name = study_class_name
        store._parameters = parameters
        store._model = model
        store._simulation_options = simulation_options
        store._sample = sample
        return store

    def to_study(self):
        """Return a ready-to-use Sampler with the loaded sample."""
        self._require_model()
        sampler_cls = _import_class(self._study_class_name)
        sampler = sampler_cls(self._parameters, self._model, self._simulation_options)
        if self._sample is not None and not self._sample.values.empty:
            sampler.sample = self._sample
        return sampler

    def _build_manifest(self, has_results: bool, path: Path) -> dict:
        manifest = self._base_manifest("sampling", has_results)
        if has_results and self._sample is not None and not self._sample.is_dynamic:
            try:
                df = self._sample.get_static_results_as_df()
                stats = {}
                for col in df.columns:
                    s = df[col].dropna()
                    stats[col] = {
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "p5": float(s.quantile(0.05)),
                        "p50": float(s.quantile(0.50)),
                        "p95": float(s.quantile(0.95)),
                    }
                manifest["output_statistics"] = stats
            except Exception:
                pass
        return manifest


# ─── Internal dummy model ─────────────────────────────────────────────────────


class _DummyModel(Model):
    """Placeholder used when reconstructing a study class before model is injected."""

    def __init__(self):
        super().__init__(is_dynamic=True)

    def simulate(self, property_dict=None, simulation_options=None, **kwargs):
        raise RuntimeError(
            "Call to_study() with a real model loaded before simulating."
        )
