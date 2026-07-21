from dataclasses import dataclass, field

import numpy as np
from scipy import stats

DISTRIBUTIONS = ["normal", "truncnormal", "uniform", "halfcauchy", "lognormal"]

REQUIRED_PARAMS = {
    "normal": {"mean", "std"},
    "truncnormal": {"mean", "std", "low", "high"},
    "uniform": {"low", "high"},
    "halfcauchy": {"loc", "scale"},
    "lognormal": {"mean", "sigma"},
}


@dataclass
class Distribution:
    """
    Probability distribution assigned to a `Parameter`, used to draw random
    values for uncertainty propagation (see `corrai.sampling.MonteCarloSampler`).

    Parameters
    ----------
    dist : str
        Distribution family. One of `DISTRIBUTIONS`:
        `"normal"`, `"truncnormal"`, `"uniform"`, `"halfcauchy"`, `"lognormal"`.
    params : dict
        Distribution parameters. Required keys depend on `dist`:

        - `"normal"`: `mean`, `std`
        - `"truncnormal"`: `mean`, `std`, `low`, `high`
        - `"uniform"`: `low`, `high`
        - `"halfcauchy"`: `loc`, `scale`
        - `"lognormal"`: `mean`, `sigma` (parameters of the underlying normal)

    Examples
    --------
    >>> Distribution("normal", {"mean": 0.036, "std": 0.002})
    >>> Distribution(
    ...     "truncnormal", {"mean": 0.036, "std": 0.002, "low": 0.03, "high": 0.04}
    ... )
    >>> Distribution("uniform", {"low": 0.03, "high": 0.04})
    >>> Distribution("halfcauchy", {"loc": 0, "scale": 1})
    """

    dist: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dist not in DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.dist!r}. Must be one of {DISTRIBUTIONS}."
            )

        required = REQUIRED_PARAMS[self.dist]
        missing = required - self.params.keys()
        if missing:
            raise ValueError(
                f"Missing parameters {sorted(missing)} for distribution {self.dist!r}. "
                f"Required: {sorted(required)}."
            )

    def _frozen(self):
        p = self.params
        if self.dist == "normal":
            return stats.norm(loc=p["mean"], scale=p["std"])
        if self.dist == "truncnormal":
            a = (p["low"] - p["mean"]) / p["std"]
            b = (p["high"] - p["mean"]) / p["std"]
            return stats.truncnorm(a, b, loc=p["mean"], scale=p["std"])
        if self.dist == "uniform":
            return stats.uniform(loc=p["low"], scale=p["high"] - p["low"])
        if self.dist == "halfcauchy":
            return stats.halfcauchy(loc=p["loc"], scale=p["scale"])
        if self.dist == "lognormal":
            return stats.lognorm(s=p["sigma"], scale=np.exp(p["mean"]))
        raise ValueError(f"Invalid distribution: {self.dist!r}")

    def rvs(self, size, random_state=None):
        return self._frozen().rvs(size=size, random_state=random_state)

    def quantile_range(self, low: float = 0.005, high: float = 0.995):
        """
        Return a display-friendly (low, high) range, used as a fallback
        interval for plotting when a `Parameter` has no explicit `interval`.
        """
        lo, hi = self._frozen().ppf([low, high])
        return float(lo), float(hi)
