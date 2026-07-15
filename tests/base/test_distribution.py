import numpy as np
import pytest

from corrai.base.distribution import Distribution


class TestDistribution:
    def test_invalid_dist_name(self):
        with pytest.raises(ValueError, match="Invalid distribution"):
            Distribution("not_a_dist", {})

    def test_missing_params(self):
        with pytest.raises(ValueError, match="Missing parameters"):
            Distribution("normal", {"mean": 0})

    def test_normal_rvs(self):
        dist = Distribution("normal", {"mean": 10, "std": 2})
        samples = dist.rvs(size=5000, random_state=0)
        assert samples.shape == (5000,)
        assert np.isclose(samples.mean(), 10, atol=0.2)
        assert np.isclose(samples.std(), 2, atol=0.2)

    def test_truncnormal_rvs_within_bounds(self):
        dist = Distribution(
            "truncnormal", {"mean": 0.036, "std": 0.01, "low": 0.03, "high": 0.04}
        )
        samples = dist.rvs(size=2000, random_state=0)
        assert samples.min() >= 0.03
        assert samples.max() <= 0.04

    def test_uniform_rvs_within_bounds(self):
        dist = Distribution("uniform", {"low": 2, "high": 5})
        samples = dist.rvs(size=2000, random_state=0)
        assert samples.min() >= 2
        assert samples.max() <= 5

    def test_halfcauchy_rvs_non_negative(self):
        dist = Distribution("halfcauchy", {"loc": 0, "scale": 1})
        samples = dist.rvs(size=2000, random_state=0)
        assert samples.min() >= 0

    def test_lognormal_rvs_positive(self):
        dist = Distribution("lognormal", {"mean": 0, "sigma": 0.5})
        samples = dist.rvs(size=2000, random_state=0)
        assert samples.min() > 0

    def test_quantile_range(self):
        dist = Distribution("uniform", {"low": 0, "high": 10})
        low, high = dist.quantile_range()
        assert low == pytest.approx(0.05, abs=1e-6)
        assert high == pytest.approx(9.95, abs=1e-6)

    def test_rvs_reproducible_with_seed(self):
        dist = Distribution("normal", {"mean": 0, "std": 1})
        s1 = dist.rvs(size=100, random_state=np.random.default_rng(42))
        s2 = dist.rvs(size=100, random_state=np.random.default_rng(42))
        np.testing.assert_allclose(s1, s2)
