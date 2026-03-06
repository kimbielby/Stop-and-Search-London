"""
tests/test_gini.py
==================
Tests for the Gini coefficient calculation in config/gini.py.

The Gini coefficient measures inequality in a distribution of values.
A value of 0 represents perfect equality (all values identical) and
a value of 1 represents maximum inequality (one value holds everything).
"""
import numpy as np
import pytest
from utils import gini

class TestGiniKnownValues:
    """Test Gini against analytically known results."""

    def test_perfect_equality(self):
        """All identical values should produce a Gini of 0."""
        values = [100_000, 100_000, 100_000, 100_000]
        assert gini(values) == pytest.approx(0.0, abs=1e-10)

    def test_two_values_known_result(self):
        """
        For two values [1, 3], the Gini coefficient is 0.25.
        Verified analytically using the sorted-values formula:
        (2 * (1*1 + 2*3) / (2 * 4)) - (3/2) = 1.75 - 1.5 = 0.25
        """
        result = gini([1, 3])
        assert result == pytest.approx(0.25, abs=1e-10)

    def test_three_values_known_result(self):
        """
        For [1, 2, 3], Gini = 2/9 ≈ 0.2222.
        Verified using the standard formula.
        """
        result = gini([1, 2, 3])
        assert result == pytest.approx(2 / 9, abs=1e-10)

    def test_result_between_zero_and_one(self):
        """Gini coefficient must always be in [0, 1)."""
        values = [150_000, 300_000, 500_000, 750_000, 2_000_000]
        result = gini(values)
        assert 0.0 <= result < 1.0

    def test_order_invariant(self):
        """Result should be identical regardless of input order."""
        values = [500_000, 150_000, 2_000_000, 300_000]
        assert gini(values) == pytest.approx(gini(sorted(values)), abs=1e-10)

    def test_scale_invariant(self):
        """
        Multiplying all values by a constant should not change the Gini.
        This is a standard property of the Gini coefficient.
        """
        values = [100_000, 300_000, 600_000]
        assert gini(values) == pytest.approx(gini([v * 10 for v in values]), abs=1e-10)

    def test_realistic_property_prices(self):
        """
        Realistic London property prices should produce a Gini in a
        plausible range. Based on our analytical dataset mean of 0.221.
        """
        prices = [
            280_000, 320_000, 410_000, 450_000, 500_000,
            550_000, 620_000, 750_000, 950_000, 1_800_000
        ]
        result = gini(prices)
        assert 0.1 < result < 0.6


class TestGiniEdgeCases:
    """Test Gini behaviour at edge cases and with invalid inputs."""

    def test_single_value_returns_nan(self):
        """A single value provides no inequality information — return NaN."""
        assert np.isnan(gini([500_000]))

    def test_empty_array_returns_nan(self):
        """An empty array should return NaN."""
        assert np.isnan(gini([]))

    def test_all_zeros_returns_nan(self):
        """
        All-zero input: zeros are dropped inside gini(), leaving an
        empty array, which should return NaN.
        """
        assert np.isnan(gini([0, 0, 0]))

    def test_zeros_are_dropped(self):
        """
        Zeros represent missing or invalid transactions and are dropped.
        gini([0, 1, 3]) should equal gini([1, 3]).
        """
        assert gini([0, 1, 3]) == pytest.approx(gini([1, 3]), abs=1e-10)

    def test_single_nonzero_among_zeros_returns_nan(self):
        """
        After dropping zeros, a single remaining value cannot produce
        a meaningful Gini — should return NaN.
        """
        assert np.isnan(gini([0, 0, 500_000]))

    def test_below_threshold_not_enforced_in_function(self):
        """
        The MIN_TRANSACTIONS threshold (n < 30) is enforced at the
        pipeline level, not inside gini(). The function itself will
        happily compute a Gini for small samples.
        """
        values = [100_000, 200_000, 300_000]
        result = gini(values)
        assert not np.isnan(result)

    def test_numpy_array_input(self):
        """Function should accept numpy arrays as well as lists."""
        values = np.array([100_000, 200_000, 400_000])
        result = gini(values)
        assert not np.isnan(result)

    def test_large_sample_performance(self):
        """
        Function should handle a realistic sample size (n=500)
        without error or significant performance degradation.
        """
        rng = np.random.default_rng(42)
        prices = rng.lognormal(mean=13.0, sigma=0.6, size=500)
        result = gini(prices)
        assert 0.0 <= result < 1.0
