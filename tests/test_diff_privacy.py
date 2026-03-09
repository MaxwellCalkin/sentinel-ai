"""Tests for differential privacy noise injection."""

import pytest
from sentinel.diff_privacy import DiffPrivacy, NoiseResult, PrivateHistogram, BudgetExhausted


# ---------------------------------------------------------------------------
# Laplace noise
# ---------------------------------------------------------------------------

class TestLaplaceNoise:
    def test_add_noise_changes_value(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.add_noise(100.0)
        assert result.noised != result.original
        assert result.noise_added != 0

    def test_noise_result_structure(self):
        dp = DiffPrivacy(epsilon=0.5, seed=1)
        result = dp.add_noise(50.0, sensitivity=2.0)
        assert isinstance(result, NoiseResult)
        assert result.original == 50.0
        assert result.epsilon == 0.5
        assert result.mechanism == "laplace"

    def test_higher_epsilon_less_noise(self):
        """Higher epsilon = less privacy = less noise on average."""
        noises_high = []
        noises_low = []
        for seed in range(100):
            dp_high = DiffPrivacy(epsilon=10.0, seed=seed)
            dp_low = DiffPrivacy(epsilon=0.1, seed=seed)
            noises_high.append(abs(dp_high.add_noise(100.0).noise_added))
            noises_low.append(abs(dp_low.add_noise(100.0).noise_added))
        assert sum(noises_high) / len(noises_high) < sum(noises_low) / len(noises_low)

    def test_sensitivity_scales_noise(self):
        """Higher sensitivity = more noise."""
        noises_high = []
        noises_low = []
        for seed in range(100):
            dp = DiffPrivacy(epsilon=1.0, seed=seed)
            noises_low.append(abs(dp.add_noise(100.0, sensitivity=1.0).noise_added))
            dp2 = DiffPrivacy(epsilon=1.0, seed=seed + 1000)
            noises_high.append(abs(dp2.add_noise(100.0, sensitivity=10.0).noise_added))
        assert sum(noises_high) / len(noises_high) > sum(noises_low) / len(noises_low)

    def test_deterministic_with_seed(self):
        dp1 = DiffPrivacy(epsilon=1.0, seed=42)
        dp2 = DiffPrivacy(epsilon=1.0, seed=42)
        assert dp1.add_noise(100.0).noised == dp2.add_noise(100.0).noised

    def test_batch_noise(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        results = dp.add_noise_batch([10.0, 20.0, 30.0])
        assert len(results) == 3
        assert all(isinstance(r, NoiseResult) for r in results)


# ---------------------------------------------------------------------------
# Private count
# ---------------------------------------------------------------------------

class TestPrivateCount:
    def test_count_returns_integer(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.private_count(100)
        assert isinstance(result.noised, int)

    def test_count_non_negative(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.private_count(0)
        assert result.noised >= 0

    def test_count_approximately_correct(self):
        """Over many trials, average should be close to true value."""
        true_count = 1000
        noised_values = []
        for seed in range(200):
            dp = DiffPrivacy(epsilon=1.0, seed=seed)
            noised_values.append(dp.private_count(true_count).noised)
        avg = sum(noised_values) / len(noised_values)
        assert abs(avg - true_count) < 5  # Should be very close with 200 samples


# ---------------------------------------------------------------------------
# Private sum
# ---------------------------------------------------------------------------

class TestPrivateSum:
    def test_private_sum(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.private_sum(5000.0, max_contribution=100.0)
        assert result.original == 5000.0
        assert result.noised != 5000.0


# ---------------------------------------------------------------------------
# Private mean
# ---------------------------------------------------------------------------

class TestPrivateMean:
    def test_private_mean(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.private_mean(total=500.0, count=100, max_value=10.0)
        assert result.original == 5.0  # 500/100

    def test_mean_zero_count(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.private_mean(total=0, count=0, max_value=10.0)
        assert result.noised == 0.0


# ---------------------------------------------------------------------------
# Private histogram
# ---------------------------------------------------------------------------

class TestHistogram:
    def test_histogram_basic(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        counts = {"cat_a": 100, "cat_b": 200, "cat_c": 50}
        result = dp.private_histogram(counts)
        assert isinstance(result, PrivateHistogram)
        assert len(result.bins) == 3
        assert result.epsilon == 1.0

    def test_histogram_suppression(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        counts = {"common": 100, "rare": 2}
        result = dp.private_histogram(counts, suppression_threshold=5)
        assert "rare" in result.suppressed
        assert "rare" not in result.bins
        assert "common" in result.bins

    def test_histogram_no_suppression(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        counts = {"a": 10, "b": 20}
        result = dp.private_histogram(counts, suppression_threshold=0)
        assert len(result.suppressed) == 0


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_unlimited_budget(self):
        dp = DiffPrivacy(epsilon=1.0)
        for _ in range(100):
            dp.add_noise(1.0)
        assert dp.budget_remaining is None

    def test_budget_tracking(self):
        dp = DiffPrivacy(epsilon=1.0, budget=5.0, seed=42)
        dp.add_noise(1.0)  # costs 1.0
        dp.add_noise(1.0)  # costs 1.0
        status = dp.budget_status()
        assert status.used_epsilon == pytest.approx(2.0)
        assert status.remaining_epsilon == pytest.approx(3.0)
        assert status.queries == 2

    def test_budget_exhausted(self):
        dp = DiffPrivacy(epsilon=1.0, budget=2.0, seed=42)
        dp.add_noise(1.0)
        dp.add_noise(1.0)
        with pytest.raises(BudgetExhausted):
            dp.add_noise(1.0)

    def test_budget_reset(self):
        dp = DiffPrivacy(epsilon=1.0, budget=2.0, seed=42)
        dp.add_noise(1.0)
        dp.add_noise(1.0)
        dp.reset_budget()
        status = dp.budget_status()
        assert status.used_epsilon == 0.0
        assert status.queries == 0
        # Can query again
        dp.add_noise(1.0)

    def test_custom_epsilon_per_query(self):
        dp = DiffPrivacy(epsilon=1.0, budget=3.0, seed=42)
        dp.add_noise(1.0, epsilon=2.0)
        status = dp.budget_status()
        assert status.used_epsilon == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_epsilon_rejected(self):
        with pytest.raises(ValueError):
            DiffPrivacy(epsilon=0)

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError):
            DiffPrivacy(epsilon=-1.0)

    def test_large_value(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.add_noise(1_000_000.0)
        assert isinstance(result.noised, float)

    def test_negative_value(self):
        dp = DiffPrivacy(epsilon=1.0, seed=42)
        result = dp.add_noise(-50.0)
        assert result.original == -50.0
