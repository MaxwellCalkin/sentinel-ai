"""Differential privacy noise injection for LLM outputs.

Add calibrated noise to sensitive numeric outputs and redact
low-count categories to protect individual privacy in
aggregated results.

Usage:
    from sentinel.diff_privacy import DiffPrivacy

    dp = DiffPrivacy(epsilon=1.0)
    noisy = dp.add_noise(exact_count, sensitivity=1.0)
"""

from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NoiseResult:
    """Result of adding differential privacy noise."""
    original: float
    noised: float
    noise_added: float
    epsilon: float
    mechanism: str


@dataclass
class PrivateHistogram:
    """Histogram with differential privacy applied."""
    bins: dict[str, float]  # category -> noisy count
    suppressed: list[str]   # categories below threshold
    epsilon: float
    total_noise: float


@dataclass
class PrivacyBudget:
    """Track cumulative privacy budget usage."""
    total_epsilon: float
    used_epsilon: float
    remaining_epsilon: float
    queries: int


class DiffPrivacy:
    """Differential privacy mechanisms for protecting sensitive outputs.

    Supports Laplace mechanism for numeric queries and
    histogram sanitization with suppression.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        budget: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            epsilon: Privacy parameter (smaller = more private).
            budget: Total privacy budget. None = unlimited.
            seed: Random seed for reproducibility.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = epsilon
        self._budget = budget
        self._used = 0.0
        self._queries = 0
        self._rng = random.Random(seed)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def budget_remaining(self) -> float | None:
        if self._budget is None:
            return None
        return max(0.0, self._budget - self._used)

    def _check_budget(self, cost: float) -> None:
        if self._budget is not None:
            if self._used + cost > self._budget:
                raise BudgetExhausted(
                    f"Privacy budget exhausted: used {self._used:.4f} + "
                    f"cost {cost:.4f} > budget {self._budget:.4f}"
                )

    def _laplace_noise(self, scale: float) -> float:
        """Generate Laplace noise with given scale."""
        u = self._rng.random() - 0.5
        return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))

    def add_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
        epsilon: float | None = None,
    ) -> NoiseResult:
        """Add Laplace noise to a numeric value.

        Args:
            value: The exact value to protect.
            sensitivity: Query sensitivity (max change from one record).
            epsilon: Per-query epsilon. Defaults to instance epsilon.

        Returns:
            NoiseResult with original and noised values.
        """
        eps = epsilon or self._epsilon
        self._check_budget(eps)

        scale = sensitivity / eps
        noise = self._laplace_noise(scale)
        noised = value + noise

        self._used += eps
        self._queries += 1

        return NoiseResult(
            original=value,
            noised=round(noised, 6),
            noise_added=round(noise, 6),
            epsilon=eps,
            mechanism="laplace",
        )

    def add_noise_batch(
        self,
        values: list[float],
        sensitivity: float = 1.0,
        epsilon: float | None = None,
    ) -> list[NoiseResult]:
        """Add noise to multiple values."""
        return [self.add_noise(v, sensitivity, epsilon) for v in values]

    def private_count(
        self,
        count: int,
        epsilon: float | None = None,
    ) -> NoiseResult:
        """Add noise to a count query (sensitivity=1)."""
        result = self.add_noise(float(count), sensitivity=1.0, epsilon=epsilon)
        result.noised = max(0, round(result.noised))
        return result

    def private_sum(
        self,
        total: float,
        max_contribution: float,
        epsilon: float | None = None,
    ) -> NoiseResult:
        """Add noise to a sum query.

        Args:
            total: The exact sum.
            max_contribution: Maximum any individual can contribute.
            epsilon: Per-query epsilon.
        """
        return self.add_noise(total, sensitivity=max_contribution, epsilon=epsilon)

    def private_mean(
        self,
        total: float,
        count: int,
        max_value: float,
        epsilon: float | None = None,
    ) -> NoiseResult:
        """Add noise to a mean query.

        Args:
            total: Sum of values.
            count: Number of records.
            max_value: Maximum individual value.
            epsilon: Per-query epsilon.
        """
        if count == 0:
            return NoiseResult(
                original=0.0, noised=0.0, noise_added=0.0,
                epsilon=epsilon or self._epsilon, mechanism="laplace",
            )
        mean = total / count
        sensitivity = max_value / count
        return self.add_noise(mean, sensitivity=sensitivity, epsilon=epsilon)

    def private_histogram(
        self,
        counts: dict[str, int],
        suppression_threshold: int = 5,
        epsilon: float | None = None,
    ) -> PrivateHistogram:
        """Create a differentially private histogram.

        Args:
            counts: Category -> count mapping.
            suppression_threshold: Suppress categories below this count.
            epsilon: Per-query epsilon.
        """
        eps = epsilon or self._epsilon
        # Each bin uses eps/len(counts) of the budget (parallel composition)
        per_bin_eps = eps
        self._check_budget(eps)

        noisy_bins: dict[str, float] = {}
        suppressed: list[str] = []
        total_noise = 0.0

        for cat, count in counts.items():
            if count < suppression_threshold:
                suppressed.append(cat)
                continue
            scale = 1.0 / per_bin_eps
            noise = self._laplace_noise(scale)
            noisy_count = max(0, round(count + noise))
            noisy_bins[cat] = noisy_count
            total_noise += abs(noise)

        self._used += eps
        self._queries += 1

        return PrivateHistogram(
            bins=noisy_bins,
            suppressed=suppressed,
            epsilon=eps,
            total_noise=round(total_noise, 4),
        )

    def budget_status(self) -> PrivacyBudget:
        """Get current privacy budget status."""
        return PrivacyBudget(
            total_epsilon=self._budget or float("inf"),
            used_epsilon=round(self._used, 6),
            remaining_epsilon=(
                round(self._budget - self._used, 6)
                if self._budget is not None
                else float("inf")
            ),
            queries=self._queries,
        )

    def reset_budget(self) -> None:
        """Reset privacy budget tracking."""
        self._used = 0.0
        self._queries = 0


class BudgetExhausted(Exception):
    """Raised when privacy budget is exhausted."""
