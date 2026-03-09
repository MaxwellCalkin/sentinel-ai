"""A/B testing for guardrail configurations.

Compare different safety configurations side-by-side to optimize
the balance between safety and user experience.

Usage:
    from sentinel.ab_tester import ABTester

    tester = ABTester()
    tester.add_variant("strict", check_fn=strict_check)
    tester.add_variant("relaxed", check_fn=relaxed_check)
    result = tester.run(texts)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class VariantResult:
    """Result from a single variant's check."""
    variant: str
    blocked: bool
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    """A single test case with results from all variants."""
    text: str
    results: dict[str, VariantResult] = field(default_factory=dict)


@dataclass
class ABReport:
    """Report comparing variant performance."""
    variants: list[str]
    total_cases: int
    results: dict[str, VariantStats]
    agreement_rate: float  # How often variants agree
    cases: list[TestCase] = field(default_factory=list)


@dataclass
class VariantStats:
    """Aggregated stats for one variant."""
    variant: str
    block_count: int = 0
    allow_count: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    block_rate: float = 0.0


class ABTester:
    """A/B test different guardrail configurations.

    Register variants with different check functions, then run them
    all against the same inputs to compare behavior.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._variants: dict[str, Callable[[str], bool]] = {}
        self._rng = random.Random(seed)

    def add_variant(
        self,
        name: str,
        check_fn: Callable[[str], bool],
    ) -> None:
        """Add a variant. check_fn(text) -> True if blocked."""
        self._variants[name] = check_fn

    def remove_variant(self, name: str) -> None:
        """Remove a variant."""
        self._variants.pop(name, None)

    @property
    def variant_count(self) -> int:
        return len(self._variants)

    def run(
        self,
        texts: list[str],
        include_cases: bool = False,
    ) -> ABReport:
        """Run all variants against all texts.

        Args:
            texts: Input texts to test.
            include_cases: Whether to include per-case details in report.

        Returns:
            ABReport comparing variant performance.
        """
        cases: list[TestCase] = []
        latencies: dict[str, list[float]] = {v: [] for v in self._variants}
        block_counts: dict[str, int] = {v: 0 for v in self._variants}

        for text in texts:
            tc = TestCase(text=text)
            for name, check_fn in self._variants.items():
                start = time.perf_counter()
                blocked = check_fn(text)
                elapsed = (time.perf_counter() - start) * 1000

                tc.results[name] = VariantResult(
                    variant=name,
                    blocked=blocked,
                    latency_ms=elapsed,
                )
                latencies[name].append(elapsed)
                if blocked:
                    block_counts[name] += 1

            cases.append(tc)

        # Compute stats
        variant_stats: dict[str, VariantStats] = {}
        n = len(texts) or 1
        for name in self._variants:
            lats = sorted(latencies[name]) if latencies[name] else [0.0]
            p99_idx = max(0, int(len(lats) * 0.99) - 1)
            variant_stats[name] = VariantStats(
                variant=name,
                block_count=block_counts[name],
                allow_count=len(texts) - block_counts[name],
                avg_latency_ms=sum(lats) / len(lats) if lats else 0,
                p99_latency_ms=lats[p99_idx],
                block_rate=block_counts[name] / n,
            )

        # Agreement rate
        agreements = 0
        for tc in cases:
            decisions = set(r.blocked for r in tc.results.values())
            if len(decisions) <= 1:
                agreements += 1
        agreement_rate = agreements / n if texts else 1.0

        return ABReport(
            variants=list(self._variants.keys()),
            total_cases=len(texts),
            results=variant_stats,
            agreement_rate=agreement_rate,
            cases=cases if include_cases else [],
        )

    def pick_variant(self, weights: dict[str, float] | None = None) -> str:
        """Randomly pick a variant (for online A/B testing).

        Args:
            weights: Optional weight per variant. Equal if None.

        Returns:
            Name of the selected variant.
        """
        names = list(self._variants.keys())
        if not names:
            raise ValueError("No variants registered")
        if weights:
            w = [weights.get(n, 1.0) for n in names]
        else:
            w = [1.0] * len(names)
        return self._rng.choices(names, weights=w, k=1)[0]

    def check_with_variant(self, text: str, variant: str) -> VariantResult:
        """Run a specific variant's check.

        Args:
            text: Input to check.
            variant: Variant name.

        Returns:
            VariantResult with blocked status and latency.
        """
        if variant not in self._variants:
            raise KeyError(f"Unknown variant: {variant}")
        fn = self._variants[variant]
        start = time.perf_counter()
        blocked = fn(text)
        elapsed = (time.perf_counter() - start) * 1000
        return VariantResult(variant=variant, blocked=blocked, latency_ms=elapsed)
