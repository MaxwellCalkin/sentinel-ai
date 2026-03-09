"""Embedding vector validation and anomaly detection.

Validates embedding vectors for dimensionality, magnitude, NaN/Inf values,
zero ratios, and uniformity. Catches corrupted or abnormal vectors before
they enter downstream systems.

Usage:
    from sentinel.embedding_validator import EmbeddingValidator, EmbeddingProfile

    profile = EmbeddingProfile(expected_dim=384)
    validator = EmbeddingValidator(profile)
    report = validator.validate([0.1, 0.2, ...])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EmbeddingCheck:
    """Result of a single validation check on an embedding vector."""

    name: str
    passed: bool
    message: str
    value: float | None = None


@dataclass
class EmbeddingReport:
    """Full validation report for an embedding vector."""

    vector_length: int
    is_valid: bool
    checks: list[EmbeddingCheck]
    overall_score: float


@dataclass
class EmbeddingProfile:
    """Configuration profile defining expected embedding properties."""

    expected_dim: int
    min_magnitude: float = 0.01
    max_magnitude: float = 100.0
    max_zero_ratio: float = 0.5
    max_nan_ratio: float = 0.0


@dataclass
class ValidatorStats:
    """Cumulative validation statistics."""

    total_validated: int = 0
    passed: int = 0
    failed: int = 0
    avg_magnitude: float = 0.0


class EmbeddingValidator:
    """Validates embedding vectors for anomalies and quality issues.

    Runs a suite of checks including dimensionality, NaN/Inf detection,
    zero vector detection, magnitude bounds, zero ratio, and uniformity.
    """

    def __init__(self, profile: EmbeddingProfile | None = None) -> None:
        self._profile = profile
        self._stats = ValidatorStats()
        self._magnitude_sum = 0.0

    def validate(self, vector: list[float]) -> EmbeddingReport:
        """Run all validation checks on a single embedding vector."""
        checks = [
            self._check_dimensionality(vector),
            self._check_nan_inf(vector),
            self._check_zero_vector(vector),
            self._check_magnitude(vector),
            self._check_zero_ratio(vector),
            self._check_uniformity(vector),
        ]

        passed_count = sum(1 for c in checks if c.passed)
        overall_score = passed_count / len(checks) if checks else 0.0
        is_valid = all(c.passed for c in checks)

        magnitude = self._compute_magnitude(vector)
        self._update_stats(is_valid, magnitude)

        return EmbeddingReport(
            vector_length=len(vector),
            is_valid=is_valid,
            checks=checks,
            overall_score=overall_score,
        )

    def validate_batch(self, vectors: list[list[float]]) -> list[EmbeddingReport]:
        """Validate a batch of embedding vectors."""
        return [self.validate(v) for v in vectors]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero magnitude.
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vector dimensions must match: {len(a)} != {len(b)}"
            )

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def stats(self) -> ValidatorStats:
        """Return cumulative validation statistics."""
        return ValidatorStats(
            total_validated=self._stats.total_validated,
            passed=self._stats.passed,
            failed=self._stats.failed,
            avg_magnitude=self._stats.avg_magnitude,
        )

    def _check_dimensionality(self, vector: list[float]) -> EmbeddingCheck:
        if self._profile is None:
            return EmbeddingCheck(
                name="dimensionality",
                passed=True,
                message="No profile set; skipping dimensionality check",
                value=float(len(vector)),
            )

        expected = self._profile.expected_dim
        actual = len(vector)
        passed = actual == expected

        message = (
            f"Dimension matches expected {expected}"
            if passed
            else f"Expected {expected} dimensions, got {actual}"
        )

        return EmbeddingCheck(
            name="dimensionality",
            passed=passed,
            message=message,
            value=float(actual),
        )

    def _check_nan_inf(self, vector: list[float]) -> EmbeddingCheck:
        nan_count = sum(1 for v in vector if math.isnan(v))
        inf_count = sum(1 for v in vector if math.isinf(v))
        bad_count = nan_count + inf_count

        if bad_count == 0:
            return EmbeddingCheck(
                name="nan_inf",
                passed=True,
                message="No NaN or Inf values detected",
                value=0.0,
            )

        parts = []
        if nan_count > 0:
            parts.append(f"{nan_count} NaN")
        if inf_count > 0:
            parts.append(f"{inf_count} Inf")

        return EmbeddingCheck(
            name="nan_inf",
            passed=False,
            message=f"Found {', '.join(parts)} values",
            value=float(bad_count),
        )

    def _check_zero_vector(self, vector: list[float]) -> EmbeddingCheck:
        is_zero = all(v == 0.0 for v in vector)

        if is_zero:
            return EmbeddingCheck(
                name="zero_vector",
                passed=False,
                message="Vector is all zeros",
                value=0.0,
            )

        return EmbeddingCheck(
            name="zero_vector",
            passed=True,
            message="Vector is not all zeros",
            value=1.0,
        )

    def _check_magnitude(self, vector: list[float]) -> EmbeddingCheck:
        safe_values = [v for v in vector if math.isfinite(v)]
        magnitude = math.sqrt(sum(v * v for v in safe_values))

        if self._profile is None:
            return EmbeddingCheck(
                name="magnitude",
                passed=True,
                message=f"Magnitude is {magnitude:.4f} (no bounds set)",
                value=magnitude,
            )

        min_mag = self._profile.min_magnitude
        max_mag = self._profile.max_magnitude
        passed = min_mag <= magnitude <= max_mag

        if passed:
            message = f"Magnitude {magnitude:.4f} within [{min_mag}, {max_mag}]"
        elif magnitude < min_mag:
            message = f"Magnitude {magnitude:.4f} below minimum {min_mag}"
        else:
            message = f"Magnitude {magnitude:.4f} above maximum {max_mag}"

        return EmbeddingCheck(
            name="magnitude",
            passed=passed,
            message=message,
            value=magnitude,
        )

    def _check_zero_ratio(self, vector: list[float]) -> EmbeddingCheck:
        if len(vector) == 0:
            return EmbeddingCheck(
                name="zero_ratio",
                passed=True,
                message="Empty vector; skipping zero ratio check",
                value=0.0,
            )

        zero_count = sum(1 for v in vector if v == 0.0)
        ratio = zero_count / len(vector)

        if self._profile is None:
            threshold = 0.5
        else:
            threshold = self._profile.max_zero_ratio

        passed = ratio <= threshold

        message = (
            f"Zero ratio {ratio:.2f} within threshold {threshold}"
            if passed
            else f"Zero ratio {ratio:.2f} exceeds threshold {threshold}"
        )

        return EmbeddingCheck(
            name="zero_ratio",
            passed=passed,
            message=message,
            value=ratio,
        )

    def _check_uniformity(self, vector: list[float]) -> EmbeddingCheck:
        if len(vector) == 0:
            return EmbeddingCheck(
                name="uniformity",
                passed=True,
                message="Empty vector; skipping uniformity check",
            )

        finite_values = [v for v in vector if math.isfinite(v)]
        if not finite_values:
            return EmbeddingCheck(
                name="uniformity",
                passed=True,
                message="No finite values to check uniformity",
            )

        first = finite_values[0]
        all_identical = all(v == first for v in finite_values)

        if all_identical and first != 0.0 and len(finite_values) > 1:
            return EmbeddingCheck(
                name="uniformity",
                passed=False,
                message=f"All {len(finite_values)} values are identical ({first})",
                value=0.0,
            )

        return EmbeddingCheck(
            name="uniformity",
            passed=True,
            message="Values are not uniformly identical",
            value=1.0,
        )

    def _compute_magnitude(self, vector: list[float]) -> float:
        safe_values = [v for v in vector if math.isfinite(v)]
        return math.sqrt(sum(v * v for v in safe_values))

    def _update_stats(self, is_valid: bool, magnitude: float) -> None:
        total = self._stats.total_validated
        self._magnitude_sum += magnitude
        self._stats.total_validated += 1

        if is_valid:
            self._stats.passed += 1
        else:
            self._stats.failed += 1

        self._stats.avg_magnitude = self._magnitude_sum / self._stats.total_validated
