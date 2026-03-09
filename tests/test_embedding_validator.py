"""Tests for embedding vector validator."""

import math

import pytest
from sentinel.embedding_validator import (
    EmbeddingCheck,
    EmbeddingProfile,
    EmbeddingReport,
    EmbeddingValidator,
    ValidatorStats,
)


# ---------------------------------------------------------------------------
# Valid vectors
# ---------------------------------------------------------------------------


class TestValidVectors:
    def test_normalized_vector_passes_all_checks(self):
        profile = EmbeddingProfile(expected_dim=3)
        validator = EmbeddingValidator(profile)
        # L2 norm = 1.0 with varied values
        mag = math.sqrt(0.5**2 + 0.7**2 + 0.5099**2)
        vec = [0.5 / mag, 0.7 / mag, 0.5099 / mag]
        report = validator.validate(vec)

        assert report.is_valid is True
        assert report.overall_score == 1.0
        assert report.vector_length == 3
        assert all(c.passed for c in report.checks)

    def test_valid_vector_without_profile(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.5, -0.3, 0.8])

        assert report.is_valid is True
        assert report.overall_score == 1.0

    def test_single_element_valid_vector(self):
        profile = EmbeddingProfile(expected_dim=1)
        validator = EmbeddingValidator(profile)
        report = validator.validate([0.5])

        assert report.is_valid is True


# ---------------------------------------------------------------------------
# Zero vector detection
# ---------------------------------------------------------------------------


class TestZeroVector:
    def test_all_zeros_fails(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.0, 0.0, 0.0])

        assert report.is_valid is False
        zero_check = _find_check(report, "zero_vector")
        assert zero_check.passed is False
        assert "all zeros" in zero_check.message

    def test_mostly_zeros_but_not_all_passes_zero_vector_check(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.0, 0.0, 0.1])

        zero_check = _find_check(report, "zero_vector")
        assert zero_check.passed is True


# ---------------------------------------------------------------------------
# NaN and Inf detection
# ---------------------------------------------------------------------------


class TestNanInf:
    def test_nan_value_fails(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.1, float("nan"), 0.3])

        assert report.is_valid is False
        nan_check = _find_check(report, "nan_inf")
        assert nan_check.passed is False
        assert "NaN" in nan_check.message

    def test_inf_value_fails(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.1, float("inf"), 0.3])

        assert report.is_valid is False
        nan_check = _find_check(report, "nan_inf")
        assert nan_check.passed is False
        assert "Inf" in nan_check.message

    def test_negative_inf_fails(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.1, float("-inf"), 0.3])

        nan_check = _find_check(report, "nan_inf")
        assert nan_check.passed is False

    def test_no_nan_or_inf_passes(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.1, 0.2, 0.3])

        nan_check = _find_check(report, "nan_inf")
        assert nan_check.passed is True


# ---------------------------------------------------------------------------
# Dimensionality check
# ---------------------------------------------------------------------------


class TestDimensionality:
    def test_wrong_dimensionality_fails(self):
        profile = EmbeddingProfile(expected_dim=3)
        validator = EmbeddingValidator(profile)
        report = validator.validate([0.1, 0.2])

        assert report.is_valid is False
        dim_check = _find_check(report, "dimensionality")
        assert dim_check.passed is False
        assert "Expected 3" in dim_check.message
        assert "got 2" in dim_check.message

    def test_correct_dimensionality_passes(self):
        profile = EmbeddingProfile(expected_dim=4)
        validator = EmbeddingValidator(profile)
        report = validator.validate([0.1, 0.2, 0.3, 0.4])

        dim_check = _find_check(report, "dimensionality")
        assert dim_check.passed is True
        assert dim_check.value == 4.0

    def test_no_profile_skips_dimensionality_check(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.1, 0.2])

        dim_check = _find_check(report, "dimensionality")
        assert dim_check.passed is True
        assert "skipping" in dim_check.message.lower()


# ---------------------------------------------------------------------------
# Magnitude check
# ---------------------------------------------------------------------------


class TestMagnitude:
    def test_too_low_magnitude_fails(self):
        profile = EmbeddingProfile(expected_dim=3, min_magnitude=0.1)
        validator = EmbeddingValidator(profile)
        report = validator.validate([0.001, 0.001, 0.001])

        mag_check = _find_check(report, "magnitude")
        assert mag_check.passed is False
        assert "below" in mag_check.message

    def test_too_high_magnitude_fails(self):
        profile = EmbeddingProfile(expected_dim=2, max_magnitude=2.0)
        validator = EmbeddingValidator(profile)
        report = validator.validate([100.0, 100.0])

        mag_check = _find_check(report, "magnitude")
        assert mag_check.passed is False
        assert "above" in mag_check.message

    def test_magnitude_within_range_passes(self):
        profile = EmbeddingProfile(
            expected_dim=3, min_magnitude=0.5, max_magnitude=5.0
        )
        validator = EmbeddingValidator(profile)
        report = validator.validate([1.0, 0.0, 0.0])

        mag_check = _find_check(report, "magnitude")
        assert mag_check.passed is True
        assert mag_check.value == pytest.approx(1.0)

    def test_magnitude_no_profile_always_passes(self):
        validator = EmbeddingValidator()
        report = validator.validate([100.0, 200.0])

        mag_check = _find_check(report, "magnitude")
        assert mag_check.passed is True


# ---------------------------------------------------------------------------
# Zero ratio check
# ---------------------------------------------------------------------------


class TestZeroRatio:
    def test_high_zero_ratio_fails(self):
        profile = EmbeddingProfile(expected_dim=10, max_zero_ratio=0.3)
        validator = EmbeddingValidator(profile)
        # 8 out of 10 are zero = 0.8 ratio
        vector = [0.0] * 8 + [1.0, 1.0]
        report = validator.validate(vector)

        ratio_check = _find_check(report, "zero_ratio")
        assert ratio_check.passed is False
        assert ratio_check.value == pytest.approx(0.8)

    def test_low_zero_ratio_passes(self):
        profile = EmbeddingProfile(expected_dim=4, max_zero_ratio=0.5)
        validator = EmbeddingValidator(profile)
        report = validator.validate([0.0, 1.0, 2.0, 3.0])

        ratio_check = _find_check(report, "zero_ratio")
        assert ratio_check.passed is True
        assert ratio_check.value == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Uniformity check
# ---------------------------------------------------------------------------


class TestUniformity:
    def test_uniform_nonzero_vector_flagged(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.5, 0.5, 0.5, 0.5])

        uni_check = _find_check(report, "uniformity")
        assert uni_check.passed is False
        assert "identical" in uni_check.message

    def test_varied_vector_passes_uniformity(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.1, 0.2, 0.3])

        uni_check = _find_check(report, "uniformity")
        assert uni_check.passed is True

    def test_single_element_vector_not_flagged(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.5])

        uni_check = _find_check(report, "uniformity")
        assert uni_check.passed is True


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_similarity_one(self):
        validator = EmbeddingValidator()
        vec = [0.5, 0.3, 0.1]
        similarity = validator.cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        validator = EmbeddingValidator()
        similarity = validator.cosine_similarity([1.0, 0.0], [0.0, 1.0])

        assert similarity == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors_similarity_negative_one(self):
        validator = EmbeddingValidator()
        similarity = validator.cosine_similarity([1.0, 0.0], [-1.0, 0.0])

        assert similarity == pytest.approx(-1.0)

    def test_mismatched_dimensions_raises_error(self):
        validator = EmbeddingValidator()

        with pytest.raises(ValueError, match="dimensions must match"):
            validator.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_cosine_returns_zero(self):
        validator = EmbeddingValidator()
        similarity = validator.cosine_similarity([0.0, 0.0], [1.0, 2.0])

        assert similarity == 0.0


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


class TestBatchValidation:
    def test_batch_returns_reports_for_each_vector(self):
        validator = EmbeddingValidator()
        vectors = [
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [1.0, float("nan"), 0.5],
        ]
        reports = validator.validate_batch(vectors)

        assert len(reports) == 3
        assert reports[0].is_valid is True
        assert reports[1].is_valid is False
        assert reports[2].is_valid is False

    def test_empty_batch_returns_empty_list(self):
        validator = EmbeddingValidator()
        reports = validator.validate_batch([])

        assert reports == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


class TestStats:
    def test_initial_stats_are_zero(self):
        validator = EmbeddingValidator()
        s = validator.stats()

        assert s.total_validated == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.avg_magnitude == 0.0

    def test_stats_track_passed_and_failed(self):
        validator = EmbeddingValidator()
        validator.validate([0.5, 0.3, 0.1])  # passes
        validator.validate([0.0, 0.0, 0.0])  # fails (zero vector)

        s = validator.stats()
        assert s.total_validated == 2
        assert s.passed == 1
        assert s.failed == 1

    def test_avg_magnitude_updates(self):
        validator = EmbeddingValidator()
        validator.validate([3.0, 4.0])  # magnitude = 5.0
        validator.validate([0.0, 0.0, 1.0])  # magnitude = 1.0 (fails, but still counted)

        s = validator.stats()
        assert s.avg_magnitude == pytest.approx(3.0)

    def test_stats_returns_snapshot(self):
        validator = EmbeddingValidator()
        validator.validate([1.0, 0.0])
        s1 = validator.stats()
        validator.validate([0.0, 1.0])
        s2 = validator.stats()

        assert s1.total_validated == 1
        assert s2.total_validated == 2


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


class TestScoreCalculation:
    def test_all_checks_pass_score_is_one(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.5, -0.3, 0.8])

        assert report.overall_score == 1.0

    def test_partial_failures_reduce_score(self):
        validator = EmbeddingValidator()
        # Uniform vector: only uniformity check fails (6 checks, 5 pass)
        report = validator.validate([0.5, 0.5, 0.5])

        assert report.overall_score == pytest.approx(5.0 / 6.0)

    def test_zero_vector_has_low_score(self):
        validator = EmbeddingValidator()
        report = validator.validate([0.0, 0.0, 0.0])

        assert report.overall_score < 1.0
        assert report.is_valid is False


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_embedding_check_fields(self):
        check = EmbeddingCheck(name="test", passed=True, message="ok")
        assert check.value is None

    def test_embedding_profile_defaults(self):
        profile = EmbeddingProfile(expected_dim=768)
        assert profile.min_magnitude == 0.01
        assert profile.max_magnitude == 100.0
        assert profile.max_zero_ratio == 0.5
        assert profile.max_nan_ratio == 0.0

    def test_validator_stats_defaults(self):
        s = ValidatorStats()
        assert s.total_validated == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.avg_magnitude == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_check(report: EmbeddingReport, name: str) -> EmbeddingCheck:
    """Find a check by name in a report."""
    for check in report.checks:
        if check.name == name:
            return check
    raise ValueError(f"Check '{name}' not found in report")
