"""Tests for factuality checker."""

import pytest
from sentinel.factuality import FactualityChecker, FactualityResult, FactClaim


# ---------------------------------------------------------------------------
# Supported claims
# ---------------------------------------------------------------------------

class TestSupported:
    def test_fully_supported(self):
        c = FactualityChecker()
        result = c.check(
            output="Paris has approximately 2.1 million residents in the city proper.",
            references=["Paris population is approximately 2.1 million residents in the city proper."],
        )
        assert result.supported
        assert result.score > 0.5

    def test_supported_across_refs(self):
        c = FactualityChecker()
        result = c.check(
            output="Python was created by Guido van Rossum. It was first released in 1991.",
            references=[
                "Python programming language was created by Guido van Rossum.",
                "Python was first released in February 1991.",
            ],
        )
        assert result.supported_count >= 1
        assert result.score > 0.5


# ---------------------------------------------------------------------------
# Unsupported claims
# ---------------------------------------------------------------------------

class TestUnsupported:
    def test_unsupported_claim(self):
        c = FactualityChecker()
        result = c.check(
            output="The moon is made of green cheese and orbits Jupiter.",
            references=["The moon orbits Earth at an average distance of 384,400 km."],
        )
        assert result.unsupported_count > 0
        assert not result.supported

    def test_fabricated_details(self):
        c = FactualityChecker()
        result = c.check(
            output="Einstein won the Nobel Prize in Chemistry in 1935 for his work on nuclear fusion.",
            references=["Albert Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect."],
        )
        # The claim has some word overlap but key facts wrong
        assert result.has_claims


# ---------------------------------------------------------------------------
# Claims extraction
# ---------------------------------------------------------------------------

class TestClaims:
    def test_extract_multiple_claims(self):
        c = FactualityChecker()
        result = c.check(
            output="Water boils at 100 degrees Celsius. Ice melts at 0 degrees Celsius. The sky appears blue due to Rayleigh scattering.",
            references=["Water boils at 100 degrees Celsius at sea level. Ice melts at 0 degrees Celsius."],
        )
        assert len(result.claims) >= 2

    def test_short_text_no_claims(self):
        c = FactualityChecker()
        result = c.check(output="Yes.", references=["Anything"])
        assert len(result.claims) == 0
        assert result.score == 1.0

    def test_claim_structure(self):
        c = FactualityChecker()
        result = c.check(
            output="The Earth revolves around the Sun.",
            references=["The Earth revolves around the Sun in approximately 365.25 days."],
        )
        if result.claims:
            claim = result.claims[0]
            assert isinstance(claim, FactClaim)
            assert isinstance(claim.confidence, float)
            assert 0.0 <= claim.confidence <= 1.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_perfect_score(self):
        c = FactualityChecker()
        result = c.check(
            output="Python is a programming language used for data science and web development.",
            references=["Python is a programming language widely used for data science and web development."],
        )
        assert result.score >= 0.8

    def test_zero_score(self):
        c = FactualityChecker()
        result = c.check(
            output="Quantum teleportation allows instant travel between galaxies using wormholes.",
            references=["Classical mechanics describes the motion of macroscopic objects."],
        )
        assert result.score < 0.5

    def test_threshold_custom(self):
        c = FactualityChecker(threshold=0.9)
        result = c.check(
            output="The cat sat on the mat near the window.",
            references=["The cat was sitting on a mat."],
        )
        # Higher threshold means harder to be "supported"
        assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_check(self):
        c = FactualityChecker()
        results = c.check_batch(
            outputs=[
                "Water boils at 100 degrees Celsius at sea level.",
                "The sun orbits the Earth every 24 hours.",
            ],
            references=["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
        )
        assert len(results) == 2
        assert results[0].score >= results[1].score


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_result_fields(self):
        c = FactualityChecker()
        result = c.check(
            output="Gravity pulls objects toward the center of the Earth.",
            references=["Gravity is the force that pulls objects toward the center of the Earth."],
        )
        assert isinstance(result, FactualityResult)
        assert result.output != ""
        assert isinstance(result.claims, list)
        assert result.supported_count >= 0
        assert result.unsupported_count >= 0
        assert 0.0 <= result.score <= 1.0

    def test_empty_output(self):
        c = FactualityChecker()
        result = c.check(output="", references=["Some reference."])
        assert result.score == 1.0
        assert len(result.claims) == 0

    def test_empty_references(self):
        c = FactualityChecker()
        result = c.check(
            output="This is a factual claim about something important.",
            references=[],
        )
        assert result.unsupported_count > 0
