"""Tests for claim extraction."""

import pytest
from sentinel.claim_extractor import (
    ClaimExtractor,
    ExtractedClaim,
    ExtractionResult,
    ExtractorStats,
)


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    def test_extract_single_fact(self):
        e = ClaimExtractor()
        result = e.extract("Paris is the capital of France.")
        assert result.total_claims == 1
        assert result.claims[0].claim_type == "verifiable_fact"

    def test_extract_multiple_claims(self):
        e = ClaimExtractor()
        result = e.extract(
            "Water boils at 100 degrees Celsius. The Earth orbits the Sun. "
            "Python was created by Guido van Rossum."
        )
        assert result.total_claims == 3

    def test_empty_and_short_text_yields_no_claims(self):
        e = ClaimExtractor()
        assert e.extract("").total_claims == 0
        assert e.extract("").claims == []
        assert e.extract("Yes.").total_claims == 0


# ---------------------------------------------------------------------------
# Claim classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_opinion_detected(self):
        e = ClaimExtractor()
        result = e.extract("I think chocolate ice cream is the best flavor.")
        assert result.total_claims == 1
        assert result.claims[0].claim_type == "opinion"

    def test_subjective_detected(self):
        e = ClaimExtractor()
        result = e.extract("The sunset was absolutely beautiful and wonderful.")
        assert result.total_claims == 1
        assert result.claims[0].claim_type == "subjective"

    def test_statistical_detected(self):
        e = ClaimExtractor()
        result = e.extract("Approximately 71% of the Earth's surface is covered by water.")
        assert result.total_claims == 1
        assert result.claims[0].claim_type == "statistical"

    def test_temporal_detected(self):
        e = ClaimExtractor()
        result = e.extract("The internet was invented in 1983.")
        assert result.total_claims == 1
        assert result.claims[0].claim_type == "temporal"

    def test_verifiable_fact_default(self):
        e = ClaimExtractor()
        result = e.extract("Hydrogen is the lightest chemical element.")
        assert result.total_claims == 1
        assert result.claims[0].claim_type == "verifiable_fact"


# ---------------------------------------------------------------------------
# Questions and instructions skipped
# ---------------------------------------------------------------------------

class TestNonClaims:
    def test_questions_and_instructions_skipped(self):
        e = ClaimExtractor()
        assert e.extract("What is the capital of France?").total_claims == 0
        assert e.extract("Please open the file and review the contents.").total_claims == 0


# ---------------------------------------------------------------------------
# Subject and predicate extraction
# ---------------------------------------------------------------------------

class TestSubjectPredicate:
    def test_subject_and_predicate_extracted(self):
        e = ClaimExtractor()
        result = e.extract("The Earth orbits the Sun.")
        claim = result.claims[0]
        assert claim.subject != ""
        assert "Earth" in claim.subject
        assert claim.predicate != ""
        assert "orbits" in claim.predicate

    def test_predicate_contains_verb(self):
        e = ClaimExtractor()
        result = e.extract("Water boils at 100 degrees Celsius.")
        claim = result.claims[0]
        assert "boils" in claim.predicate


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_in_range(self):
        e = ClaimExtractor()
        result = e.extract("The speed of light is 299,792,458 meters per second.")
        for claim in result.claims:
            assert 0.0 <= claim.confidence <= 1.0

    def test_opinion_higher_confidence_than_fact(self):
        e = ClaimExtractor()
        opinion_result = e.extract("I believe this approach is correct.")
        fact_result = e.extract("Oxygen is required for combustion.")
        assert opinion_result.claims[0].confidence >= fact_result.claims[0].confidence


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_filter_by_type(self):
        e = ClaimExtractor()
        result = e.extract(
            "I think Python is great. Python was released in 1991. "
            "Roughly 8 million developers use Python."
        )
        opinions = e.filter_claims(result, claim_type="opinion")
        assert all(c.claim_type == "opinion" for c in opinions)
        assert len(opinions) >= 1

    def test_filter_by_confidence_and_combined(self):
        e = ClaimExtractor()
        result = e.extract(
            "I believe the sky is blue. The ocean covers most of the Earth. "
            "About 67 million people live in France."
        )
        high_conf = e.filter_claims(result, min_confidence=0.80)
        for claim in high_conf:
            assert claim.confidence >= 0.80
        stats = e.filter_claims(result, claim_type="statistical", min_confidence=0.5)
        for claim in stats:
            assert claim.claim_type == "statistical"
            assert claim.confidence >= 0.5


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_extraction(self):
        e = ClaimExtractor()
        results = e.extract_batch([
            "The Moon orbits the Earth.",
            "I think cats are better than dogs.",
        ])
        assert len(results) == 2
        assert results[0].claims[0].claim_type == "verifiable_fact"
        assert results[1].claims[0].claim_type == "opinion"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_accumulate(self):
        e = ClaimExtractor()
        e.extract("The Earth is round.")
        e.extract("Water freezes at 0 degrees. Python was created in 1991.")
        stats = e.stats
        assert stats.total_texts == 2
        assert stats.total_claims == 3
        assert stats.avg_claims_per_text == 1.5

    def test_stats_by_type(self):
        e = ClaimExtractor()
        e.extract("I believe this is true.")
        e.extract("Roughly 70% of Earth is ocean.")
        stats = e.stats
        assert "opinion" in stats.by_type
        assert "statistical" in stats.by_type

    def test_stats_reset(self):
        e = ClaimExtractor()
        e.extract("The sky is blue.")
        e.reset_stats()
        stats = e.stats
        assert stats.total_texts == 0
        assert stats.total_claims == 0
        assert stats.by_type == {}


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_dataclass_fields(self):
        e = ClaimExtractor()
        result = e.extract("Gravity pulls objects toward the Earth.")
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.text, str)
        assert isinstance(result.claims, list)
        assert isinstance(result.total_claims, int)
        assert isinstance(result.by_type, dict)

    def test_claim_dataclass_fields(self):
        e = ClaimExtractor()
        result = e.extract("Iron is a chemical element.")
        claim = result.claims[0]
        assert isinstance(claim, ExtractedClaim)
        assert isinstance(claim.text, str)
        assert isinstance(claim.claim_type, str)
        assert isinstance(claim.confidence, float)
        assert isinstance(claim.subject, str)
        assert isinstance(claim.predicate, str)
        assert isinstance(claim.source_sentence, str)

    def test_by_type_breakdown(self):
        e = ClaimExtractor()
        result = e.extract(
            "I think this is useful. The Sun is a star. "
            "Approximately 150 million km separates Earth from the Sun."
        )
        assert result.by_type.get("opinion", 0) >= 1
        assert result.by_type.get("verifiable_fact", 0) + result.by_type.get("statistical", 0) >= 1
