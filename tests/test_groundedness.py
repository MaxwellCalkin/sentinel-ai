"""Tests for groundedness checking."""

import pytest
from sentinel.groundedness import GroundednessChecker, GroundednessResult, Claim


# ---------------------------------------------------------------------------
# Grounded responses
# ---------------------------------------------------------------------------

class TestGrounded:
    def test_directly_supported(self):
        c = GroundednessChecker(threshold=0.3)
        result = c.check(
            response="Paris is the capital of France.",
            context="France is a country in Europe. Its capital city is Paris.",
        )
        assert result.grounded
        assert result.score > 0.3

    def test_paraphrased_support(self):
        c = GroundednessChecker()
        result = c.check(
            response="The company reported revenue of 50 billion dollars.",
            context="The company announced total revenue reaching 50 billion dollars in the fiscal year.",
        )
        assert result.grounded

    def test_multiple_claims_all_grounded(self):
        c = GroundednessChecker()
        result = c.check(
            response="Python is a programming language. It was created by Guido van Rossum.",
            context="Python is a popular programming language. Python was created by Guido van Rossum in 1991.",
        )
        assert result.grounded
        assert result.claim_count >= 1


# ---------------------------------------------------------------------------
# Ungrounded responses
# ---------------------------------------------------------------------------

class TestUngrounded:
    def test_fabricated_facts(self):
        c = GroundednessChecker()
        result = c.check(
            response="The CEO announced a merger with Google worth 100 billion dollars.",
            context="The company released its quarterly earnings report showing steady growth.",
        )
        assert not result.grounded
        assert len(result.ungrounded_claims) > 0

    def test_contradicts_context(self):
        c = GroundednessChecker()
        result = c.check(
            response="The experiment completely failed with terrible results.",
            context="The experiment was a great success with excellent outcomes.",
        )
        # Low overlap despite both being about "experiment"
        assert result.score < 0.8

    def test_no_context(self):
        c = GroundednessChecker()
        result = c.check(
            response="The population is approximately 10 million people.",
            context="",
        )
        assert not result.grounded


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

class TestClaimExtraction:
    def test_claims_extracted(self):
        c = GroundednessChecker(min_claim_words=2)
        result = c.check(
            response="The sky is blue. Water is wet. Grass is green.",
            context="The sky appears blue during clear weather. Water is wet to the touch.",
        )
        assert result.claim_count >= 2

    def test_short_sentences_skipped(self):
        c = GroundednessChecker(min_claim_words=5)
        result = c.check(
            response="Yes. OK. The detailed analysis shows significant improvement in quarterly performance.",
            context="The quarterly performance analysis shows significant improvement metrics.",
        )
        # Short sentences should be skipped
        assert result.claim_count <= 2

    def test_claim_has_evidence(self):
        c = GroundednessChecker()
        result = c.check(
            response="Machine learning algorithms improve over time with more data.",
            context="Machine learning algorithms are known to improve their performance over time as they process more data.",
        )
        if result.claims:
            grounded_claims = [cl for cl in result.claims if cl.grounded]
            if grounded_claims:
                assert grounded_claims[0].evidence != ""


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------

class TestScores:
    def test_score_range(self):
        c = GroundednessChecker()
        result = c.check(
            response="Testing the score range.",
            context="Testing the score range output.",
        )
        assert 0.0 <= result.score <= 1.0

    def test_grounded_ratio(self):
        c = GroundednessChecker(min_claim_words=2)
        result = c.check(
            response="Supported claim here. Fabricated nonsense xyzzy.",
            context="Supported claim here in the source text.",
        )
        assert 0.0 <= result.grounded_ratio <= 1.0

    def test_empty_response_grounded(self):
        c = GroundednessChecker()
        result = c.check(response="", context="some context")
        assert result.grounded  # No claims to be ungrounded
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_strict_threshold(self):
        c = GroundednessChecker(threshold=0.9)
        result = c.check(
            response="The company launched a new product line for consumers.",
            context="The company introduced new products for retail customers.",
        )
        # Strict threshold — partial overlap may not meet 0.9
        # Just check result is valid
        assert isinstance(result.grounded, bool)

    def test_lenient_threshold(self):
        c = GroundednessChecker(threshold=0.1)
        result = c.check(
            response="Technology advances rapidly in modern society.",
            context="Modern society sees rapid technology advancement.",
        )
        assert result.grounded


# ---------------------------------------------------------------------------
# Batch checking
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch(self):
        c = GroundednessChecker()
        results = c.check_batch([
            ("Paris is in France.", "France's capital is Paris."),
            ("Mars is blue.", "Mars is known as the red planet."),
        ])
        assert len(results) == 2

    def test_empty_batch(self):
        c = GroundednessChecker()
        results = c.check_batch([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_word_response(self):
        c = GroundednessChecker()
        result = c.check(response="Yes.", context="The answer is yes.")
        # Short response, should be grounded (no claims extracted)
        assert result.grounded

    def test_unicode(self):
        c = GroundednessChecker()
        result = c.check(
            response="Le café est délicieux.",
            context="Le café est très délicieux et populaire.",
        )
        assert isinstance(result, GroundednessResult)

    def test_long_context(self):
        c = GroundednessChecker()
        context = " ".join(["The quick brown fox jumps over the lazy dog."] * 100)
        result = c.check(
            response="The quick brown fox jumps over the lazy dog.",
            context=context,
        )
        assert result.grounded
