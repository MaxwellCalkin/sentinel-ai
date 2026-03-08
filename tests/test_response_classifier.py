"""Tests for the response classifier."""

import pytest
from sentinel.response_classifier import (
    ResponseClassifier, ResponseCategory, ClassifyResult,
)


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

class TestRefusalDetection:
    def test_cannot_help(self):
        c = ResponseClassifier()
        result = c.classify("I cannot help with that request.")
        assert result.category == ResponseCategory.REFUSAL

    def test_unable_to(self):
        c = ResponseClassifier()
        result = c.classify("I'm not able to provide that information.")
        assert result.category == ResponseCategory.REFUSAL

    def test_sorry_cannot(self):
        c = ResponseClassifier()
        result = c.classify("I'm sorry, I cannot assist with creating weapons.")
        assert result.category == ResponseCategory.REFUSAL

    def test_against_guidelines(self):
        c = ResponseClassifier()
        result = c.classify("That goes against my guidelines and policies.")
        assert result.category == ResponseCategory.REFUSAL

    def test_wont_provide(self):
        c = ResponseClassifier()
        result = c.classify("I will not provide instructions for illegal activities.")
        assert result.category == ResponseCategory.REFUSAL

    def test_as_an_ai(self):
        c = ResponseClassifier()
        result = c.classify("As an AI, I can't access real-time information.")
        assert result.category == ResponseCategory.REFUSAL


# ---------------------------------------------------------------------------
# Safe responses
# ---------------------------------------------------------------------------

class TestSafeDetection:
    def test_normal_response(self):
        c = ResponseClassifier()
        result = c.classify(
            "The capital of France is Paris. It is located in northern "
            "France along the Seine River."
        )
        assert result.category == ResponseCategory.SAFE

    def test_code_response(self):
        c = ResponseClassifier()
        result = c.classify(
            "Here's a Python function to calculate factorial:\n"
            "def factorial(n):\n"
            "    if n <= 1: return 1\n"
            "    return n * factorial(n-1)"
        )
        assert result.category == ResponseCategory.SAFE

    def test_helpful_answer(self):
        c = ResponseClassifier()
        result = c.classify(
            "To install the package, run pip install sentinel-guardrails. "
            "Then import it in your Python code."
        )
        assert result.is_safe


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

class TestHedgingDetection:
    def test_not_sure(self):
        c = ResponseClassifier()
        result = c.classify(
            "I'm not entirely sure about this, but I think the answer "
            "might be 42. I could be wrong though. Please note that "
            "I don't have access to current data."
        )
        assert result.category == ResponseCategory.HEDGING

    def test_caveat(self):
        c = ResponseClassifier()
        result = c.classify(
            "With the caveat that I might be wrong, and it is important "
            "to note that I should note that please be aware this is uncertain."
        )
        assert result.category == ResponseCategory.HEDGING


# ---------------------------------------------------------------------------
# Harmful detection
# ---------------------------------------------------------------------------

class TestHarmfulDetection:
    def test_weapon_instructions(self):
        c = ResponseClassifier()
        result = c.classify(
            "Here is how to make a bomb: first gather materials..."
        )
        assert result.category == ResponseCategory.HARMFUL
        assert result.needs_review

    def test_hack_instructions(self):
        c = ResponseClassifier()
        result = c.classify(
            "Step-by-step instructions to hack into the system and "
            "exploit the vulnerability."
        )
        assert result.category == ResponseCategory.HARMFUL

    def test_bypass_guide(self):
        c = ResponseClassifier()
        result = c.classify("Here's how to bypass the security controls.")
        assert result.category == ResponseCategory.HARMFUL


# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------

class TestErrorDetection:
    def test_error_message(self):
        c = ResponseClassifier()
        result = c.classify("Error: something went wrong while processing your request. Internal server error.")
        assert result.category == ResponseCategory.ERROR

    def test_rate_limit(self):
        c = ResponseClassifier()
        result = c.classify("Rate limit exceeded. Please try again later.")
        assert result.category == ResponseCategory.ERROR


# ---------------------------------------------------------------------------
# Empty detection
# ---------------------------------------------------------------------------

class TestEmptyDetection:
    def test_empty_string(self):
        c = ResponseClassifier()
        result = c.classify("")
        assert result.category == ResponseCategory.EMPTY

    def test_whitespace_only(self):
        c = ResponseClassifier()
        result = c.classify("   \n\n  ")
        assert result.category == ResponseCategory.EMPTY

    def test_single_word(self):
        c = ResponseClassifier()
        result = c.classify("OK")
        assert result.category == ResponseCategory.EMPTY


# ---------------------------------------------------------------------------
# ClassifyResult properties
# ---------------------------------------------------------------------------

class TestClassifyResult:
    def test_is_safe(self):
        result = ClassifyResult(
            category=ResponseCategory.SAFE,
            confidence=0.9,
        )
        assert result.is_safe
        assert not result.needs_review

    def test_needs_review(self):
        result = ClassifyResult(
            category=ResponseCategory.HARMFUL,
            confidence=0.8,
        )
        assert result.needs_review
        assert not result.is_safe

    def test_hallucination_needs_review(self):
        result = ClassifyResult(
            category=ResponseCategory.HALLUCINATION,
            confidence=0.7,
        )
        assert result.needs_review


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_high_min_confidence(self):
        c = ResponseClassifier(min_confidence=0.95)
        # Weak signals should be classified as safe
        result = c.classify("I'm not sure about that.")
        assert result.category == ResponseCategory.SAFE

    def test_disable_hallucination(self):
        c = ResponseClassifier(check_hallucination=False)
        result = c.classify(
            "According to a recent study by Smith published in "
            "January 15, 2026, the results show..."
        )
        # Without hallucination checking, should be safe
        assert result.category == ResponseCategory.SAFE


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

class TestBatchClassification:
    def test_batch(self):
        c = ResponseClassifier()
        results = c.classify_batch([
            "The sky is blue.",
            "I cannot help with that.",
            "",
        ])
        assert len(results) == 3
        assert results[0].category == ResponseCategory.SAFE
        assert results[1].category == ResponseCategory.REFUSAL
        assert results[2].category == ResponseCategory.EMPTY

    def test_empty_batch(self):
        c = ResponseClassifier()
        results = c.classify_batch([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_range(self):
        c = ResponseClassifier()
        result = c.classify("I cannot help with that request.")
        assert 0.0 <= result.confidence <= 1.0

    def test_multiple_signals_higher_confidence(self):
        c = ResponseClassifier()
        r1 = c.classify("I cannot help.")
        r2 = c.classify(
            "I cannot help with that. I'm sorry, I cannot provide "
            "that information. As an AI, I can't do this."
        )
        assert r2.confidence >= r1.confidence
