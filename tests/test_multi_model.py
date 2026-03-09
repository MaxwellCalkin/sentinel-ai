"""Tests for multi-model guard."""

import pytest
from sentinel.multi_model import MultiModelGuard, ConsistencyResult


# ---------------------------------------------------------------------------
# Basic consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_consistent_outputs(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "Paris is the capital of France"},
            {"model": "gpt", "output": "Paris is the capital of France"},
        ])
        assert result.consistent
        assert result.agreement_score == 1.0

    def test_inconsistent_outputs(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "The answer is 42"},
            {"model": "gpt", "output": "Elephants are large mammals in Africa"},
        ])
        assert not result.consistent
        assert result.agreement_score < 0.7

    def test_partial_agreement(self):
        g = MultiModelGuard(threshold=0.4)
        result = g.compare([
            {"model": "claude", "output": "Python is a great programming language for data science"},
            {"model": "gpt", "output": "Python is an excellent programming language used in data science"},
        ])
        assert result.agreement_score > 0.4

    def test_single_output(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "Hello world"},
        ])
        assert result.consistent
        assert result.agreement_score == 1.0


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

class TestVoting:
    def test_majority_vote(self):
        g = MultiModelGuard()
        answer = g.vote([
            {"model": "claude", "output": "The sky is blue"},
            {"model": "gpt", "output": "The sky is blue"},
            {"model": "llama", "output": "The ocean is green and deep"},
        ])
        assert "sky" in answer.lower()

    def test_two_models_vote(self):
        g = MultiModelGuard()
        answer = g.vote([
            {"model": "claude", "output": "test response"},
            {"model": "gpt", "output": "test response"},
        ])
        assert answer == "test response"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutliers:
    def test_detect_outlier(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "Water boils at 100 degrees Celsius"},
            {"model": "gpt", "output": "Water boils at 100 degrees Celsius at sea level"},
            {"model": "llama", "output": "Bananas are yellow fruit"},
        ])
        assert "llama" in result.outliers

    def test_no_outliers_when_consistent(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "The earth orbits the sun"},
            {"model": "gpt", "output": "The earth orbits the sun"},
        ])
        assert len(result.outliers) == 0

    def test_filter_outliers(self):
        g = MultiModelGuard()
        filtered = g.filter_outliers([
            {"model": "claude", "output": "Water boils at 100 degrees Celsius"},
            {"model": "gpt", "output": "Water boils at 100 degrees Celsius at sea level"},
            {"model": "llama", "output": "Bananas are yellow fruit"},
        ])
        models = [o["model"] for o in filtered]
        assert "llama" not in models


# ---------------------------------------------------------------------------
# Disagreements
# ---------------------------------------------------------------------------

class TestDisagreements:
    def test_disagreement_reported(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "The capital is Paris"},
            {"model": "gpt", "output": "Completely different unrelated text about coding"},
        ])
        assert len(result.disagreements) > 0
        assert "claude vs gpt" in result.disagreements[0]

    def test_no_disagreements(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "a", "output": "same text here"},
            {"model": "b", "output": "same text here"},
        ])
        assert len(result.disagreements) == 0


# ---------------------------------------------------------------------------
# Custom similarity
# ---------------------------------------------------------------------------

class TestCustomSimilarity:
    def test_custom_similarity_fn(self):
        # Always returns 1.0 — everything is consistent
        g = MultiModelGuard(similarity_fn=lambda a, b: 1.0)
        result = g.compare([
            {"model": "a", "output": "completely different"},
            {"model": "b", "output": "text that has nothing in common"},
        ])
        assert result.consistent
        assert result.agreement_score == 1.0

    def test_strict_threshold(self):
        g = MultiModelGuard(threshold=0.99)
        result = g.compare([
            {"model": "a", "output": "nearly the same text"},
            {"model": "b", "output": "nearly the same words"},
        ])
        assert not result.consistent


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_result_structure(self):
        g = MultiModelGuard()
        result = g.compare([
            {"model": "claude", "output": "test"},
            {"model": "gpt", "output": "test"},
        ])
        assert isinstance(result, ConsistencyResult)
        assert isinstance(result.agreement_score, float)
        assert isinstance(result.disagreements, list)
        assert isinstance(result.outliers, list)
        assert result.majority_answer == "test"

    def test_empty_outputs(self):
        g = MultiModelGuard()
        result = g.compare([])
        assert result.consistent
        assert result.majority_answer == ""
