"""Tests for composite safety scoring."""

import pytest
from sentinel.safety_score import SafetyScorer, SafetyScore


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------

class TestBasicScoring:
    def test_perfect_score(self):
        s = SafetyScorer()
        result = s.score("safe text", context={})
        assert result.value == 100
        assert result.grade == "A"
        assert result.is_safe

    def test_dangerous_score(self):
        s = SafetyScorer()
        result = s.score("", context={
            "injection_score": 0.9,
            "toxicity_score": 0.9,
            "harmful_score": 0.9,
            "pii_detected": True,
            "hallucination_risk": 0.8,
            "bias_score": 0.8,
            "profanity_score": 0.9,
        })
        assert result.value < 30
        assert result.grade == "F"
        assert result.is_critical

    def test_medium_score(self):
        s = SafetyScorer()
        result = s.score("", context={
            "injection_score": 0.3,
            "hallucination_risk": 0.4,
        })
        assert 50 <= result.value <= 95

    def test_score_range(self):
        s = SafetyScorer()
        result = s.score("", context={"injection_score": 0.5})
        assert 0 <= result.value <= 100


# ---------------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------------

class TestGrades:
    def test_grade_a(self):
        s = SafetyScorer()
        result = s.score("", context={})
        assert result.grade == "A"

    def test_grade_f(self):
        s = SafetyScorer()
        result = s.score("", context={
            "injection_score": 1.0,
            "toxicity_score": 1.0,
            "harmful_score": 1.0,
            "pii_detected": True,
        })
        assert result.grade == "F"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class TestComponents:
    def test_components_tracked(self):
        s = SafetyScorer()
        result = s.score("", context={
            "injection_score": 0.3,
            "toxicity_score": 0.1,
        })
        assert "injection" in result.components
        assert "toxicity" in result.components
        assert result.components["injection"] == 0.3

    def test_pii_boolean(self):
        s = SafetyScorer()
        result = s.score("", context={"pii_detected": True})
        assert result.components["pii"] == 1.0

    def test_pii_score(self):
        s = SafetyScorer()
        result = s.score("", context={"pii_score": 0.5})
        assert result.components["pii"] == 0.5


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestFlags:
    def test_high_risk_flagged(self):
        s = SafetyScorer()
        result = s.score("", context={"injection_score": 0.8})
        assert any("injection" in f for f in result.flags)

    def test_no_flags_when_safe(self):
        s = SafetyScorer()
        result = s.score("", context={"injection_score": 0.1})
        assert len(result.flags) == 0


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestWeights:
    def test_custom_weights(self):
        s1 = SafetyScorer(weights={"injection": 10.0, "toxicity": 1.0})
        s2 = SafetyScorer(weights={"injection": 1.0, "toxicity": 10.0})

        ctx = {"injection_score": 0.8, "toxicity_score": 0.1}
        r1 = s1.score("", context=ctx)
        r2 = s2.score("", context=ctx)
        # High injection weight should make r1 lower
        assert r1.value < r2.value


# ---------------------------------------------------------------------------
# Custom components
# ---------------------------------------------------------------------------

class TestCustomComponents:
    def test_custom_component(self):
        s = SafetyScorer()
        s.add_component("length_risk", lambda text, ctx: min(1.0, len(text) / 1000), weight=1.0)
        result = s.score("x" * 500, context={})
        assert "length_risk" in result.components
        assert result.components["length_risk"] == 0.5

    def test_custom_component_error_safe(self):
        s = SafetyScorer()
        s.add_component("bad", lambda text, ctx: 1 / 0, weight=1.0)  # Will throw
        result = s.score("text", context={})
        assert result.components["bad"] == 0.0  # Error defaults to 0


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch(self):
        s = SafetyScorer()
        results = s.score_batch([
            ("safe", {}),
            ("risky", {"injection_score": 0.9}),
        ])
        assert len(results) == 2
        assert results[0].value > results[1].value


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_is_safe(self):
        r = SafetyScore(value=80, grade="B")
        assert r.is_safe

    def test_not_safe(self):
        r = SafetyScore(value=50, grade="D")
        assert not r.is_safe

    def test_is_critical(self):
        r = SafetyScore(value=20, grade="F")
        assert r.is_critical

    def test_not_critical(self):
        r = SafetyScore(value=50, grade="D")
        assert not r.is_critical
