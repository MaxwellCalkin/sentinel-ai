"""Tests for risk scorecard generation."""

import pytest
from sentinel.risk_scorecard import (
    RiskScorecard,
    ScorecardResult,
    Finding,
    DimensionScore,
    ComparisonResult,
)


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_add_dimension(self):
        card = RiskScorecard("test")
        card.add_dimension("injection", score=0.9, weight=2.0, details="Prompt injection safety")
        result = card.calculate()
        assert len(result.dimensions) == 1
        assert result.dimensions[0].name == "injection"
        assert result.dimensions[0].score == 0.9

    def test_multiple_dimensions(self):
        card = RiskScorecard("test")
        card.add_dimension("injection", score=0.9)
        card.add_dimension("pii", score=0.8)
        card.add_dimension("toxicity", score=0.7)
        result = card.calculate()
        assert len(result.dimensions) == 3

    def test_weighted_score(self):
        card = RiskScorecard("weighted")
        card.add_dimension("high_weight", score=1.0, weight=9.0)
        card.add_dimension("low_weight", score=0.0, weight=1.0)
        result = card.calculate()
        # Weighted avg: (1.0*9 + 0.0*1) / 10 = 0.9
        assert result.overall_score == 0.9
        assert result.grade == "A"


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class TestFindings:
    def test_add_finding(self):
        card = RiskScorecard("test")
        card.add_dimension("injection", score=0.95)
        card.add_finding("injection", "medium", "SQL-like pattern detected")
        result = card.calculate()
        assert result.findings_count == 1
        assert result.dimensions[0].findings[0].description == "SQL-like pattern detected"

    def test_finding_reduces_score(self):
        card_clean = RiskScorecard("clean")
        card_clean.add_dimension("injection", score=0.95)

        card_dirty = RiskScorecard("dirty")
        card_dirty.add_dimension("injection", score=0.95)
        card_dirty.add_finding("injection", "high", "Injection attempt found")

        clean_result = card_clean.calculate()
        dirty_result = card_dirty.calculate()

        # high severity penalty = 0.2, so 0.95 - 0.2 = 0.75
        assert dirty_result.overall_score < clean_result.overall_score
        assert dirty_result.dimensions[0].score == pytest.approx(0.75, abs=0.01)

    def test_critical_finding(self):
        card = RiskScorecard("test")
        card.add_dimension("security", score=0.9)
        card.add_finding("security", "critical", "RCE vulnerability")
        result = card.calculate()
        assert result.critical_count == 1
        assert not result.passed

    def test_invalid_severity_raises(self):
        card = RiskScorecard("test")
        card.add_dimension("security", score=0.9)
        with pytest.raises(ValueError, match="Invalid severity"):
            card.add_finding("security", "extreme", "Bad severity")

    def test_missing_dimension_raises(self):
        card = RiskScorecard("test")
        with pytest.raises(KeyError, match="not found"):
            card.add_finding("nonexistent", "low", "No dimension")


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------

class TestCalculation:
    def test_calculate_basic(self):
        card = RiskScorecard("basic")
        card.add_dimension("a", score=0.8)
        card.add_dimension("b", score=0.6)
        result = card.calculate()
        # Equal weights: (0.8 + 0.6) / 2 = 0.7
        assert result.overall_score == pytest.approx(0.7, abs=0.01)
        assert result.name == "basic"

    def test_grade_assignment(self):
        cases = [
            (0.95, "A"),
            (0.85, "B"),
            (0.75, "C"),
            (0.65, "D"),
            (0.5, "F"),
        ]
        for score, expected_grade in cases:
            card = RiskScorecard("grade-test")
            card.add_dimension("dim", score=score)
            result = card.calculate()
            assert result.grade == expected_grade, (
                f"Score {score} should yield grade {expected_grade}, got {result.grade}"
            )

    def test_passed_threshold(self):
        card_pass = RiskScorecard("pass")
        card_pass.add_dimension("dim", score=0.75)
        assert card_pass.calculate().passed is True

        card_fail_score = RiskScorecard("fail-score")
        card_fail_score.add_dimension("dim", score=0.6)
        assert card_fail_score.calculate().passed is False

        card_fail_critical = RiskScorecard("fail-critical")
        card_fail_critical.add_dimension("dim", score=0.9)
        card_fail_critical.add_finding("dim", "critical", "Blocklist bypass")
        assert card_fail_critical.calculate().passed is False


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_improved(self):
        old_card = RiskScorecard("v1")
        old_card.add_dimension("injection", score=0.6)
        old_card.add_finding("injection", "high", "Old vulnerability")

        new_card = RiskScorecard("v2")
        new_card.add_dimension("injection", score=0.9)

        comparison = new_card.compare(old_card)
        assert comparison.improved is True
        assert comparison.score_delta > 0
        assert comparison.resolved_findings == 1
        assert comparison.new_findings == 0

    def test_compare_declined(self):
        old_card = RiskScorecard("v1")
        old_card.add_dimension("injection", score=0.9)

        new_card = RiskScorecard("v2")
        new_card.add_dimension("injection", score=0.5)
        new_card.add_finding("injection", "high", "New vulnerability")

        comparison = new_card.compare(old_card)
        assert comparison.improved is False
        assert comparison.score_delta < 0
        assert comparison.new_findings == 1
        assert comparison.resolved_findings == 0


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_dict(self):
        card = RiskScorecard("export-test")
        card.add_dimension("injection", score=0.9, weight=2.0, details="Injection check")
        card.add_finding("injection", "low", "Minor pattern")

        exported = card.export()
        assert exported["name"] == "export-test"
        assert len(exported["dimensions"]) == 1
        assert exported["dimensions"][0]["name"] == "injection"
        assert exported["dimensions"][0]["score"] == 0.9
        assert exported["dimensions"][0]["weight"] == 2.0
        assert len(exported["findings"]) == 1
        assert exported["findings"][0]["severity"] == "low"

    def test_from_dict(self):
        data = {
            "name": "imported",
            "dimensions": [
                {"name": "pii", "score": 0.85, "weight": 1.5, "details": "PII scan"},
            ],
            "findings": [
                {"dimension": "pii", "severity": "medium", "description": "Email found"},
            ],
        }
        card = RiskScorecard.from_dict(data)
        assert card.name == "imported"
        result = card.calculate()
        assert len(result.dimensions) == 1
        assert result.findings_count == 1
        # 0.85 - 0.1 (medium penalty) = 0.75
        assert result.dimensions[0].score == pytest.approx(0.75, abs=0.01)

    def test_roundtrip(self):
        card = RiskScorecard("roundtrip")
        card.add_dimension("injection", score=0.95, weight=3.0, details="Prompt injection")
        card.add_dimension("pii", score=0.8, weight=2.0, details="PII detection")
        card.add_finding("injection", "low", "Minor suspicious pattern")
        card.add_finding("pii", "medium", "Phone number detected")

        exported = card.export()
        restored = RiskScorecard.from_dict(exported)

        original_result = card.calculate()
        restored_result = restored.calculate()

        assert original_result.overall_score == restored_result.overall_score
        assert original_result.grade == restored_result.grade
        assert original_result.findings_count == restored_result.findings_count
        assert original_result.passed == restored_result.passed


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears(self):
        card = RiskScorecard("reset-test")
        card.add_dimension("injection", score=0.9)
        card.add_finding("injection", "high", "Some finding")
        card.reset()

        result = card.calculate()
        assert len(result.dimensions) == 0
        assert result.findings_count == 0
        assert result.overall_score == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_scorecard(self):
        card = RiskScorecard("empty")
        result = card.calculate()
        assert result.overall_score == 0.0
        assert result.grade == "F"
        assert result.findings_count == 0
        assert result.passed is False

    def test_score_clamp(self):
        card = RiskScorecard("clamp")
        card.add_dimension("over", score=1.5)
        card.add_dimension("under", score=-0.5)
        result = card.calculate()
        # Clamped to 1.0 and 0.0 respectively; avg = 0.5
        assert result.dimensions[0].score <= 1.0
        assert result.dimensions[1].score >= 0.0
        assert result.overall_score == pytest.approx(0.5, abs=0.01)

    def test_finding_penalty_does_not_go_below_zero(self):
        card = RiskScorecard("penalty-floor")
        card.add_dimension("dim", score=0.1)
        card.add_finding("dim", "critical", "Severe issue")
        card.add_finding("dim", "critical", "Another severe issue")
        result = card.calculate()
        # 0.1 - 0.4 - 0.4 = -0.7, clamped to 0.0
        assert result.dimensions[0].score == 0.0

    def test_info_finding_no_penalty(self):
        card = RiskScorecard("info-test")
        card.add_dimension("dim", score=0.9)
        card.add_finding("dim", "info", "Informational note")
        result = card.calculate()
        assert result.dimensions[0].score == 0.9
        assert result.findings_count == 1
