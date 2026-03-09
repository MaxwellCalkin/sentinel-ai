"""Tests for sensitivity classifier."""

import pytest
from sentinel.sensitivity_classifier import SensitivityClassifier, SensitivityResult


class TestClassification:
    def test_public_content(self):
        c = SensitivityClassifier()
        result = c.classify("What is the weather today?")
        assert result.level == "public"

    def test_restricted_ssn(self):
        c = SensitivityClassifier()
        result = c.classify("What is the employee's social security number?")
        assert result.level == "restricted"

    def test_restricted_medical(self):
        c = SensitivityClassifier()
        result = c.classify("Show me the patient diagnosis records")
        assert result.level == "restricted"

    def test_confidential_salary(self):
        c = SensitivityClassifier()
        result = c.classify("What is the CEO's salary and bonus?")
        assert result.level == "confidential"

    def test_confidential_financial(self):
        c = SensitivityClassifier()
        result = c.classify("Show the quarterly revenue and profit numbers")
        assert result.level == "confidential"

    def test_internal_roadmap(self):
        c = SensitivityClassifier()
        result = c.classify("Share the product roadmap for next sprint")
        assert result.level == "internal"

    def test_restricted_overrides_lower(self):
        c = SensitivityClassifier()
        result = c.classify("Show the patient medical record and salary details")
        assert result.level == "restricted"


class TestTriggers:
    def test_triggers_populated(self):
        c = SensitivityClassifier()
        result = c.classify("Send me the credit card number")
        assert len(result.triggers) > 0
        assert any("restricted" in t for t in result.triggers)

    def test_multiple_triggers(self):
        c = SensitivityClassifier()
        result = c.classify("Show salary, revenue, and profit data")
        assert len(result.triggers) >= 2

    def test_no_triggers_for_public(self):
        c = SensitivityClassifier()
        result = c.classify("Hello, how are you?")
        assert len(result.triggers) == 0


class TestActions:
    def test_restricted_actions(self):
        c = SensitivityClassifier()
        result = c.classify("Show the encryption key")
        assert "Require explicit authorization" in result.recommended_actions

    def test_public_actions(self):
        c = SensitivityClassifier()
        result = c.classify("What is Python?")
        assert "No restrictions" in result.recommended_actions


class TestCustomPatterns:
    def test_custom_pattern(self):
        c = SensitivityClassifier(custom_patterns={
            "restricted": [(r'(?i)\bproject\s+alpha\b', "Secret project")],
        })
        result = c.classify("Tell me about Project Alpha")
        assert result.level == "restricted"


class TestBatch:
    def test_batch_classify(self):
        c = SensitivityClassifier()
        results = c.classify_batch([
            "What is the weather?",
            "Show the patient diagnosis",
        ])
        assert len(results) == 2
        assert results[0].level == "public"
        assert results[1].level == "restricted"


class TestStructure:
    def test_result_structure(self):
        c = SensitivityClassifier()
        result = c.classify("Test input")
        assert isinstance(result, SensitivityResult)
        assert isinstance(result.triggers, list)
        assert isinstance(result.recommended_actions, list)
        assert 0.0 <= result.confidence <= 1.0
