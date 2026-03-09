"""Tests for bias detection."""

import pytest
from sentinel.bias_detector import BiasDetector, BiasResult


# ---------------------------------------------------------------------------
# Gender bias
# ---------------------------------------------------------------------------

class TestGenderBias:
    def test_capability_generalization(self):
        d = BiasDetector()
        result = d.check("Women can't do math as well as men")
        assert result.biased
        assert "gender" in result.categories

    def test_role_prescription(self):
        d = BiasDetector()
        result = d.check("Women should stay home and take care of children")
        assert result.biased

    def test_stereotyping(self):
        d = BiasDetector()
        result = d.check("That's typical woman behavior")
        assert result.biased

    def test_no_gender_bias(self):
        d = BiasDetector()
        result = d.check("Both men and women make excellent engineers")
        assert not result.biased


# ---------------------------------------------------------------------------
# Racial bias
# ---------------------------------------------------------------------------

class TestRacialBias:
    def test_generalization(self):
        d = BiasDetector()
        result = d.check("All black people are good at sports")
        assert result.biased
        assert "racial" in result.categories

    def test_no_racial_bias(self):
        d = BiasDetector()
        result = d.check("People from diverse backgrounds bring different perspectives")
        assert not result.biased


# ---------------------------------------------------------------------------
# Age bias
# ---------------------------------------------------------------------------

class TestAgeBias:
    def test_elderly_generalization(self):
        d = BiasDetector()
        result = d.check("Old people can't learn new technology")
        assert result.biased
        assert "age" in result.categories

    def test_generational_stereotyping(self):
        d = BiasDetector()
        result = d.check("Millennials are always lazy and entitled")
        assert result.biased

    def test_age_discrimination(self):
        d = BiasDetector()
        result = d.check("She's too old for this job")
        assert result.biased


# ---------------------------------------------------------------------------
# Disability bias
# ---------------------------------------------------------------------------

class TestDisabilityBias:
    def test_ableist_language(self):
        d = BiasDetector()
        result = d.check("That idea is totally crazy and insane")
        assert result.biased
        assert "disability" in result.categories


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------

class TestCategoryFiltering:
    def test_filter_categories(self):
        d = BiasDetector(categories=["gender"])
        result = d.check("Old people can't learn technology")
        assert not result.biased  # Age bias not checked

    def test_only_selected_category(self):
        d = BiasDetector(categories=["age"])
        result = d.check("Old people can't learn technology")
        assert result.biased


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_high_threshold(self):
        d = BiasDetector(threshold=0.9)
        result = d.check("That's typical woman behavior")
        # severity 0.5 < threshold 0.9
        assert not result.biased

    def test_low_threshold(self):
        d = BiasDetector(threshold=0.1)
        result = d.check("Those people are different")
        assert result.biased  # Othering language at 0.4


# ---------------------------------------------------------------------------
# Result properties
# ---------------------------------------------------------------------------

class TestResultProperties:
    def test_bias_count(self):
        d = BiasDetector()
        result = d.check("Women can't do math. That's crazy.")
        assert result.bias_count >= 2

    def test_categories(self):
        d = BiasDetector()
        result = d.check("Women can't drive. Old people can't learn.")
        cats = result.categories
        assert "gender" in cats
        assert "age" in cats

    def test_overall_score(self):
        d = BiasDetector()
        result = d.check("Women should stay home")
        assert result.overall_score > 0.5

    def test_overall_score_clean(self):
        d = BiasDetector()
        result = d.check("Everyone deserves equal treatment")
        assert result.overall_score == 0.0


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch(self):
        d = BiasDetector()
        results = d.check_batch([
            "This is neutral text.",
            "Women can't do math.",
        ])
        assert len(results) == 2
        assert not results[0].biased
        assert results[1].biased


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        d = BiasDetector()
        result = d.check("")
        assert not result.biased

    def test_clean_text(self):
        d = BiasDetector()
        result = d.check("The weather is nice today and the code compiles.")
        assert not result.biased
