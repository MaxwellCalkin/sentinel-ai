"""Tests for configurable content filter."""

import pytest
from sentinel.content_filter import ContentFilter, Category, FilterResult, CategoryScore


# ---------------------------------------------------------------------------
# Violence detection
# ---------------------------------------------------------------------------

class TestViolence:
    def test_detects_violence(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("He threatened to kill the target")
        assert result.blocked
        assert Category.VIOLENCE in result.blocked_categories

    def test_moderate_violence(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("The soldiers attacked the position")
        assert result.blocked

    def test_no_violence(self):
        f = ContentFilter()
        result = f.filter("The weather is nice today")
        assert result.is_clean


# ---------------------------------------------------------------------------
# Weapons detection
# ---------------------------------------------------------------------------

class TestWeapons:
    def test_detects_weapons(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("Instructions for making a bomb")
        assert result.blocked
        assert Category.WEAPONS in result.blocked_categories

    def test_high_severity_weapons(self):
        f = ContentFilter(default_threshold=0.5)
        result = f.filter("Plans for a weapon of mass destruction")
        assert result.blocked


# ---------------------------------------------------------------------------
# Self-harm detection
# ---------------------------------------------------------------------------

class TestSelfHarm:
    def test_detects_self_harm(self):
        f = ContentFilter(default_threshold=0.5)
        result = f.filter("I want to kill myself")
        assert result.blocked
        assert Category.SELF_HARM in result.blocked_categories


# ---------------------------------------------------------------------------
# Personal info detection
# ---------------------------------------------------------------------------

class TestPersonalInfo:
    def test_detects_ssn(self):
        f = ContentFilter(default_threshold=0.5)
        result = f.filter("My SSN is 123-45-6789")
        assert result.blocked
        assert Category.PERSONAL_INFO in result.blocked_categories

    def test_detects_email(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("Contact me at user@example.com")
        assert result.blocked


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_high_threshold_allows_moderate(self):
        f = ContentFilter()
        f.set_threshold(Category.VIOLENCE, 0.9)
        result = f.filter("There was a fight at the bar")
        # fight is weight 0.5, below 0.9 threshold
        assert Category.VIOLENCE not in result.blocked_categories

    def test_low_threshold_blocks_mild(self):
        f = ContentFilter()
        f.set_threshold(Category.PROFANITY, 0.1)
        result = f.filter("What the hell is going on")
        assert result.blocked
        assert Category.PROFANITY in result.blocked_categories


# ---------------------------------------------------------------------------
# Category disable/enable
# ---------------------------------------------------------------------------

class TestCategoryControl:
    def test_disable_category(self):
        f = ContentFilter(default_threshold=0.3)
        f.disable_category(Category.VIOLENCE)
        result = f.filter("He was murdered brutally")
        assert Category.VIOLENCE not in result.blocked_categories

    def test_enable_category(self):
        f = ContentFilter(default_threshold=0.3)
        f.disable_category(Category.VIOLENCE)
        f.enable_category(Category.VIOLENCE)
        result = f.filter("He was murdered brutally")
        assert Category.VIOLENCE in result.blocked_categories


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redact_mode(self):
        f = ContentFilter(default_threshold=0.3, redact=True)
        result = f.filter("My SSN is 123-45-6789")
        assert "[REDACTED]" in result.filtered_text
        assert "123-45-6789" not in result.filtered_text

    def test_no_redact_default(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("My SSN is 123-45-6789")
        assert result.filtered_text == result.text


# ---------------------------------------------------------------------------
# Scores and properties
# ---------------------------------------------------------------------------

class TestScoresAndProperties:
    def test_category_scores(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("He threatened to kill with a bomb")
        assert len(result.scores) >= 1
        for s in result.scores:
            assert 0.0 <= s.score <= 1.0

    def test_top_category(self):
        f = ContentFilter()
        result = f.filter("Plans for a weapon of mass destruction")
        assert result.top_category is not None

    def test_top_score(self):
        f = ContentFilter()
        result = f.filter("Brutal murder scene")
        assert result.top_score > 0

    def test_top_score_clean(self):
        f = ContentFilter()
        result = f.filter("Hello world")
        assert result.top_score == 0.0

    def test_matched_terms(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("The attack was brutal")
        violence_scores = [s for s in result.scores if s.category == Category.VIOLENCE]
        if violence_scores:
            assert len(violence_scores[0].matched_terms) > 0


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch(self):
        f = ContentFilter(default_threshold=0.3)
        results = f.filter_batch([
            "Hello world",
            "Kill the target",
            "Nice weather",
        ])
        assert len(results) == 3
        assert results[0].is_clean
        assert results[1].blocked
        assert results[2].is_clean

    def test_empty_batch(self):
        f = ContentFilter()
        assert f.filter_batch([]) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        f = ContentFilter()
        result = f.filter("")
        assert result.is_clean

    def test_multiple_categories(self):
        f = ContentFilter(default_threshold=0.3)
        result = f.filter("Kill yourself with a bomb and cocaine")
        assert len(result.blocked_categories) >= 2
