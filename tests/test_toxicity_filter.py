"""Tests for multi-category toxicity detection and filtering."""

import pytest
from sentinel.toxicity_filter import (
    ToxicityFilter,
    ToxicityMatch,
    ToxicityResult,
    ToxicityStats,
)


# ---------------------------------------------------------------------------
# Hate speech detection
# ---------------------------------------------------------------------------

class TestHateSpeech:
    def test_detects_supremacy_and_ethnic_cleansing(self):
        f = ToxicityFilter()
        result_supremacy = f.scan("white supremacy is on the rise")
        assert result_supremacy.is_toxic
        assert "hate_speech" in result_supremacy.categories_triggered

        result_ethnic = f.scan("they advocate for ethnic cleansing")
        assert result_ethnic.is_toxic
        assert "hate_speech" in result_ethnic.categories_triggered


# ---------------------------------------------------------------------------
# Harassment detection
# ---------------------------------------------------------------------------

class TestHarassment:
    def test_detects_personal_attacks(self):
        f = ToxicityFilter()
        result = f.scan("you're worthless and nobody likes you")
        assert result.is_toxic
        assert "harassment" in result.categories_triggered

    def test_mild_insults_below_default_threshold(self):
        f = ToxicityFilter(threshold=0.5)
        result = f.scan("shut up already")
        assert not result.is_toxic


# ---------------------------------------------------------------------------
# Threats detection
# ---------------------------------------------------------------------------

class TestThreats:
    def test_detects_death_threats(self):
        f = ToxicityFilter()
        result = f.scan("I will kill you for this")
        assert result.is_toxic
        assert "threats" in result.categories_triggered

    def test_detects_tracking_threats(self):
        f = ToxicityFilter()
        result = f.scan("I will find you and make you pay")
        assert result.is_toxic
        assert "threats" in result.categories_triggered


# ---------------------------------------------------------------------------
# Profanity detection
# ---------------------------------------------------------------------------

class TestProfanity:
    def test_detects_strong_profanity(self):
        f = ToxicityFilter(threshold=0.3)
        result = f.scan("what the fuck is this shit")
        assert result.is_toxic
        assert "profanity" in result.categories_triggered


# ---------------------------------------------------------------------------
# Self-harm detection
# ---------------------------------------------------------------------------

class TestSelfHarm:
    def test_detects_suicidal_content(self):
        f = ToxicityFilter()
        result = f.scan("I want to kill myself")
        assert result.is_toxic
        assert "self_harm" in result.categories_triggered


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_and_empty_text_not_flagged(self):
        f = ToxicityFilter()
        result = f.scan("The weather is lovely today")
        assert not result.is_toxic
        assert result.score == 0.0
        assert result.matches == []
        assert result.categories_triggered == []

        empty_result = f.scan("")
        assert not empty_result.is_toxic
        assert empty_result.score == 0.0


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_low_threshold_catches_mild_content(self):
        f = ToxicityFilter(threshold=0.15)
        result = f.scan("damn this is annoying")
        assert result.is_toxic

    def test_high_threshold_allows_moderate_content(self):
        f = ToxicityFilter(threshold=0.99)
        result = f.scan("you're pathetic, you loser")
        assert not result.is_toxic


# ---------------------------------------------------------------------------
# Filter / redaction mode
# ---------------------------------------------------------------------------

class TestFilterRedaction:
    def test_filter_replaces_toxic_text(self):
        f = ToxicityFilter(threshold=0.3)
        filtered = f.filter("what the fuck is going on")
        assert "[FILTERED]" in filtered
        assert "fuck" not in filtered

    def test_filter_returns_original_when_clean(self):
        f = ToxicityFilter()
        original = "This is a perfectly fine sentence"
        assert f.filter(original) == original


# ---------------------------------------------------------------------------
# Custom category registration
# ---------------------------------------------------------------------------

class TestCustomCategory:
    def test_register_and_detect_custom_category(self):
        f = ToxicityFilter(threshold=0.3)
        f.register_category("spam", [
            (r"\bbuy now\b", 0.6),
            (r"\bfree money\b", 0.7),
        ])
        assert "spam" in f.categories
        result = f.scan("buy now and get free money")
        assert result.is_toxic
        assert "spam" in result.categories_triggered


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_increases_with_multiple_categories(self):
        f = ToxicityFilter(threshold=0.1)
        single = f.scan("you're worthless")
        multi = f.scan("you're worthless, I will kill you, burn down everything")
        assert multi.score > single.score

    def test_score_bounded_at_one(self):
        f = ToxicityFilter(threshold=0.1)
        result = f.scan(
            "supremacy ethnic cleansing inferior race subhuman "
            "I will kill you I will destroy you death threat "
            "you're worthless you're pathetic loser moron"
        )
        assert result.score <= 1.0


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatch:
    def test_scan_batch_returns_correct_count(self):
        f = ToxicityFilter()
        results = f.scan_batch([
            "Hello there, nice day",
            "I will kill you",
            "Great weather outside",
        ])
        assert len(results) == 3
        assert not results[0].is_toxic
        assert results[1].is_toxic
        assert not results[2].is_toxic


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_tracking(self):
        f = ToxicityFilter()
        f.scan("The weather is nice")
        f.scan("I will kill you for this")
        f.scan("Have a great day")
        stats = f.stats()
        assert stats.total_scanned == 3
        assert stats.flagged_count == 1
        assert stats.flag_rate == pytest.approx(1 / 3, abs=0.01)

    def test_stats_by_category(self):
        f = ToxicityFilter()
        f.scan("I will kill you")
        f.scan("ethnic cleansing is evil")
        stats = f.stats()
        assert "threats" in stats.by_category
        assert "hate_speech" in stats.by_category

    def test_stats_empty_on_fresh_instance(self):
        f = ToxicityFilter()
        stats = f.stats()
        assert stats.total_scanned == 0
        assert stats.flagged_count == 0
        assert stats.flag_rate == 0.0
        assert stats.by_category == {}


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_toxicity_match_fields(self):
        m = ToxicityMatch(text="kill", category="threats", severity=0.9, pattern=r"\bkill\b")
        assert m.text == "kill"
        assert m.category == "threats"
        assert m.severity == 0.9
        assert m.pattern == r"\bkill\b"

    def test_toxicity_result_fields(self):
        r = ToxicityResult(text="hello", is_toxic=False, score=0.0)
        assert r.text == "hello"
        assert not r.is_toxic
        assert r.matches == []
        assert r.categories_triggered == []
