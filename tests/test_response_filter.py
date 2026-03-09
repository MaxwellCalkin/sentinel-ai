"""Tests for post-processing response filter."""

import pytest
from sentinel.response_filter import ResponseFilter, FilterResult, FilterStats


# ---------------------------------------------------------------------------
# Custom filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_add_filter(self):
        rf = ResponseFilter()
        rf.add_filter("upper", str.upper)
        result = rf.apply("hello")
        assert result.filtered == "HELLO"
        assert "upper" in result.filters_applied

    def test_remove_filter(self):
        rf = ResponseFilter()
        rf.add_filter("upper", str.upper)
        rf.remove_filter("upper")
        result = rf.apply("hello")
        assert result.filtered == "hello"
        assert result.filters_applied == []

    def test_list_filters(self):
        rf = ResponseFilter()
        rf.add_filter("low", str.lower, priority=1)
        rf.add_filter("strip", str.strip, priority=5)
        rf.add_filter("title", str.title, priority=3)
        names = rf.list_filters()
        assert names == ["strip", "title", "low"]

    def test_filter_priority_order(self):
        """Filters with higher priority run first."""
        rf = ResponseFilter()
        rf.add_filter("add_prefix", lambda t: "PREFIX:" + t, priority=10)
        rf.add_filter("upper", str.upper, priority=5)
        result = rf.apply("hello")
        # add_prefix runs first (priority 10), then upper (priority 5)
        assert result.filtered == "PREFIX:HELLO"


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------


class TestBlocklist:
    def test_blocklist_replacement(self):
        rf = ResponseFilter()
        rf.add_blocklist(["badword", "evil"])
        result = rf.apply("This contains badword and evil stuff")
        assert "badword" not in result.filtered
        assert "evil" not in result.filtered
        assert result.filtered == "This contains [BLOCKED] and [BLOCKED] stuff"
        assert result.blocked_words_found == 2

    def test_blocklist_case_insensitive(self):
        rf = ResponseFilter()
        rf.add_blocklist(["secret"])
        result = rf.apply("This is SECRET data and Secret info")
        assert "SECRET" not in result.filtered
        assert "Secret" not in result.filtered
        assert result.blocked_words_found == 2


# ---------------------------------------------------------------------------
# Regex replacement
# ---------------------------------------------------------------------------


class TestReplacement:
    def test_regex_replacement(self):
        rf = ResponseFilter()
        rf.add_replacement(r"\d{4}-\d{4}-\d{4}-\d{4}", "[CARD_REDACTED]")
        result = rf.apply("Card number is 1234-5678-9012-3456")
        assert result.filtered == "Card number is [CARD_REDACTED]"
        assert "replacement" in result.filters_applied


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_basic(self):
        rf = ResponseFilter()
        rf.add_blocklist(["dangerous"])
        result = rf.apply("This is dangerous content")
        assert isinstance(result, FilterResult)
        assert result.original == "This is dangerous content"
        assert "[BLOCKED]" in result.filtered
        assert result.modifications > 0
        assert result.blocked_words_found == 1

    def test_apply_no_changes(self):
        rf = ResponseFilter()
        result = rf.apply("Clean text with no issues")
        assert result.filtered == result.original
        assert result.modifications == 0
        assert result.filters_applied == []
        assert result.truncated is False
        assert result.blocked_words_found == 0

    def test_apply_with_pii_strip(self):
        rf = ResponseFilter(strip_pii=True)
        result = rf.apply("Contact user@example.com or call 555-123-4567")
        assert "user@example.com" not in result.filtered
        assert "555-123-4567" not in result.filtered
        assert "[PII_REDACTED]" in result.filtered
        assert "pii_strip" in result.filters_applied

    def test_truncation(self):
        rf = ResponseFilter(max_length=10)
        result = rf.apply("This is a very long text that exceeds the limit")
        assert len(result.filtered) == 13  # 10 chars + "..."
        assert result.filtered.endswith("...")
        assert result.truncated is True
        assert "truncation" in result.filters_applied


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_apply_batch(self):
        rf = ResponseFilter()
        rf.add_blocklist(["bad"])
        results = rf.apply_batch(["good text", "bad text", "another good"])
        assert len(results) == 3
        assert results[0].blocked_words_found == 0
        assert results[1].blocked_words_found == 1
        assert results[2].blocked_words_found == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_tracking(self):
        rf = ResponseFilter(max_length=20)
        rf.add_blocklist(["foo"])
        rf.apply("foo bar")
        rf.apply("a very long text that will definitely be truncated here")
        s = rf.stats()
        assert s.total_processed == 2
        assert s.total_blocked_words == 1
        assert s.total_truncated == 1
        assert s.total_modifications > 0

    def test_avg_modifications(self):
        rf = ResponseFilter()
        rf.add_blocklist(["x"])
        rf.apply("x")
        rf.apply("x")
        s = rf.stats()
        assert s.total_processed == 2
        assert s.avg_modifications == s.total_modifications / 2


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets(self):
        rf = ResponseFilter()
        rf.add_filter("upper", str.upper)
        rf.add_blocklist(["word"])
        rf.add_replacement(r"\d+", "NUM")
        rf.apply("word 123")
        rf.clear()
        assert rf.list_filters() == []
        s = rf.stats()
        assert s.total_processed == 0
        assert s.total_modifications == 0
        result = rf.apply("word 123")
        assert result.filtered == "word 123"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdge:
    def test_empty_text(self):
        rf = ResponseFilter(strip_pii=True, max_length=100)
        rf.add_blocklist(["word"])
        result = rf.apply("")
        assert result.filtered == ""
        assert result.modifications == 0
        assert result.truncated is False

    def test_no_filters(self):
        rf = ResponseFilter()
        result = rf.apply("Some text goes here")
        assert result.filtered == "Some text goes here"
        assert result.modifications == 0
        assert result.filters_applied == []
