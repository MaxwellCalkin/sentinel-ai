"""Tests for feedback loop."""

import pytest
from sentinel.feedback_loop import FeedbackLoop, FeedbackEntry, FeedbackReport


# ---------------------------------------------------------------------------
# Recording feedback
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_feedback(self):
        loop = FeedbackLoop()
        entry = loop.record("out-1", rating=4, comment="Good")
        assert entry.output_id == "out-1"
        assert entry.rating == 4
        assert entry.positive

    def test_thumbs_up(self):
        loop = FeedbackLoop()
        entry = loop.thumbs_up("out-1", comment="Great!")
        assert entry.positive
        assert entry.rating == 5

    def test_thumbs_down(self):
        loop = FeedbackLoop()
        entry = loop.thumbs_down("out-1", correction="Better answer")
        assert not entry.positive
        assert entry.rating == 1
        assert entry.correction == "Better answer"

    def test_rating_clamped(self):
        loop = FeedbackLoop()
        e1 = loop.record("a", rating=0)
        e2 = loop.record("b", rating=10)
        assert e1.rating == 1
        assert e2.rating == 5

    def test_labels(self):
        loop = FeedbackLoop()
        entry = loop.record("out-1", labels=["accuracy", "format"])
        assert "accuracy" in entry.labels


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class TestReports:
    def test_basic_report(self):
        loop = FeedbackLoop()
        loop.record("a", rating=5)
        loop.record("b", rating=4)
        loop.record("c", rating=2)
        report = loop.report()
        assert report.total == 3
        assert report.positive_count == 2
        assert report.negative_count == 1
        assert 3.0 < report.avg_rating < 4.0

    def test_empty_report(self):
        loop = FeedbackLoop()
        report = loop.report()
        assert report.total == 0
        assert report.avg_rating == 0.0
        assert report.recent_trend == "stable"

    def test_positive_rate(self):
        loop = FeedbackLoop()
        loop.thumbs_up("a")
        loop.thumbs_up("b")
        loop.thumbs_down("c")
        report = loop.report()
        assert abs(report.positive_rate - 0.6667) < 0.01

    def test_by_label(self):
        loop = FeedbackLoop()
        loop.record("a", rating=5, labels=["accuracy"])
        loop.record("b", rating=1, labels=["accuracy"])
        loop.record("c", rating=4, labels=["format"])
        report = loop.report()
        assert "accuracy" in report.by_label
        assert report.by_label["accuracy"]["count"] == 2

    def test_corrections_count(self):
        loop = FeedbackLoop()
        loop.thumbs_down("a", correction="Fixed answer")
        loop.thumbs_up("b")
        report = loop.report()
        assert report.corrections_count == 1


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

class TestTrends:
    def test_improving_trend(self):
        loop = FeedbackLoop()
        for i in range(10):
            loop.record(f"bad-{i}", rating=2)
        for i in range(10):
            loop.record(f"good-{i}", rating=5)
        report = loop.report()
        assert report.recent_trend == "improving"

    def test_declining_trend(self):
        loop = FeedbackLoop()
        for i in range(10):
            loop.record(f"good-{i}", rating=5)
        for i in range(10):
            loop.record(f"bad-{i}", rating=1)
        report = loop.report()
        assert report.recent_trend == "declining"

    def test_stable_trend(self):
        loop = FeedbackLoop()
        for i in range(10):
            loop.record(f"ok-{i}", rating=3)
        report = loop.report()
        assert report.recent_trend == "stable"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_get_corrections(self):
        loop = FeedbackLoop()
        loop.thumbs_down("a", correction="Fix A")
        loop.thumbs_up("b")
        loop.thumbs_down("c", correction="Fix C")
        corrections = loop.get_corrections()
        assert len(corrections) == 2

    def test_get_by_label(self):
        loop = FeedbackLoop()
        loop.record("a", labels=["bug"])
        loop.record("b", labels=["feature"])
        loop.record("c", labels=["bug"])
        bugs = loop.get_by_label("bug")
        assert len(bugs) == 2

    def test_clear(self):
        loop = FeedbackLoop()
        loop.record("a", rating=5)
        loop.record("b", rating=3)
        cleared = loop.clear()
        assert cleared == 2
        assert loop.report().total == 0


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_entry_structure(self):
        loop = FeedbackLoop()
        entry = loop.record("x", rating=4, labels=["test"], metadata={"key": "val"})
        assert isinstance(entry, FeedbackEntry)
        assert entry.timestamp > 0
        assert entry.metadata["key"] == "val"

    def test_report_structure(self):
        loop = FeedbackLoop()
        loop.record("x", rating=3)
        report = loop.report()
        assert isinstance(report, FeedbackReport)
        assert 0.0 <= report.positive_rate <= 1.0
