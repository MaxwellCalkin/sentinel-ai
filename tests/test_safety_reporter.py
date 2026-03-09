"""Tests for safety reporter."""

import time
import pytest
from sentinel.safety_reporter import (
    SafetyReporter,
    SafetyEvent,
    TrendAnalysis,
    ReporterSummary,
)


# ---------------------------------------------------------------------------
# Recording events
# ---------------------------------------------------------------------------

class TestRecordEvents:
    def test_record_returns_safety_event_with_correct_fields(self):
        reporter = SafetyReporter()
        event = reporter.record("info", "injection", "test event")
        assert isinstance(event, SafetyEvent)
        assert event.severity == "info"
        assert event.category == "injection"
        assert event.description == "test event"
        assert event.metadata == {}

    def test_record_stores_metadata(self):
        reporter = SafetyReporter()
        event = reporter.record("warning", "pii", "found PII", metadata={"field": "email"})
        assert event.metadata == {"field": "email"}

    def test_record_increments_total_and_sets_timestamp(self):
        reporter = SafetyReporter()
        before = time.time()
        reporter.record("info", "test", "first")
        reporter.record("warning", "test", "second")
        after = time.time()
        assert reporter.total_events == 2
        events = reporter.events_in_window(seconds=60)
        for event in events:
            assert before <= event.timestamp <= after

    def test_record_invalid_severity_raises(self):
        reporter = SafetyReporter()
        with pytest.raises(ValueError, match="Invalid severity"):
            reporter.record("urgent", "test", "bad severity")


# ---------------------------------------------------------------------------
# Severity and category counts
# ---------------------------------------------------------------------------

class TestCounts:
    def test_count_by_severity(self):
        reporter = SafetyReporter()
        reporter.record("info", "a", "d1")
        reporter.record("info", "a", "d2")
        reporter.record("critical", "b", "d3")
        counts = reporter.count_by_severity()
        assert counts["info"] == 2
        assert counts["critical"] == 1

    def test_count_by_category(self):
        reporter = SafetyReporter()
        reporter.record("info", "injection", "d1")
        reporter.record("warning", "injection", "d2")
        reporter.record("info", "pii", "d3")
        counts = reporter.count_by_category()
        assert counts["injection"] == 2
        assert counts["pii"] == 1

    def test_category_percentages(self):
        reporter = SafetyReporter()
        reporter.record("info", "injection", "d1")
        reporter.record("info", "injection", "d2")
        reporter.record("info", "pii", "d3")
        reporter.record("info", "pii", "d4")
        pct = reporter.category_percentages()
        assert pct["injection"] == 50.0
        assert pct["pii"] == 50.0

    def test_category_percentages_empty(self):
        reporter = SafetyReporter()
        assert reporter.category_percentages() == {}


# ---------------------------------------------------------------------------
# Time-windowed events
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_events_in_window_returns_recent(self):
        reporter = SafetyReporter()
        reporter.record("info", "test", "recent")
        events = reporter.events_in_window(seconds=60)
        assert len(events) == 1

    def test_events_in_window_excludes_old(self):
        reporter = SafetyReporter()
        event = reporter.record("info", "test", "old")
        event.timestamp = time.time() - 7200
        events = reporter.events_in_window(seconds=3600)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

class TestTrendAnalysis:
    def test_stable_when_no_events(self):
        reporter = SafetyReporter()
        trend = reporter.analyze_trend()
        assert isinstance(trend, TrendAnalysis)
        assert trend.direction == "stable"
        assert trend.score_change == 0.0

    def test_degrading_trend(self):
        reporter = SafetyReporter()
        now = time.time()
        early_event = reporter.record("info", "test", "early mild")
        early_event.timestamp = now - 3000

        for _ in range(3):
            reporter.record("critical", "test", "recent spike")

        trend = reporter.analyze_trend(window_seconds=3600)
        assert trend.direction == "degrading"
        assert trend.score_change > 0

    def test_improving_trend(self):
        reporter = SafetyReporter()
        now = time.time()
        for _ in range(3):
            event = reporter.record("critical", "test", "early spike")
            event.timestamp = now - 3000

        reporter.record("info", "test", "mild recent")

        trend = reporter.analyze_trend(window_seconds=3600)
        assert trend.direction == "improving"
        assert trend.score_change < 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        reporter = SafetyReporter()
        summary = reporter.summary()
        assert isinstance(summary, ReporterSummary)
        assert summary.total_events == 0
        assert summary.by_severity == {}
        assert summary.by_category == {}
        assert summary.escalation_needed is False

    def test_summary_includes_all_fields(self):
        reporter = SafetyReporter()
        reporter.record("warning", "injection", "test")
        summary = reporter.summary()
        assert summary.total_events == 1
        assert summary.by_severity == {"warning": 1}
        assert summary.by_category == {"injection": 1}
        assert isinstance(summary.trend, TrendAnalysis)

    def test_summary_with_window(self):
        reporter = SafetyReporter()
        old_event = reporter.record("info", "test", "old")
        old_event.timestamp = time.time() - 7200
        reporter.record("warning", "test", "recent")
        summary = reporter.summary(window_seconds=3600)
        assert summary.total_events == 1
        assert summary.by_severity == {"warning": 1}


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def test_no_escalation_below_threshold(self):
        reporter = SafetyReporter(escalation_threshold=3)
        reporter.record("critical", "test", "one")
        reporter.record("critical", "test", "two")
        assert reporter.summary().escalation_needed is False

    def test_escalation_at_and_above_threshold(self):
        reporter = SafetyReporter(escalation_threshold=3)
        for _ in range(3):
            reporter.record("critical", "test", "spike")
        assert reporter.summary().escalation_needed is True

    def test_custom_escalation_threshold(self):
        reporter = SafetyReporter(escalation_threshold=1)
        reporter.record("critical", "test", "single critical")
        assert reporter.summary().escalation_needed is True


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_returns_complete_dict(self):
        reporter = SafetyReporter()
        reporter.record("warning", "pii", "found SSN", metadata={"type": "ssn"})
        result = reporter.export()
        assert isinstance(result, dict)
        assert result["total_events"] == 1
        assert "by_severity" in result
        assert "by_category" in result
        assert "category_percentages" in result
        assert "trend" in result
        assert "escalation_needed" in result
        event_dict = result["events"][0]
        assert event_dict["severity"] == "warning"
        assert event_dict["category"] == "pii"
        assert event_dict["description"] == "found SSN"
        assert event_dict["metadata"] == {"type": "ssn"}
        assert "timestamp" in event_dict


# ---------------------------------------------------------------------------
# Max events eviction and clear
# ---------------------------------------------------------------------------

class TestEvictionAndClear:
    def test_evicts_oldest_when_full(self):
        reporter = SafetyReporter(max_events=5)
        for i in range(10):
            reporter.record("info", "test", f"event {i}")
        assert reporter.total_events == 5

    def test_clear_removes_all_events(self):
        reporter = SafetyReporter()
        reporter.record("info", "test", "a")
        reporter.record("warning", "test", "b")
        reporter.clear()
        assert reporter.total_events == 0
        assert reporter.summary().total_events == 0
