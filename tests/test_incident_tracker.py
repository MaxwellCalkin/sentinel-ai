"""Tests for incident tracking."""

import pytest
from sentinel.incident_tracker import (
    IncidentTracker, Incident, IncidentSeverity, IncidentState, IncidentSummary,
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_creates_incident(self):
        t = IncidentTracker()
        inc = t.report("injection", description="Detected prompt injection")
        assert isinstance(inc, Incident)
        assert inc.category == "injection"
        assert inc.state == IncidentState.OPEN
        assert inc.id.startswith("INC-")

    def test_report_with_severity(self):
        t = IncidentTracker()
        inc = t.report("pii_leak", severity="critical")
        assert inc.severity == IncidentSeverity.CRITICAL

    def test_report_with_metadata(self):
        t = IncidentTracker()
        inc = t.report("injection", metadata={"user_id": "u1", "prompt": "hack"})
        assert inc.metadata["user_id"] == "u1"

    def test_unique_ids(self):
        t = IncidentTracker()
        ids = {t.report("cat").id for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_get_by_id(self):
        t = IncidentTracker()
        inc = t.report("test")
        assert t.get(inc.id) is inc

    def test_get_nonexistent(self):
        t = IncidentTracker()
        assert t.get("INC-NOTREAL") is None

    def test_list_all(self):
        t = IncidentTracker()
        t.report("a")
        t.report("b")
        assert len(t.list_incidents()) == 2

    def test_list_by_category(self):
        t = IncidentTracker()
        t.report("injection")
        t.report("pii")
        t.report("injection")
        results = t.list_incidents(category="injection")
        assert len(results) == 2

    def test_list_by_severity(self):
        t = IncidentTracker()
        t.report("a", severity="high")
        t.report("b", severity="low")
        results = t.list_incidents(severity="high")
        assert len(results) == 1

    def test_list_by_state(self):
        t = IncidentTracker()
        inc = t.report("a")
        t.report("b")
        t.update_state(inc.id, "resolved")
        open_list = t.list_incidents(state="open")
        assert len(open_list) == 1

    def test_list_limit(self):
        t = IncidentTracker()
        for _ in range(10):
            t.report("test")
        assert len(t.list_incidents(limit=3)) == 3

    def test_open_incidents(self):
        t = IncidentTracker()
        t.report("a")
        inc = t.report("b")
        t.update_state(inc.id, "resolved")
        assert len(t.open_incidents()) == 1


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_update_state(self):
        t = IncidentTracker()
        inc = t.report("test")
        assert t.update_state(inc.id, "investigating")
        assert t.get(inc.id).state == IncidentState.INVESTIGATING

    def test_resolve_with_description(self):
        t = IncidentTracker()
        inc = t.report("test")
        t.update_state(inc.id, "resolved", resolution="False positive")
        resolved = t.get(inc.id)
        assert resolved.state == IncidentState.RESOLVED
        assert resolved.resolution == "False positive"
        assert resolved.resolved_at is not None

    def test_dismiss(self):
        t = IncidentTracker()
        inc = t.report("test")
        t.update_state(inc.id, "dismissed", resolution="Not actionable")
        assert t.get(inc.id).state == IncidentState.DISMISSED

    def test_update_nonexistent(self):
        t = IncidentTracker()
        assert not t.update_state("NOPE", "resolved")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_callback_on_report(self):
        t = IncidentTracker()
        received = []
        t.on_incident(lambda inc: received.append(inc))
        t.report("test")
        assert len(received) == 1
        assert received[0].category == "test"

    def test_callback_error_ignored(self):
        t = IncidentTracker()
        t.on_incident(lambda inc: (_ for _ in ()).throw(RuntimeError("boom")))
        # Should not raise
        inc = t.report("test")
        assert inc is not None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_basic(self):
        t = IncidentTracker()
        t.report("injection", severity="high")
        t.report("pii", severity="medium")
        t.report("injection", severity="high")
        summary = t.summary()
        assert isinstance(summary, IncidentSummary)
        assert summary.total == 3
        assert summary.by_severity["high"] == 2
        assert summary.by_category["injection"] == 2
        assert summary.open_count == 3

    def test_summary_empty(self):
        t = IncidentTracker()
        summary = t.summary()
        assert summary.total == 0
        assert summary.avg_resolution_time is None

    def test_summary_resolution_time(self):
        t = IncidentTracker()
        inc = t.report("test")
        t.update_state(inc.id, "resolved")
        summary = t.summary()
        assert summary.avg_resolution_time is not None
        assert summary.avg_resolution_time >= 0


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_clear_resolved(self):
        t = IncidentTracker()
        inc1 = t.report("a")
        inc2 = t.report("b")
        t.report("c")
        t.update_state(inc1.id, "resolved")
        t.update_state(inc2.id, "dismissed")
        removed = t.clear_resolved()
        assert removed == 2
        assert t.total_count == 1

    def test_clear_resolved_none(self):
        t = IncidentTracker()
        t.report("a")
        assert t.clear_resolved() == 0

    def test_total_count(self):
        t = IncidentTracker()
        assert t.total_count == 0
        t.report("a")
        assert t.total_count == 1
