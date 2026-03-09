"""Tests for safety dashboard."""

import time
import pytest
from sentinel.safety_dashboard import SafetyDashboard, DashboardSummary, TimeWindow


# ---------------------------------------------------------------------------
# Recording scans
# ---------------------------------------------------------------------------

class TestRecordScans:
    def test_record_scan(self):
        d = SafetyDashboard()
        d.record_scan(blocked=False, risk_score=0.2, scanner="injection")
        summary = d.summary()
        assert summary.total_scans == 1
        assert summary.total_blocked == 0

    def test_record_blocked_scan(self):
        d = SafetyDashboard()
        d.record_scan(blocked=True, risk_score=0.9)
        summary = d.summary()
        assert summary.total_blocked == 1
        assert summary.block_rate == 1.0

    def test_scanner_counts(self):
        d = SafetyDashboard()
        d.record_scan(scanner="injection")
        d.record_scan(scanner="injection")
        d.record_scan(scanner="pii")
        summary = d.summary()
        assert summary.scans_by_scanner["injection"] == 2
        assert summary.scans_by_scanner["pii"] == 1

    def test_avg_risk_score(self):
        d = SafetyDashboard()
        d.record_scan(risk_score=0.2)
        d.record_scan(risk_score=0.8)
        summary = d.summary()
        assert summary.avg_risk_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Recording incidents
# ---------------------------------------------------------------------------

class TestRecordIncidents:
    def test_record_incident(self):
        d = SafetyDashboard()
        d.record_incident(severity="high", category="injection")
        summary = d.summary()
        assert summary.total_incidents == 1

    def test_incidents_by_severity(self):
        d = SafetyDashboard()
        d.record_incident(severity="high")
        d.record_incident(severity="high")
        d.record_incident(severity="low")
        summary = d.summary()
        assert summary.incidents_by_severity["high"] == 2
        assert summary.incidents_by_severity["low"] == 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        d = SafetyDashboard()
        summary = d.summary()
        assert isinstance(summary, DashboardSummary)
        assert summary.total_scans == 0
        assert summary.block_rate == 0.0
        assert summary.avg_risk_score == 0.0

    def test_block_rate(self):
        d = SafetyDashboard()
        d.record_scan(blocked=True)
        d.record_scan(blocked=False)
        d.record_scan(blocked=True)
        d.record_scan(blocked=False)
        assert d.summary().block_rate == 0.5

    def test_uptime(self):
        d = SafetyDashboard()
        summary = d.summary()
        assert summary.uptime_seconds >= 0


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_window_recent(self):
        d = SafetyDashboard()
        d.record_scan(risk_score=0.5)
        d.record_scan(blocked=True, risk_score=0.9)
        w = d.window(seconds=60)
        assert isinstance(w, TimeWindow)
        assert w.scans == 2
        assert w.blocked == 1

    def test_window_empty(self):
        d = SafetyDashboard()
        w = d.window(seconds=60)
        assert w.scans == 0
        assert w.block_rate == 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset(self):
        d = SafetyDashboard()
        d.record_scan()
        d.record_incident()
        d.reset()
        summary = d.summary()
        assert summary.total_scans == 0
        assert summary.total_incidents == 0


# ---------------------------------------------------------------------------
# Max points
# ---------------------------------------------------------------------------

class TestMaxPoints:
    def test_eviction(self):
        d = SafetyDashboard(max_points=5)
        for _ in range(10):
            d.record_scan()
        # Internal list trimmed, but counters are cumulative
        summary = d.summary()
        assert summary.total_scans == 10


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_custom_labels(self):
        d = SafetyDashboard()
        d.record_scan(labels={"user_id": "u1"})
        d.record_incident(labels={"region": "us-east"})
        # Just verifying it doesn't crash
        assert d.summary().total_scans == 1
