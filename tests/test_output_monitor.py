"""Tests for output monitor."""

import pytest
from sentinel.output_monitor import (
    OutputMonitor,
    MonitorEvent,
    DriftReport,
    AnomalyEvent,
    MonitorReport,
)


# ---------------------------------------------------------------------------
# Recording events
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_event(self):
        monitor = OutputMonitor()
        event = monitor.record("Hello, world!")
        assert isinstance(event, MonitorEvent)
        assert event.output == "Hello, world!"
        assert event.length == len("Hello, world!")
        assert event.safe is True
        assert event.timestamp > 0

    def test_record_with_metadata(self):
        monitor = OutputMonitor()
        event = monitor.record("Response text", metadata={"model": "claude", "request_id": "abc"})
        assert event.metadata["model"] == "claude"
        assert event.metadata["request_id"] == "abc"

    def test_window_overflow(self):
        monitor = OutputMonitor(window_size=5)
        for i in range(10):
            monitor.record(f"output {i}")
        recent = monitor.get_recent(10)
        assert len(recent) == 5
        assert recent[0].output == "output 5"
        assert recent[-1].output == "output 9"


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDrift:
    def test_no_drift(self):
        monitor = OutputMonitor()
        for _ in range(20):
            monitor.record("This is a safe and normal response.")
        drift = monitor.check_drift()
        assert isinstance(drift, DriftReport)
        assert drift.drifting is False
        assert drift.safety_trend == "stable"

    def test_safety_drift_declining(self):
        monitor = OutputMonitor(window_size=20)
        for _ in range(10):
            monitor.record("This is a perfectly safe response.")
        for _ in range(10):
            monitor.record("I will hack and exploit and destroy everything.")
        drift = monitor.check_drift(threshold=0.3)
        assert drift.drifting is True
        assert drift.safety_trend == "declining"
        assert drift.first_half_safety > drift.second_half_safety

    def test_length_drift(self):
        monitor = OutputMonitor(window_size=20)
        for _ in range(10):
            monitor.record("short")
        for _ in range(10):
            monitor.record("a" * 1000)
        drift = monitor.check_drift(threshold=0.3)
        assert drift.drifting is True


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomaly:
    def test_length_anomaly(self):
        monitor = OutputMonitor(window_size=50)
        for _ in range(20):
            monitor.record("Normal length output here.")
        monitor.record("x" * 5000)
        anomalies = monitor.check_anomaly()
        assert len(anomalies) >= 1
        anomaly = anomalies[-1]
        assert isinstance(anomaly, AnomalyEvent)
        assert anomaly.deviation > 2.0
        assert "above" in anomaly.reason

    def test_no_anomaly(self):
        monitor = OutputMonitor()
        for _ in range(20):
            monitor.record("Consistent output length.")
        anomalies = monitor.check_anomaly()
        assert len(anomalies) == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_avg_length(self):
        monitor = OutputMonitor()
        monitor.record("abcde")
        monitor.record("abcdefghij")
        assert monitor.avg_length() == 7.5

    def test_safety_rate(self):
        monitor = OutputMonitor()
        for _ in range(10):
            monitor.record("Perfectly safe output.")
        assert monitor.safety_rate() == 1.0

    def test_safety_rate_with_harmful(self):
        monitor = OutputMonitor()
        for _ in range(8):
            monitor.record("Safe output.")
        monitor.record("I want to hack the system.")
        monitor.record("Let me exploit this vulnerability.")
        assert monitor.safety_rate() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        monitor = OutputMonitor()
        for _ in range(5):
            monitor.record("Normal output.")
        report = monitor.report()
        assert isinstance(report, MonitorReport)
        assert isinstance(report.drift, DriftReport)
        assert hasattr(report, "total_outputs")
        assert hasattr(report, "avg_length")
        assert hasattr(report, "safety_rate")
        assert hasattr(report, "anomaly_count")

    def test_report_values(self):
        monitor = OutputMonitor()
        for _ in range(10):
            monitor.record("Hello world!")
        report = monitor.report()
        assert report.total_outputs == 10
        assert report.avg_length == len("Hello world!")
        assert report.safety_rate == 1.0
        assert report.anomaly_count == 0


# ---------------------------------------------------------------------------
# Recent events
# ---------------------------------------------------------------------------

class TestRecent:
    def test_get_recent(self):
        monitor = OutputMonitor()
        for i in range(5):
            monitor.record(f"output {i}")
        recent = monitor.get_recent()
        assert len(recent) == 5
        assert recent[-1].output == "output 4"

    def test_get_recent_limit(self):
        monitor = OutputMonitor()
        for i in range(20):
            monitor.record(f"output {i}")
        recent = monitor.get_recent(n=3)
        assert len(recent) == 3
        assert recent[0].output == "output 17"
        assert recent[-1].output == "output 19"


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets(self):
        monitor = OutputMonitor()
        for _ in range(10):
            monitor.record("Some output.")
        monitor.clear()
        assert monitor.avg_length() == 0.0
        assert monitor.safety_rate() == 0.0
        assert monitor.get_recent() == []
        report = monitor.report()
        assert report.total_outputs == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_empty_monitor(self):
        monitor = OutputMonitor()
        assert monitor.avg_length() == 0.0
        assert monitor.safety_rate() == 0.0
        assert monitor.get_recent() == []
        assert monitor.check_anomaly() == []
        drift = monitor.check_drift()
        assert drift.drifting is False
        report = monitor.report()
        assert report.total_outputs == 0
        assert report.anomaly_count == 0

    def test_single_event(self):
        monitor = OutputMonitor()
        monitor.record("Only one.")
        assert monitor.avg_length() == len("Only one.")
        assert monitor.safety_rate() == 1.0
        assert monitor.check_anomaly() == []
        drift = monitor.check_drift()
        assert drift.drifting is False
