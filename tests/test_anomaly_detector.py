"""Tests for anomaly detector."""

import time
import pytest
from sentinel.anomaly_detector import AnomalyDetector, AnomalyReport, AnomalyFlag


# ---------------------------------------------------------------------------
# Recording events
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_event(self):
        d = AnomalyDetector()
        d.record("user1", tokens=100)
        assert d.user_count == 1

    def test_multiple_users(self):
        d = AnomalyDetector()
        d.record("u1")
        d.record("u2")
        d.record("u3")
        assert d.user_count == 3

    def test_window_trimming(self):
        d = AnomalyDetector(window_size=5)
        for i in range(10):
            d.record("u1", tokens=i)
        report = d.check("u1")
        assert report.stats["events"] == 5


# ---------------------------------------------------------------------------
# Normal behavior
# ---------------------------------------------------------------------------

class TestNormal:
    def test_few_events_not_anomalous(self):
        d = AnomalyDetector()
        d.record("u1")
        d.record("u1")
        report = d.check("u1")
        assert not report.is_anomalous
        assert report.risk_level == "normal"

    def test_steady_rate_not_anomalous(self):
        d = AnomalyDetector()
        for _ in range(10):
            d.record("u1", tokens=100)
        report = d.check("u1")
        # With very close timestamps and constant tokens, no volume spike
        assert report.risk_level in ("normal", "elevated")

    def test_unknown_user(self):
        d = AnomalyDetector()
        report = d.check("nonexistent")
        assert not report.is_anomalous


# ---------------------------------------------------------------------------
# Volume anomaly
# ---------------------------------------------------------------------------

class TestVolumeAnomaly:
    def test_volume_spike(self):
        d = AnomalyDetector(volume_threshold=2.0)
        # Record normal usage
        for _ in range(10):
            d.record("u1", tokens=100)
        # Spike
        d.record("u1", tokens=10000)
        report = d.check("u1")
        volume_flags = [a for a in report.anomalies if a.anomaly_type == "volume_spike"]
        assert len(volume_flags) > 0

    def test_no_volume_spike(self):
        d = AnomalyDetector()
        # Varied baseline so std dev is meaningful
        for t in [90, 110, 95, 105, 100, 98, 102, 97, 103, 100]:
            d.record("u1", tokens=t)
        d.record("u1", tokens=108)  # Within normal range
        report = d.check("u1")
        volume_flags = [a for a in report.anomalies if a.anomaly_type == "volume_spike"]
        assert len(volume_flags) == 0


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        d = AnomalyDetector()
        for _ in range(5):
            d.record("u1", tokens=50)
        report = d.check("u1")
        assert isinstance(report, AnomalyReport)
        assert report.user_id == "u1"
        assert "events" in report.stats

    def test_report_stats(self):
        d = AnomalyDetector()
        for _ in range(5):
            d.record("u1", tokens=100)
        report = d.check("u1")
        assert report.stats["events"] == 5
        assert "avg_tokens" in report.stats


# ---------------------------------------------------------------------------
# Check all
# ---------------------------------------------------------------------------

class TestCheckAll:
    def test_check_all_users(self):
        d = AnomalyDetector()
        for _ in range(5):
            d.record("u1", tokens=50)
            d.record("u2", tokens=50)
        results = d.check_all()
        assert "u1" in results
        assert "u2" in results


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_user(self):
        d = AnomalyDetector()
        d.record("u1")
        d.record("u2")
        d.clear("u1")
        assert d.user_count == 1

    def test_clear_all(self):
        d = AnomalyDetector()
        d.record("u1")
        d.record("u2")
        d.clear()
        assert d.user_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_tokens(self):
        d = AnomalyDetector()
        for _ in range(5):
            d.record("u1", tokens=0)
        report = d.check("u1")
        assert report is not None

    def test_metadata_preserved(self):
        d = AnomalyDetector()
        d.record("u1", metadata={"ip": "1.2.3.4"})
        # Just checking it doesn't crash
        report = d.check("u1")
        assert report is not None
