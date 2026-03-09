"""Tests for alert management."""

import time
from unittest.mock import patch

import pytest
from sentinel.alert_manager import AlertManager, Alert, AlertStats


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

class TestCreate:
    def test_create_alert(self):
        manager = AlertManager()
        alert = manager.create_alert("Injection detected", "critical", source="scanner")
        assert isinstance(alert, Alert)
        assert alert.title == "Injection detected"
        assert alert.severity == "critical"
        assert alert.source == "scanner"
        assert alert.status == "active"

    def test_invalid_severity(self):
        manager = AlertManager()
        with pytest.raises(ValueError, match="Invalid severity"):
            manager.create_alert("Bad alert", "extreme")

    def test_alert_id_generated(self):
        manager = AlertManager()
        ids = {manager.create_alert("test", "info").id for _ in range(20)}
        assert len(ids) == 20
        for alert_id in ids:
            assert len(alert_id) == 12


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_acknowledge(self):
        manager = AlertManager()
        alert = manager.create_alert("test", "warning")
        manager.acknowledge(alert.id, by="ops-team")
        assert alert.status == "acknowledged"
        assert alert.acknowledged_at is not None
        assert alert.acknowledged_by == "ops-team"

    def test_resolve(self):
        manager = AlertManager()
        alert = manager.create_alert("test", "error")
        manager.resolve(alert.id, resolution="False positive")
        assert alert.status == "resolved"
        assert alert.resolved_at is not None
        assert alert.resolution == "False positive"

    def test_escalate(self):
        manager = AlertManager()
        alert = manager.create_alert("test", "info")
        assert alert.severity == "info"

        manager.escalate(alert.id)
        assert alert.severity == "warning"

        manager.escalate(alert.id)
        assert alert.severity == "error"

        manager.escalate(alert.id)
        assert alert.severity == "critical"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilter:
    def test_get_active(self):
        manager = AlertManager()
        alert_a = manager.create_alert("a", "info")
        manager.create_alert("b", "warning")
        manager.resolve(alert_a.id)
        active = manager.get_active()
        assert len(active) == 1
        assert active[0].title == "b"

    def test_get_by_severity(self):
        manager = AlertManager()
        manager.create_alert("low", "info")
        manager.create_alert("high", "critical")
        manager.create_alert("also high", "critical")
        results = manager.get_by_severity("critical")
        assert len(results) == 2
        assert all(a.severity == "critical" for a in results)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

class TestChannels:
    def test_channel_called(self):
        manager = AlertManager()
        received = []
        manager.add_channel("log", lambda alert: received.append(alert))
        alert = manager.create_alert("test", "warning")
        assert len(received) == 1
        assert received[0] is alert

    def test_channel_not_called_when_suppressed(self):
        manager = AlertManager()
        received = []
        manager.add_channel("log", lambda alert: received.append(alert))
        manager.suppress("scanner", duration_seconds=60)
        manager.create_alert("test", "warning", source="scanner")
        assert len(received) == 0


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

class TestSuppress:
    def test_suppress_source(self):
        manager = AlertManager()
        received = []
        manager.add_channel("log", lambda alert: received.append(alert))
        manager.suppress("noisy-scanner", duration_seconds=60)

        manager.create_alert("suppressed", "info", source="noisy-scanner")
        manager.create_alert("not suppressed", "info", source="other-scanner")
        assert len(received) == 1
        assert received[0].title == "not suppressed"

    def test_suppression_expires(self):
        manager = AlertManager()
        received = []
        manager.add_channel("log", lambda alert: received.append(alert))

        manager.suppress("scanner", duration_seconds=1)

        with patch("sentinel.alert_manager.time") as mock_time:
            # First call: still within suppression window
            mock_time.time.return_value = time.time() + 0.5
            manager.create_alert("still suppressed", "info", source="scanner")
            assert len(received) == 0

            # Second call: suppression has expired
            mock_time.time.return_value = time.time() + 2.0
            manager.create_alert("now allowed", "info", source="scanner")
            assert len(received) == 1
            assert received[0].title == "now allowed"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_basic(self):
        manager = AlertManager()
        manager.create_alert("a", "info")
        manager.create_alert("b", "warning")
        alert_c = manager.create_alert("c", "critical")
        manager.acknowledge(alert_c.id)

        result = manager.stats()
        assert isinstance(result, AlertStats)
        assert result.total == 3
        assert result.active == 2
        assert result.acknowledged == 1
        assert result.resolved == 0
        assert result.by_severity["info"] == 1
        assert result.by_severity["warning"] == 1
        assert result.by_severity["critical"] == 1

    def test_avg_resolution_time(self):
        manager = AlertManager()
        alert = manager.create_alert("test", "error")
        manager.resolve(alert.id)
        result = manager.stats()
        assert result.avg_resolution_time is not None
        assert result.avg_resolution_time >= 0

    def test_avg_resolution_time_none_when_no_resolved(self):
        manager = AlertManager()
        manager.create_alert("test", "info")
        result = manager.stats()
        assert result.avg_resolution_time is None


# ---------------------------------------------------------------------------
# Clear resolved
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resolved(self):
        manager = AlertManager()
        alert_a = manager.create_alert("a", "info")
        manager.create_alert("b", "warning")
        manager.resolve(alert_a.id)

        manager.clear_resolved()
        assert len(manager.get_active()) == 1
        assert manager.stats().total == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_missing_alert_raises(self):
        manager = AlertManager()
        with pytest.raises(KeyError, match="not found"):
            manager.acknowledge("nonexistent")
        with pytest.raises(KeyError, match="not found"):
            manager.resolve("nonexistent")
        with pytest.raises(KeyError, match="not found"):
            manager.escalate("nonexistent")

    def test_escalate_critical_stays(self):
        manager = AlertManager()
        alert = manager.create_alert("top", "critical")
        manager.escalate(alert.id)
        assert alert.severity == "critical"
