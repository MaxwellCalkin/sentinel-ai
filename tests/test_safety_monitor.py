"""Tests for real-time safety metric monitoring."""

import time

import pytest
from sentinel.safety_monitor import (
    SafetyMonitor,
    AlertRule,
    Alert,
    MetricPoint,
    MonitorReport,
    MonitorStats,
)


# ---------------------------------------------------------------------------
# Recording Metrics
# ---------------------------------------------------------------------------

class TestRecordMetrics:
    def test_record_single_point(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.5)
        points = monitor.get_metric("toxicity")
        assert len(points) == 1
        assert points[0].value == 0.5
        assert points[0].name == "toxicity"

    def test_record_with_tags(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.7, tags={"model": "gpt-4"})
        points = monitor.get_metric("toxicity")
        assert points[0].tags == {"model": "gpt-4"}

    def test_record_multiple_points(self):
        monitor = SafetyMonitor()
        for i in range(5):
            monitor.record("latency", float(i))
        points = monitor.get_metric("latency")
        assert len(points) == 5
        assert [p.value for p in points] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_record_assigns_timestamp(self):
        before = time.time()
        monitor = SafetyMonitor()
        monitor.record("score", 1.0)
        after = time.time()
        point = monitor.get_metric("score")[0]
        assert before <= point.timestamp <= after


# ---------------------------------------------------------------------------
# Get Metric History
# ---------------------------------------------------------------------------

class TestGetMetric:
    def test_get_metric_returns_only_matching_name(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.3)
        monitor.record("latency", 100.0)
        monitor.record("toxicity", 0.5)
        assert len(monitor.get_metric("toxicity")) == 2
        assert len(monitor.get_metric("latency")) == 1

    def test_get_metric_last_n(self):
        monitor = SafetyMonitor()
        for i in range(10):
            monitor.record("score", float(i))
        recent = monitor.get_metric("score", last_n=3)
        assert len(recent) == 3
        assert [p.value for p in recent] == [7.0, 8.0, 9.0]

    def test_get_metric_empty(self):
        monitor = SafetyMonitor()
        assert monitor.get_metric("nonexistent") == []


# ---------------------------------------------------------------------------
# Alert Rules: Above Threshold
# ---------------------------------------------------------------------------

class TestAlertAbove:
    def test_alert_triggered_above_threshold(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="high_toxicity",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            window_seconds=60.0,
        ))
        monitor.record("toxicity", 0.9)
        alerts = monitor.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].rule_name == "high_toxicity"
        assert alerts[0].current_value > 0.5

    def test_no_alert_when_below_threshold(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="high_toxicity",
            metric="toxicity",
            condition="above",
            threshold=0.8,
        ))
        monitor.record("toxicity", 0.3)
        assert monitor.check_alerts() == []


# ---------------------------------------------------------------------------
# Alert Rules: Below Threshold
# ---------------------------------------------------------------------------

class TestAlertBelow:
    def test_alert_triggered_below_threshold(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="low_accuracy",
            metric="accuracy",
            condition="below",
            threshold=0.9,
            severity="critical",
        ))
        monitor.record("accuracy", 0.6)
        alerts = monitor.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].current_value < 0.9

    def test_no_alert_when_above_threshold(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="low_accuracy",
            metric="accuracy",
            condition="below",
            threshold=0.5,
        ))
        monitor.record("accuracy", 0.95)
        assert monitor.check_alerts() == []


# ---------------------------------------------------------------------------
# Alert Rules: Equals Condition
# ---------------------------------------------------------------------------

class TestAlertEquals:
    def test_alert_triggered_on_equals(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="exact_match",
            metric="score",
            condition="equals",
            threshold=1.0,
        ))
        monitor.record("score", 1.0)
        alerts = monitor.check_alerts()
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Time Window Filtering
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_only_points_within_window_considered(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="recent_toxicity",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            window_seconds=10.0,
        ))
        # Record an old point outside the window
        old_point = MetricPoint(
            name="toxicity",
            value=0.9,
            timestamp=time.time() - 20.0,
        )
        monitor._points.append(old_point)
        # Record a recent point below threshold
        monitor.record("toxicity", 0.3)
        alerts = monitor.check_alerts()
        # The old high-value point is outside the window, so the average
        # of in-window points (0.3) is below 0.5 => no alert
        assert len(alerts) == 0

    def test_points_inside_window_trigger_alert(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="recent_toxicity",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            window_seconds=60.0,
        ))
        monitor.record("toxicity", 0.9)
        alerts = monitor.check_alerts()
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Minimum Samples
# ---------------------------------------------------------------------------

class TestMinSamples:
    def test_no_alert_below_min_samples(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="needs_samples",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            min_samples=3,
        ))
        monitor.record("toxicity", 0.9)
        monitor.record("toxicity", 0.8)
        # Only 2 samples, need 3
        assert monitor.check_alerts() == []

    def test_alert_when_min_samples_reached(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="needs_samples",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            min_samples=3,
        ))
        monitor.record("toxicity", 0.9)
        monitor.record("toxicity", 0.8)
        monitor.record("toxicity", 0.7)
        alerts = monitor.check_alerts()
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Multiple Metrics
# ---------------------------------------------------------------------------

class TestMultipleMetrics:
    def test_track_multiple_metrics(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.1)
        monitor.record("latency", 50.0)
        monitor.record("accuracy", 0.95)
        assert len(monitor.get_metric("toxicity")) == 1
        assert len(monitor.get_metric("latency")) == 1
        assert len(monitor.get_metric("accuracy")) == 1

    def test_multiple_alerts_at_once(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="high_toxicity",
            metric="toxicity",
            condition="above",
            threshold=0.5,
        ))
        monitor.add_rule(AlertRule(
            name="low_accuracy",
            metric="accuracy",
            condition="below",
            threshold=0.9,
            severity="critical",
        ))
        monitor.record("toxicity", 0.9)
        monitor.record("accuracy", 0.3)
        alerts = monitor.check_alerts()
        assert len(alerts) == 2
        rule_names = {a.rule_name for a in alerts}
        assert rule_names == {"high_toxicity", "low_accuracy"}


# ---------------------------------------------------------------------------
# Remove Rule
# ---------------------------------------------------------------------------

class TestRemoveRule:
    def test_remove_existing_rule(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="test_rule",
            metric="toxicity",
            condition="above",
            threshold=0.5,
        ))
        monitor.remove_rule("test_rule")
        monitor.record("toxicity", 0.9)
        assert monitor.check_alerts() == []

    def test_remove_nonexistent_rule_raises_key_error(self):
        monitor = SafetyMonitor()
        with pytest.raises(KeyError, match="Rule not found"):
            monitor.remove_rule("nonexistent")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.1)
        monitor.record("latency", 50.0)
        report = monitor.report()
        assert isinstance(report, MonitorReport)
        assert report.metrics_tracked == 2
        assert report.total_points == 2
        assert report.rules_count == 0
        assert report.health == "healthy"

    def test_report_includes_active_alerts(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="high_toxicity",
            metric="toxicity",
            condition="above",
            threshold=0.5,
        ))
        monitor.record("toxicity", 0.9)
        report = monitor.report()
        assert len(report.active_alerts) == 1


# ---------------------------------------------------------------------------
# Health Status
# ---------------------------------------------------------------------------

class TestHealthStatus:
    def test_healthy_when_no_alerts(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.1)
        assert monitor.report().health == "healthy"

    def test_degraded_when_warning_alert(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="warn_rule",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            severity="warning",
        ))
        monitor.record("toxicity", 0.9)
        assert monitor.report().health == "degraded"

    def test_critical_when_critical_alert(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="crit_rule",
            metric="accuracy",
            condition="below",
            threshold=0.9,
            severity="critical",
        ))
        monitor.record("accuracy", 0.2)
        assert monitor.report().health == "critical"

    def test_critical_overrides_warning(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="warn_rule",
            metric="toxicity",
            condition="above",
            threshold=0.5,
            severity="warning",
        ))
        monitor.add_rule(AlertRule(
            name="crit_rule",
            metric="accuracy",
            condition="below",
            threshold=0.9,
            severity="critical",
        ))
        monitor.record("toxicity", 0.9)
        monitor.record("accuracy", 0.2)
        assert monitor.report().health == "critical"


# ---------------------------------------------------------------------------
# Stats Tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_total_points(self):
        monitor = SafetyMonitor()
        monitor.record("a", 1.0)
        monitor.record("b", 2.0)
        monitor.record("a", 3.0)
        stats = monitor.stats()
        assert stats.total_points == 3

    def test_stats_by_metric(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.1)
        monitor.record("toxicity", 0.2)
        monitor.record("latency", 50.0)
        stats = monitor.stats()
        assert stats.by_metric["toxicity"] == 2
        assert stats.by_metric["latency"] == 1

    def test_stats_total_alerts(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="rule1",
            metric="toxicity",
            condition="above",
            threshold=0.5,
        ))
        monitor.record("toxicity", 0.9)
        monitor.check_alerts()
        monitor.check_alerts()
        stats = monitor.stats()
        assert stats.total_alerts == 2

    def test_stats_returns_copy(self):
        monitor = SafetyMonitor()
        monitor.record("x", 1.0)
        stats1 = monitor.stats()
        monitor.record("x", 2.0)
        stats2 = monitor.stats()
        assert stats1.total_points == 1
        assert stats2.total_points == 2


# ---------------------------------------------------------------------------
# Clear Metric
# ---------------------------------------------------------------------------

class TestClearMetric:
    def test_clear_removes_points(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.5)
        monitor.record("toxicity", 0.6)
        monitor.clear_metric("toxicity")
        assert monitor.get_metric("toxicity") == []

    def test_clear_does_not_affect_other_metrics(self):
        monitor = SafetyMonitor()
        monitor.record("toxicity", 0.5)
        monitor.record("latency", 100.0)
        monitor.clear_metric("toxicity")
        assert len(monitor.get_metric("latency")) == 1


# ---------------------------------------------------------------------------
# Max Points FIFO
# ---------------------------------------------------------------------------

class TestMaxPointsFifo:
    def test_evicts_oldest_when_exceeding_max(self):
        monitor = SafetyMonitor(max_points=5)
        for i in range(8):
            monitor.record("score", float(i))
        all_points = monitor.get_metric("score")
        assert len(all_points) == 5
        # Should retain the last 5: values 3, 4, 5, 6, 7
        assert [p.value for p in all_points] == [3.0, 4.0, 5.0, 6.0, 7.0]

    def test_max_points_preserves_recent(self):
        monitor = SafetyMonitor(max_points=3)
        monitor.record("a", 1.0)
        monitor.record("b", 2.0)
        monitor.record("a", 3.0)
        monitor.record("b", 4.0)
        # Only 3 points remain: the last three recorded
        assert len(monitor.get_metric("a")) == 1
        assert monitor.get_metric("a")[0].value == 3.0


# ---------------------------------------------------------------------------
# Alert Message Content
# ---------------------------------------------------------------------------

class TestAlertMessage:
    def test_alert_message_contains_rule_name(self):
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="toxic_alert",
            metric="toxicity",
            condition="above",
            threshold=0.5,
        ))
        monitor.record("toxicity", 0.9)
        alert = monitor.check_alerts()[0]
        assert "toxic_alert" in alert.message
        assert "toxicity" in alert.message
        assert "above" in alert.message

    def test_alert_has_triggered_timestamp(self):
        before = time.time()
        monitor = SafetyMonitor()
        monitor.add_rule(AlertRule(
            name="rule1",
            metric="score",
            condition="above",
            threshold=0.0,
        ))
        monitor.record("score", 1.0)
        alert = monitor.check_alerts()[0]
        after = time.time()
        assert before <= alert.triggered_at <= after
