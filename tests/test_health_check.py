"""Tests for health check."""

import pytest
from sentinel.health_check import HealthCheck, HealthReport, ComponentHealth


class TestRegistration:
    def test_register_component(self):
        h = HealthCheck()
        h.register("scanner", check_fn=lambda: True)
        assert "scanner" in h.list_components()

    def test_unregister(self):
        h = HealthCheck()
        h.register("scanner")
        assert h.unregister("scanner")
        assert "scanner" not in h.list_components()

    def test_unregister_missing(self):
        h = HealthCheck()
        assert not h.unregister("nonexistent")

    def test_list_components(self):
        h = HealthCheck()
        h.register("a")
        h.register("b")
        assert len(h.list_components()) == 2


class TestSingleCheck:
    def test_healthy_component(self):
        h = HealthCheck()
        h.register("scanner", check_fn=lambda: True)
        result = h.check("scanner")
        assert result.healthy
        assert result.latency_ms >= 0

    def test_unhealthy_component(self):
        h = HealthCheck()
        h.register("scanner", check_fn=lambda: False)
        result = h.check("scanner")
        assert not result.healthy

    def test_error_in_check(self):
        def bad_check():
            raise RuntimeError("Connection failed")
        h = HealthCheck()
        h.register("scanner", check_fn=bad_check)
        result = h.check("scanner")
        assert not result.healthy
        assert "RuntimeError" in result.message

    def test_unknown_component(self):
        h = HealthCheck()
        result = h.check("nonexistent")
        assert not result.healthy
        assert "not registered" in result.message


class TestCheckAll:
    def test_all_healthy(self):
        h = HealthCheck()
        h.register("a", check_fn=lambda: True)
        h.register("b", check_fn=lambda: True)
        report = h.check_all()
        assert report.healthy
        assert report.healthy_count == 2
        assert report.unhealthy_count == 0

    def test_some_unhealthy(self):
        h = HealthCheck()
        h.register("a", check_fn=lambda: True)
        h.register("b", check_fn=lambda: False)
        report = h.check_all()
        assert not report.healthy
        assert report.unhealthy_count == 1

    def test_empty_check(self):
        h = HealthCheck()
        report = h.check_all()
        assert report.healthy
        assert report.total == 0


class TestLatency:
    def test_latency_tracked(self):
        h = HealthCheck()
        h.register("a", check_fn=lambda: True)
        report = h.check_all()
        assert report.avg_latency_ms >= 0
        assert report.max_latency_ms >= 0


class TestStructure:
    def test_report_structure(self):
        h = HealthCheck()
        h.register("test", check_fn=lambda: True)
        report = h.check_all()
        assert isinstance(report, HealthReport)
        assert isinstance(report.components, list)
        assert report.timestamp > 0

    def test_component_health_structure(self):
        h = HealthCheck()
        h.register("test", check_fn=lambda: True, metadata={"v": "1.0"})
        result = h.check("test")
        assert isinstance(result, ComponentHealth)
        assert result.metadata["v"] == "1.0"
