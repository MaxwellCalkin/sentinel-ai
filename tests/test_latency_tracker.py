"""Tests for latency tracker."""

import pytest
from sentinel.latency_tracker import LatencyTracker, LatencyReport


class TestRecording:
    def test_basic_record(self):
        t = LatencyTracker()
        t.record("scan", 12.5)
        t.record("scan", 15.3)
        report = t.report("scan")
        assert report.count == 2

    def test_multiple_operations(self):
        t = LatencyTracker()
        t.record("scan", 10.0)
        t.record("api_call", 200.0)
        ops = t.list_operations()
        assert len(ops) == 2

    def test_max_records_eviction(self):
        t = LatencyTracker(max_records=5)
        for i in range(10):
            t.record("op", float(i))
        report = t.report("op")
        assert report.count == 5


class TestStatistics:
    def test_min_max(self):
        t = LatencyTracker()
        for v in [10, 20, 30, 40, 50]:
            t.record("op", float(v))
        report = t.report("op")
        assert report.min_ms == 10.0
        assert report.max_ms == 50.0

    def test_mean(self):
        t = LatencyTracker()
        for v in [10, 20, 30]:
            t.record("op", float(v))
        report = t.report("op")
        assert report.mean_ms == 20.0

    def test_median_odd(self):
        t = LatencyTracker()
        for v in [5, 10, 15]:
            t.record("op", float(v))
        report = t.report("op")
        assert report.median_ms == 10.0

    def test_median_even(self):
        t = LatencyTracker()
        for v in [5, 10, 15, 20]:
            t.record("op", float(v))
        report = t.report("op")
        assert report.median_ms == 12.5

    def test_percentiles(self):
        t = LatencyTracker()
        for i in range(100):
            t.record("op", float(i))
        report = t.report("op")
        assert report.p95 >= 90
        assert report.p99 >= 95

    def test_std_dev(self):
        t = LatencyTracker()
        for v in [10, 10, 10]:
            t.record("op", float(v))
        report = t.report("op")
        assert report.std_dev == 0.0


class TestSLA:
    def test_sla_met(self):
        t = LatencyTracker(sla_p99_ms=100.0)
        for v in [10, 20, 30, 40, 50]:
            t.record("op", float(v))
        report = t.report("op")
        assert report.sla_met

    def test_sla_violated(self):
        t = LatencyTracker(sla_p99_ms=50.0)
        for v in [10, 20, 30, 40, 500]:
            t.record("op", float(v))
        report = t.report("op")
        assert not report.sla_met
        assert report.sla_violations == 1


class TestReportAll:
    def test_report_all(self):
        t = LatencyTracker()
        t.record("a", 10.0)
        t.record("b", 20.0)
        reports = t.report_all()
        assert len(reports) == 2
        assert "a" in reports
        assert "b" in reports


class TestClear:
    def test_clear_specific(self):
        t = LatencyTracker()
        t.record("a", 10.0)
        t.record("b", 20.0)
        cleared = t.clear("a")
        assert cleared == 1
        assert t.report("a") is None
        assert t.report("b") is not None

    def test_clear_all(self):
        t = LatencyTracker()
        t.record("a", 10.0)
        t.record("b", 20.0)
        cleared = t.clear()
        assert cleared == 2
        assert len(t.list_operations()) == 0


class TestEdgeCases:
    def test_unknown_operation(self):
        t = LatencyTracker()
        assert t.report("nonexistent") is None

    def test_single_record(self):
        t = LatencyTracker()
        t.record("op", 42.0)
        report = t.report("op")
        assert report.mean_ms == 42.0
        assert report.median_ms == 42.0
        assert report.p95 == 42.0
        assert report.p99 == 42.0


class TestStructure:
    def test_report_structure(self):
        t = LatencyTracker()
        t.record("op", 10.0)
        report = t.report("op")
        assert isinstance(report, LatencyReport)
        assert report.operation == "op"
        assert isinstance(report.sla_met, bool)
