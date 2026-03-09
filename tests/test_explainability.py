"""Tests for explainability tracer."""

import pytest
from sentinel.explainability import ExplainabilityTracer, Explanation, TraceRecord, CheckRecord


class TestTracing:
    def test_basic_trace(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("scan_a", passed=True, detail="Clean")
        record = tracer.get_trace("t1")
        assert len(record.checks) == 1
        assert record.decision == "allow"

    def test_failed_trace(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("scan_a", passed=True)
            t.record_check("scan_b", passed=False, detail="Injection found", severity="high")
        record = tracer.get_trace("t1")
        assert record.decision == "block"
        assert not record.all_passed

    def test_flagged_trace(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("scan_a", passed=False, detail="Minor issue", severity="low")
        record = tracer.get_trace("t1")
        assert record.decision == "flag"

    def test_manual_decision(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("scan_a", passed=False, severity="high")
            t.set_decision("allow")  # override
        record = tracer.get_trace("t1")
        assert record.decision == "allow"

    def test_metadata(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1", metadata={"user": "admin"}) as t:
            t.set_metadata("source", "api")
        record = tracer.get_trace("t1")
        assert record.metadata["user"] == "admin"
        assert record.metadata["source"] == "api"


class TestExplanation:
    def test_allow_explanation(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("pii", passed=True, detail="No PII")
            t.record_check("injection", passed=True, detail="Clean")
        exp = tracer.explain("t1")
        assert exp.decision == "allow"
        assert "allowed" in exp.summary
        assert exp.checks_run == 2
        assert exp.checks_passed == 2
        assert exp.severity == "none"

    def test_block_explanation(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("pii", passed=True)
            t.record_check("injection", passed=False, detail="SQL injection", severity="critical")
        exp = tracer.explain("t1")
        assert exp.decision == "block"
        assert "blocked" in exp.summary
        assert exp.checks_failed == 1
        assert exp.severity == "critical"
        assert "injection: SQL injection" in exp.failed_details[0]

    def test_unknown_trace(self):
        tracer = ExplainabilityTracer()
        assert tracer.explain("nonexistent") is None

    def test_duration_tracked(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("a", passed=True)
        exp = tracer.explain("t1")
        assert exp.duration_ms >= 0


class TestListing:
    def test_list_all_traces(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("a", passed=True)
        with tracer.trace("t2") as t:
            t.record_check("b", passed=False, severity="high")
        ids = tracer.list_traces()
        assert len(ids) == 2

    def test_list_by_decision(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("a", passed=True)
        with tracer.trace("t2") as t:
            t.record_check("b", passed=False, severity="high")
        blocked = tracer.list_traces(decision="block")
        assert len(blocked) == 1
        assert "t2" in blocked

    def test_clear(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("a", passed=True)
        cleared = tracer.clear()
        assert cleared == 1
        assert len(tracer.list_traces()) == 0


class TestEviction:
    def test_max_traces_eviction(self):
        tracer = ExplainabilityTracer(max_traces=3)
        for i in range(5):
            with tracer.trace(f"t{i}") as t:
                t.record_check("a", passed=True)
        ids = tracer.list_traces()
        assert len(ids) <= 3


class TestStructure:
    def test_explanation_structure(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("a", passed=True)
        exp = tracer.explain("t1")
        assert isinstance(exp, Explanation)
        assert isinstance(exp.failed_details, list)
        assert isinstance(exp.duration_ms, float)

    def test_trace_record_structure(self):
        tracer = ExplainabilityTracer()
        with tracer.trace("t1") as t:
            t.record_check("a", passed=True, metadata={"key": "val"})
        record = tracer.get_trace("t1")
        assert isinstance(record, TraceRecord)
        assert isinstance(record.checks[0], CheckRecord)
        assert record.checks[0].metadata["key"] == "val"
