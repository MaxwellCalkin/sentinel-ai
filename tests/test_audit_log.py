"""Tests for the audit logging system."""

import json
import time
import pytest
from pathlib import Path
from sentinel.audit_log import AuditLog, AuditEvent, AuditStats


# ---------------------------------------------------------------------------
# Recording events
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_event(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan", input_text="Hello"))
        assert log.count == 1

    def test_record_multiple(self):
        log = AuditLog()
        for i in range(5):
            log.record(AuditEvent(action="scan"))
        assert log.count == 5

    def test_max_events(self):
        log = AuditLog(max_events=3)
        for i in range(10):
            log.record(AuditEvent(action="scan", input_text=str(i)))
        assert log.count == 3
        assert log.events[0].input_text == "7"  # oldest remaining

    def test_event_listener(self):
        log = AuditLog()
        received = []
        log.on_event(lambda e: received.append(e))
        log.record(AuditEvent(action="scan"))
        assert len(received) == 1

    def test_multiple_listeners(self):
        log = AuditLog()
        count = [0]
        log.on_event(lambda e: count.__setitem__(0, count[0] + 1))
        log.on_event(lambda e: count.__setitem__(0, count[0] + 1))
        log.record(AuditEvent(action="scan"))
        assert count[0] == 2


# ---------------------------------------------------------------------------
# Querying events
# ---------------------------------------------------------------------------

class TestQuerying:
    def setup_method(self):
        self.log = AuditLog()
        self.log.record(AuditEvent(action="scan", blocked=False, risk_level="none", scanner="pii"))
        self.log.record(AuditEvent(action="scan", blocked=True, risk_level="high", scanner="injection"))
        self.log.record(AuditEvent(action="block", blocked=True, risk_level="critical", scanner="injection"))
        self.log.record(AuditEvent(action="flag", blocked=False, risk_level="medium", scanner="toxicity"))
        self.log.record(AuditEvent(action="scan", blocked=False, user_id="user1", session_id="s1"))

    def test_query_by_action(self):
        results = self.log.query(action="scan")
        assert len(results) == 3

    def test_query_by_blocked(self):
        results = self.log.query(blocked=True)
        assert len(results) == 2

    def test_query_by_risk_level(self):
        results = self.log.query(risk_level="high")
        assert len(results) == 1

    def test_query_by_scanner(self):
        results = self.log.query(scanner="injection")
        assert len(results) == 2

    def test_query_by_user_id(self):
        results = self.log.query(user_id="user1")
        assert len(results) == 1

    def test_query_by_session_id(self):
        results = self.log.query(session_id="s1")
        assert len(results) == 1

    def test_query_combined(self):
        results = self.log.query(action="scan", blocked=True)
        assert len(results) == 1

    def test_query_limit(self):
        results = self.log.query(limit=2)
        assert len(results) == 2

    def test_query_since(self):
        future = time.time() + 1000
        results = self.log.query(since=future)
        assert len(results) == 0

    def test_query_all(self):
        results = self.log.query()
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_empty_stats(self):
        log = AuditLog()
        stats = log.stats()
        assert stats.total_events == 0
        assert stats.block_rate == 0.0

    def test_stats_computation(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan", blocked=False, risk_level="none", duration_ms=1.0))
        log.record(AuditEvent(action="scan", blocked=True, risk_level="high", duration_ms=2.0))
        log.record(AuditEvent(action="flag", blocked=False, risk_level="medium", duration_ms=3.0))

        stats = log.stats()
        assert stats.total_events == 3
        assert stats.total_blocked == 1
        assert stats.total_flagged == 1
        assert stats.by_action["scan"] == 2
        assert stats.by_action["flag"] == 1
        assert stats.by_risk_level["none"] == 1
        assert stats.by_risk_level["high"] == 1
        assert stats.block_rate == pytest.approx(1 / 3)
        assert stats.avg_duration_ms == pytest.approx(2.0)

    def test_stats_by_scanner(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan", scanner="pii"))
        log.record(AuditEvent(action="scan", scanner="pii"))
        log.record(AuditEvent(action="scan", scanner="injection"))

        stats = log.stats()
        assert stats.by_scanner["pii"] == 2
        assert stats.by_scanner["injection"] == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_jsonl(self, tmp_path):
        log = AuditLog()
        log.record(AuditEvent(action="scan", input_text="test1"))
        log.record(AuditEvent(action="block", input_text="test2"))

        path = tmp_path / "audit.jsonl"
        count = log.export_jsonl(path)
        assert count == 2

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        event = json.loads(lines[0])
        assert event["action"] == "scan"
        assert event["input_text"] == "test1"

    def test_export_json(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan"))
        data = log.export_json()
        assert len(data) == 1
        assert data[0]["action"] == "scan"

    def test_export_empty(self, tmp_path):
        log = AuditLog()
        path = tmp_path / "empty.jsonl"
        count = log.export_jsonl(path)
        assert count == 0


# ---------------------------------------------------------------------------
# Compliance report
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def test_report_structure(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan", blocked=True, risk_level="high", scanner="pii", duration_ms=5.0))
        log.record(AuditEvent(action="scan", blocked=False, risk_level="none", scanner="pii", duration_ms=3.0))

        report = log.compliance_report()
        assert "report_generated" in report
        assert report["total_requests"] == 2
        assert report["blocked_requests"] == 1
        assert report["block_rate"] == 0.5
        assert "high" in report["risk_distribution"]
        assert "pii" in report["scanner_activity"]
        assert report["average_latency_ms"] == 4.0

    def test_empty_report(self):
        log = AuditLog()
        report = log.compliance_report()
        assert report["total_requests"] == 0
        assert report["block_rate"] == 0


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------

class TestAuditEvent:
    def test_to_dict(self):
        e = AuditEvent(action="scan", input_text="hello", blocked=True)
        d = e.to_dict()
        assert d["action"] == "scan"
        assert d["input_text"] == "hello"
        assert d["blocked"] is True
        assert "timestamp" in d

    def test_default_timestamp(self):
        before = time.time()
        e = AuditEvent(action="scan")
        after = time.time()
        assert before <= e.timestamp <= after

    def test_metadata(self):
        e = AuditEvent(action="scan", metadata={"model": "claude", "tokens": 100})
        assert e.metadata["tokens"] == 100


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan"))
        log.record(AuditEvent(action="scan"))
        log.clear()
        assert log.count == 0

    def test_events_copy(self):
        log = AuditLog()
        log.record(AuditEvent(action="scan"))
        events = log.events
        events.append(AuditEvent(action="fake"))
        assert log.count == 1  # Original not modified
