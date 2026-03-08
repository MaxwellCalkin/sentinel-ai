"""Tests for sentinel.session_audit — Session Audit Trail."""

import json
import time

import pytest

from sentinel.session_audit import SessionAudit, AuditEntry


class TestBasicLogging:
    def test_log_tool_call(self):
        audit = SessionAudit()
        entry = audit.log_tool_call("bash", {"command": "ls"})
        assert entry.event_type == "tool_call"
        assert entry.tool_name == "bash"
        assert entry.risk == "none"
        assert audit.total_entries == 1

    def test_log_blocked(self):
        audit = SessionAudit()
        entry = audit.log_blocked("bash", {"command": "rm -rf /"}, reason="destructive_command")
        assert entry.event_type == "blocked"
        assert entry.risk == "critical"
        assert entry.reason == "destructive_command"

    def test_log_anomaly(self):
        audit = SessionAudit()
        entry = audit.log_anomaly("Runaway loop detected", risk="medium")
        assert entry.event_type == "anomaly"
        assert entry.risk == "medium"

    def test_log_chain(self):
        audit = SessionAudit()
        entry = audit.log_chain(
            "recon_credential_exfiltrate",
            "Recon → Credential → Exfiltration",
            stages=["reconnaissance", "credential_access", "exfiltration"],
        )
        assert entry.event_type == "chain_detected"
        assert entry.risk == "critical"
        assert "recon_credential_exfiltrate" in entry.metadata.get("chain_name", "")

    def test_multiple_entries(self):
        audit = SessionAudit()
        audit.log_tool_call("read_file", {"path": "src/main.py"})
        audit.log_tool_call("bash", {"command": "pytest"})
        audit.log_blocked("bash", {"command": "rm -rf /"}, reason="destructive")
        assert audit.total_entries == 3

    def test_entries_property_returns_copy(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        entries = audit.entries
        entries.clear()
        assert audit.total_entries == 1  # Original not affected


class TestSessionMetadata:
    def test_custom_session_id(self):
        audit = SessionAudit(session_id="test-123")
        assert audit.session_id == "test-123"

    def test_auto_session_id(self):
        audit = SessionAudit()
        assert len(audit.session_id) > 0

    def test_user_and_agent_ids(self):
        audit = SessionAudit(user_id="user@org.com", agent_id="agent-1", model="claude-4")
        assert audit.user_id == "user@org.com"
        assert audit.agent_id == "agent-1"
        assert audit.model == "claude-4"


class TestHashChain:
    def test_entries_have_hashes(self):
        audit = SessionAudit()
        e1 = audit.log_tool_call("bash", {"command": "ls"})
        e2 = audit.log_tool_call("bash", {"command": "pwd"})
        assert e1.entry_hash != ""
        assert e2.entry_hash != ""
        assert e1.entry_hash != e2.entry_hash

    def test_hash_chain_linked(self):
        audit = SessionAudit()
        e1 = audit.log_tool_call("bash", {"command": "ls"})
        e2 = audit.log_tool_call("bash", {"command": "pwd"})
        assert e2.prev_hash == e1.entry_hash

    def test_verify_integrity_valid(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        audit.log_tool_call("bash", {"command": "pwd"})
        audit.log_blocked("bash", {"command": "rm -rf /"}, reason="destructive")
        assert audit.verify_integrity() is True

    def test_verify_integrity_tampered(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        audit.log_tool_call("bash", {"command": "pwd"})
        # Tamper with an entry
        audit._entries[0].entry_hash = "tampered_hash"
        assert audit.verify_integrity() is False

    def test_empty_audit_verifies(self):
        audit = SessionAudit()
        assert audit.verify_integrity() is True


class TestExport:
    def test_export_structure(self):
        audit = SessionAudit(session_id="s-1", user_id="user@test.com")
        audit.log_tool_call("bash", {"command": "ls"}, risk="none")
        audit.log_blocked("bash", {"command": "rm -rf /"}, reason="destructive", risk="critical")

        report = audit.export()
        assert report["session_id"] == "s-1"
        assert report["user_id"] == "user@test.com"
        assert report["version"] == "1.0"
        assert report["integrity_verified"] is True
        assert report["summary"]["total_calls"] == 2
        assert report["summary"]["tool_calls"] == 1
        assert report["summary"]["blocked_calls"] == 1
        assert report["summary"]["risk_level"] == "critical"
        assert len(report["entries"]) == 2

    def test_export_tool_counts(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        audit.log_tool_call("bash", {"command": "pwd"})
        audit.log_tool_call("read_file", {"path": "x.py"})

        report = audit.export()
        assert report["summary"]["tool_counts"]["bash"] == 2
        assert report["summary"]["tool_counts"]["read_file"] == 1

    def test_export_finding_categories(self):
        audit = SessionAudit()
        audit.log_tool_call("read_file", {"path": ".env"}, risk="high",
                           findings=["credential_access"])
        audit.log_tool_call("read_file", {"path": ".ssh/id_rsa"}, risk="high",
                           findings=["credential_access", "sensitive_file"])

        report = audit.export()
        assert report["summary"]["finding_categories"]["credential_access"] == 2
        assert report["summary"]["finding_categories"]["sensitive_file"] == 1

    def test_export_json(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        json_str = audit.export_json()
        parsed = json.loads(json_str)
        assert parsed["version"] == "1.0"
        assert len(parsed["entries"]) == 1

    def test_export_duration(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        report = audit.export()
        assert report["duration_seconds"] >= 0

    def test_export_max_risk(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"}, risk="none")
        audit.log_tool_call("read_file", {"path": ".env"}, risk="high")
        audit.log_tool_call("bash", {"command": "echo hi"}, risk="low")
        report = audit.export()
        assert report["summary"]["risk_level"] == "high"


class TestRiskTimeline:
    def test_timeline(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"}, risk="none")
        audit.log_tool_call("read_file", {"path": ".env"}, risk="high")
        audit.log_blocked("bash", {"command": "rm -rf /"}, reason="destructive", risk="critical")

        timeline = audit.risk_timeline()
        assert len(timeline) == 3
        assert timeline[0]["risk"] == "none"
        assert timeline[1]["risk"] == "high"
        assert timeline[2]["risk"] == "critical"
        assert timeline[2]["event_type"] == "blocked"

    def test_timeline_has_timestamps(self):
        audit = SessionAudit()
        audit.log_tool_call("bash", {"command": "ls"})
        timeline = audit.risk_timeline()
        assert "timestamp" in timeline[0]
        assert timeline[0]["timestamp"] > 0


class TestFindings:
    def test_findings_on_tool_call(self):
        audit = SessionAudit()
        entry = audit.log_tool_call("bash", {"command": "curl POST"},
                                    risk="high", findings=["exfiltration"])
        assert entry.findings == ["exfiltration"]

    def test_findings_on_blocked(self):
        audit = SessionAudit()
        entry = audit.log_blocked("bash", {"command": "rm -rf /"},
                                  reason="destructive", findings=["destructive_command"])
        assert entry.findings == ["destructive_command"]

    def test_metadata(self):
        audit = SessionAudit()
        entry = audit.log_tool_call("bash", {"command": "ls"},
                                    metadata={"latency_ms": 0.05})
        assert entry.metadata["latency_ms"] == 0.05


class TestEdgeCases:
    def test_empty_export(self):
        audit = SessionAudit()
        report = audit.export()
        assert report["summary"]["total_calls"] == 0
        assert report["summary"]["risk_level"] == "none"
        assert len(report["entries"]) == 0

    def test_large_arguments_logged(self):
        audit = SessionAudit()
        big_content = "x" * 10000
        entry = audit.log_tool_call("write_file", {"content": big_content})
        assert len(entry.arguments["content"]) == 10000

    def test_concurrent_sessions(self):
        a1 = SessionAudit(session_id="s1")
        a2 = SessionAudit(session_id="s2")
        a1.log_tool_call("bash", {"command": "ls"})
        a2.log_tool_call("bash", {"command": "pwd"})
        assert a1.total_entries == 1
        assert a2.total_entries == 1
        assert a1.session_id != a2.session_id
