"""Session Audit Trail — structured logging for agentic AI sessions.

Creates a tamper-evident audit log of all tool calls, safety verdicts,
and detected anomalies across an agentic session. Supports JSON export
for SIEM integration and compliance reporting (SOC 2, ISO 27001).

Usage:
    from sentinel.session_audit import SessionAudit

    audit = SessionAudit(session_id="session-123", user_id="user@org.com")

    # Log each tool call with its safety verdict
    audit.log_tool_call("bash", {"command": "ls src/"}, risk="none")
    audit.log_tool_call("read_file", {"path": ".env"}, risk="high",
                        findings=["credential_access"])
    audit.log_blocked("bash", {"command": "rm -rf /"}, reason="destructive_command")

    # Export for SIEM / compliance
    report = audit.export()
    print(report["summary"]["total_calls"])     # 3
    print(report["summary"]["blocked_calls"])   # 1
    print(report["summary"]["risk_level"])      # "high"
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditEntry:
    """A single entry in the audit trail."""

    timestamp: float
    entry_id: str
    event_type: str  # "tool_call", "blocked", "anomaly", "chain_detected"
    tool_name: str
    arguments: dict
    risk: str  # "none", "low", "medium", "high", "critical"
    findings: list[str] = field(default_factory=list)
    reason: str = ""
    metadata: dict = field(default_factory=dict)
    # Chain hash for tamper detection
    prev_hash: str = ""
    entry_hash: str = ""


def _hash_entry(entry: AuditEntry, prev_hash: str) -> str:
    """Create a hash chain entry for tamper detection."""
    data = f"{entry.timestamp}:{entry.entry_id}:{entry.event_type}:{entry.tool_name}:{entry.risk}:{prev_hash}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class SessionAudit:
    """Tamper-evident audit trail for agentic AI sessions.

    Logs all tool calls, blocked actions, anomalies, and attack chains
    with hash chaining for integrity verification.
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str = "",
        agent_id: str = "",
        model: str = "",
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.agent_id = agent_id
        self.model = model
        self.start_time = time.time()
        self._entries: list[AuditEntry] = []
        self._prev_hash = "0" * 16  # Genesis hash

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        risk: str = "none",
        findings: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        """Log a tool call that was allowed to execute."""
        return self._add_entry(
            event_type="tool_call",
            tool_name=tool_name,
            arguments=arguments,
            risk=risk,
            findings=findings or [],
            metadata=metadata or {},
        )

    def log_blocked(
        self,
        tool_name: str,
        arguments: dict,
        reason: str,
        risk: str = "critical",
        findings: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        """Log a tool call that was blocked by safety checks."""
        return self._add_entry(
            event_type="blocked",
            tool_name=tool_name,
            arguments=arguments,
            risk=risk,
            reason=reason,
            findings=findings or [],
            metadata=metadata or {},
        )

    def log_anomaly(
        self,
        description: str,
        risk: str = "high",
        tool_name: str = "",
        metadata: dict | None = None,
    ) -> AuditEntry:
        """Log a detected behavioral anomaly."""
        return self._add_entry(
            event_type="anomaly",
            tool_name=tool_name,
            arguments={},
            risk=risk,
            reason=description,
            metadata=metadata or {},
        )

    def log_chain(
        self,
        chain_name: str,
        chain_description: str,
        risk: str = "critical",
        stages: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        """Log a detected attack chain."""
        return self._add_entry(
            event_type="chain_detected",
            tool_name="",
            arguments={},
            risk=risk,
            reason=chain_description,
            findings=stages or [],
            metadata={"chain_name": chain_name, **(metadata or {})},
        )

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def verify_integrity(self) -> bool:
        """Verify the hash chain hasn't been tampered with."""
        prev = "0" * 16
        for entry in self._entries:
            expected = _hash_entry(entry, prev)
            if entry.entry_hash != expected:
                return False
            prev = entry.entry_hash
        return True

    def export(self) -> dict[str, Any]:
        """Export the full audit trail as a structured dict for SIEM integration."""
        now = time.time()

        # Calculate summary stats
        total = len(self._entries)
        blocked = sum(1 for e in self._entries if e.event_type == "blocked")
        anomalies = sum(1 for e in self._entries if e.event_type == "anomaly")
        chains = sum(1 for e in self._entries if e.event_type == "chain_detected")
        tool_calls = sum(1 for e in self._entries if e.event_type == "tool_call")

        risk_order = ["none", "low", "medium", "high", "critical"]
        max_risk = "none"
        for e in self._entries:
            if risk_order.index(e.risk) > risk_order.index(max_risk):
                max_risk = e.risk

        # Tool usage breakdown
        tool_counts: dict[str, int] = {}
        for e in self._entries:
            if e.tool_name:
                tool_counts[e.tool_name] = tool_counts.get(e.tool_name, 0) + 1

        # Finding categories
        finding_counts: dict[str, int] = {}
        for e in self._entries:
            for f in e.findings:
                finding_counts[f] = finding_counts.get(f, 0) + 1

        return {
            "version": "1.0",
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "start_time": self.start_time,
            "end_time": now,
            "duration_seconds": round(now - self.start_time, 2),
            "integrity_verified": self.verify_integrity(),
            "summary": {
                "total_calls": total,
                "tool_calls": tool_calls,
                "blocked_calls": blocked,
                "anomalies_detected": anomalies,
                "chains_detected": chains,
                "risk_level": max_risk,
                "tool_counts": tool_counts,
                "finding_categories": finding_counts,
            },
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "entry_id": e.entry_id,
                    "event_type": e.event_type,
                    "tool_name": e.tool_name,
                    "arguments": e.arguments,
                    "risk": e.risk,
                    "findings": e.findings,
                    "reason": e.reason,
                    "metadata": e.metadata,
                    "entry_hash": e.entry_hash,
                }
                for e in self._entries
            ],
        }

    def export_json(self, indent: int = 2) -> str:
        """Export as formatted JSON string."""
        return json.dumps(self.export(), indent=indent, default=str)

    def risk_timeline(self) -> list[dict[str, Any]]:
        """Get risk level over time for visualization."""
        return [
            {
                "timestamp": e.timestamp,
                "risk": e.risk,
                "event_type": e.event_type,
                "tool_name": e.tool_name,
            }
            for e in self._entries
        ]

    def _add_entry(
        self,
        event_type: str,
        tool_name: str,
        arguments: dict,
        risk: str,
        reason: str = "",
        findings: list[str] | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            entry_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            tool_name=tool_name,
            arguments=arguments,
            risk=risk,
            findings=findings or [],
            reason=reason,
            metadata=metadata or {},
            prev_hash=self._prev_hash,
        )
        entry.entry_hash = _hash_entry(entry, self._prev_hash)
        self._prev_hash = entry.entry_hash
        self._entries.append(entry)
        return entry
