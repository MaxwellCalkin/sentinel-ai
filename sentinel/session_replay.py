"""Session Replay — forensic analysis of agentic AI session audit trails.

Load an exported audit trail and replay it for analysis: identify attack
patterns, visualize risk escalation, extract IOCs, and generate incident
reports.

Usage:
    from sentinel.session_replay import SessionReplay

    # Load from exported audit JSON
    replay = SessionReplay.from_json(json_string)

    # Or from a file
    replay = SessionReplay.from_file("audit_trail.json")

    # Analyze
    print(replay.risk_summary())
    print(replay.attack_timeline())
    print(replay.iocs())            # Indicators of compromise
    print(replay.incident_report()) # Structured incident report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayEntry:
    """A single entry from a replayed audit trail."""

    timestamp: float
    entry_id: str
    event_type: str
    tool_name: str
    arguments: dict
    risk: str
    findings: list[str]
    reason: str
    metadata: dict
    entry_hash: str


@dataclass
class RiskEscalation:
    """A point where risk level increased."""

    timestamp: float
    from_risk: str
    to_risk: str
    trigger_tool: str
    trigger_event: str
    entry_id: str


@dataclass
class IOC:
    """Indicator of Compromise extracted from the session."""

    ioc_type: str  # "sensitive_file", "exfil_target", "credential_access", "destructive_command"
    value: str
    risk: str
    timestamp: float
    entry_id: str


@dataclass
class IncidentReport:
    """Structured incident report from session replay analysis."""

    session_id: str
    user_id: str
    agent_id: str
    model: str
    duration_seconds: float
    total_events: int
    blocked_events: int
    max_risk: str
    risk_escalations: list[RiskEscalation]
    iocs: list[IOC]
    attack_chains: list[dict]
    timeline: list[dict]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "duration_seconds": self.duration_seconds,
            "total_events": self.total_events,
            "blocked_events": self.blocked_events,
            "max_risk": self.max_risk,
            "risk_escalations": [
                {
                    "timestamp": e.timestamp,
                    "from": e.from_risk,
                    "to": e.to_risk,
                    "trigger_tool": e.trigger_tool,
                    "trigger_event": e.trigger_event,
                }
                for e in self.risk_escalations
            ],
            "iocs": [
                {
                    "type": i.ioc_type,
                    "value": i.value,
                    "risk": i.risk,
                    "timestamp": i.timestamp,
                }
                for i in self.iocs
            ],
            "attack_chains": self.attack_chains,
            "timeline": self.timeline,
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


_RISK_ORDER = ["none", "low", "medium", "high", "critical"]

_SENSITIVE_PATHS = {
    ".env", ".aws/credentials", ".aws/config",
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/config",
    ".npmrc", ".pypirc", "credentials.json",
    "/etc/shadow", "/etc/passwd",
    ".kube/config", ".docker/config.json",
}

_EXFIL_INDICATORS = ["curl", "wget", "scp", "rsync", "nc ", "base64"]
_DESTRUCTIVE_INDICATORS = ["rm -rf", "rm -f", "mkfs", "dd if=", "> /dev/"]


class SessionReplay:
    """Replay and analyze an exported session audit trail."""

    def __init__(self, audit_data: dict[str, Any]):
        self._data = audit_data
        self._entries = [
            ReplayEntry(
                timestamp=e.get("timestamp", 0),
                entry_id=e.get("entry_id", e.get("entryId", "")),
                event_type=e.get("event_type", e.get("eventType", "")),
                tool_name=e.get("tool_name", e.get("toolName", "")),
                arguments=e.get("arguments", {}),
                risk=e.get("risk", "none"),
                findings=e.get("findings", []),
                reason=e.get("reason", ""),
                metadata=e.get("metadata", {}),
                entry_hash=e.get("entry_hash", e.get("entryHash", "")),
            )
            for e in audit_data.get("entries", [])
        ]

    @classmethod
    def from_json(cls, json_string: str) -> SessionReplay:
        return cls(json.loads(json_string))

    @classmethod
    def from_file(cls, path: str) -> SessionReplay:
        with open(path) as f:
            return cls(json.load(f))

    @property
    def session_id(self) -> str:
        return self._data.get("session_id", self._data.get("sessionId", ""))

    @property
    def entries(self) -> list[ReplayEntry]:
        return list(self._entries)

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def risk_summary(self) -> dict[str, Any]:
        """Get a summary of risk levels across the session."""
        risk_counts: dict[str, int] = {}
        for e in self._entries:
            risk_counts[e.risk] = risk_counts.get(e.risk, 0) + 1

        event_counts: dict[str, int] = {}
        for e in self._entries:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        max_risk = "none"
        for e in self._entries:
            if _RISK_ORDER.index(e.risk) > _RISK_ORDER.index(max_risk):
                max_risk = e.risk

        return {
            "max_risk": max_risk,
            "risk_distribution": risk_counts,
            "event_types": event_counts,
            "total_events": len(self._entries),
            "blocked_events": sum(1 for e in self._entries if e.event_type == "blocked"),
        }

    def risk_escalations(self) -> list[RiskEscalation]:
        """Identify points where the risk level increased."""
        escalations: list[RiskEscalation] = []
        prev_risk = "none"

        for e in self._entries:
            if _RISK_ORDER.index(e.risk) > _RISK_ORDER.index(prev_risk):
                escalations.append(RiskEscalation(
                    timestamp=e.timestamp,
                    from_risk=prev_risk,
                    to_risk=e.risk,
                    trigger_tool=e.tool_name,
                    trigger_event=e.event_type,
                    entry_id=e.entry_id,
                ))
            prev_risk = max(prev_risk, e.risk, key=lambda r: _RISK_ORDER.index(r))

        return escalations

    def iocs(self) -> list[IOC]:
        """Extract Indicators of Compromise from the session."""
        indicators: list[IOC] = []
        seen: set[str] = set()

        for e in self._entries:
            text = self._extract_text(e)

            # Sensitive file access
            for path in _SENSITIVE_PATHS:
                if path in text and f"file:{path}" not in seen:
                    seen.add(f"file:{path}")
                    indicators.append(IOC(
                        ioc_type="sensitive_file",
                        value=path,
                        risk=e.risk,
                        timestamp=e.timestamp,
                        entry_id=e.entry_id,
                    ))

            # Exfiltration indicators
            for indicator in _EXFIL_INDICATORS:
                if indicator in text.lower() and f"exfil:{indicator}" not in seen:
                    seen.add(f"exfil:{indicator}")
                    indicators.append(IOC(
                        ioc_type="exfil_target",
                        value=text[:200],
                        risk="critical",
                        timestamp=e.timestamp,
                        entry_id=e.entry_id,
                    ))

            # Destructive commands
            for indicator in _DESTRUCTIVE_INDICATORS:
                if indicator in text.lower() and f"destruct:{indicator}" not in seen:
                    seen.add(f"destruct:{indicator}")
                    indicators.append(IOC(
                        ioc_type="destructive_command",
                        value=text[:200],
                        risk="critical",
                        timestamp=e.timestamp,
                        entry_id=e.entry_id,
                    ))

            # Credential access findings
            if "credential_access" in e.findings and f"cred:{e.entry_id}" not in seen:
                seen.add(f"cred:{e.entry_id}")
                indicators.append(IOC(
                    ioc_type="credential_access",
                    value=text[:200],
                    risk="high",
                    timestamp=e.timestamp,
                    entry_id=e.entry_id,
                ))

        return indicators

    def attack_timeline(self) -> list[dict[str, Any]]:
        """Get a timeline of all security-relevant events."""
        timeline: list[dict[str, Any]] = []
        for e in self._entries:
            if e.risk in ("high", "critical") or e.event_type in ("blocked", "anomaly", "chain_detected"):
                timeline.append({
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "tool_name": e.tool_name,
                    "risk": e.risk,
                    "findings": e.findings,
                    "reason": e.reason,
                    "entry_id": e.entry_id,
                })
        return timeline

    def incident_report(self) -> IncidentReport:
        """Generate a structured incident report."""
        summary = self.risk_summary()
        escalations = self.risk_escalations()
        iocs = self.iocs()
        timeline = self.attack_timeline()

        # Extract active chains if present
        chains = self._data.get("active_chains", [])

        # Generate recommendations
        recommendations: list[str] = []
        if summary["blocked_events"] > 0:
            recommendations.append(
                f"{summary['blocked_events']} tool call(s) were blocked. "
                "Review the blocked entries for attempted attacks."
            )
        if any(e.event_type == "chain_detected" for e in self._entries):
            recommendations.append(
                "Multi-step attack chain(s) detected. Investigate the full "
                "sequence of tool calls for coordinated attack patterns."
            )
        if any(i.ioc_type == "sensitive_file" for i in iocs):
            files = [i.value for i in iocs if i.ioc_type == "sensitive_file"]
            recommendations.append(
                f"Sensitive file(s) accessed: {', '.join(files)}. "
                "Verify these accesses were authorized and rotate credentials if needed."
            )
        if any(i.ioc_type == "exfil_target" for i in iocs):
            recommendations.append(
                "Potential data exfiltration detected. Check network logs for "
                "unauthorized outbound data transfers."
            )
        if summary["max_risk"] == "critical":
            recommendations.append(
                "Session reached CRITICAL risk level. Immediate investigation recommended."
            )
        if len(escalations) >= 3:
            recommendations.append(
                f"Risk escalated {len(escalations)} time(s) during the session, "
                "suggesting progressive attack behavior."
            )
        if not recommendations:
            recommendations.append("No significant security events detected.")

        return IncidentReport(
            session_id=self.session_id,
            user_id=self._data.get("user_id", self._data.get("userId", "")),
            agent_id=self._data.get("agent_id", self._data.get("agentId", "")),
            model=self._data.get("model", ""),
            duration_seconds=self._data.get("duration_seconds", self._data.get("durationSeconds", 0)),
            total_events=summary["total_events"],
            blocked_events=summary["blocked_events"],
            max_risk=summary["max_risk"],
            risk_escalations=escalations,
            iocs=iocs,
            attack_chains=chains,
            timeline=timeline,
            recommendations=recommendations,
        )

    def _extract_text(self, entry: ReplayEntry) -> str:
        parts: list[str] = []
        for key in ("command", "cmd", "path", "file_path", "content"):
            if key in entry.arguments and isinstance(entry.arguments[key], str):
                parts.append(entry.arguments[key])
        if not parts:
            for v in entry.arguments.values():
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts)
