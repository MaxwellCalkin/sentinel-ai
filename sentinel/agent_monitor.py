"""Agent Safety Monitor — behavioral anomaly detection for agentic AI sessions.

Tracks tool-use patterns across an agentic session and detects anomalous
behavior: sudden spikes in destructive operations, privilege escalation
sequences, data exfiltration patterns, and runaway loops.

Designed for monitoring Claude Code, Claude Agent SDK, LangChain agents,
and any tool-calling LLM workflow.

Usage:
    from sentinel.agent_monitor import AgentMonitor

    monitor = AgentMonitor()

    # On each tool call
    verdict = monitor.record("bash", {"command": "ls src/"})
    verdict = monitor.record("write_file", {"path": "app.py", "content": "..."})
    verdict = monitor.record("bash", {"command": "rm -rf /"})
    # verdict.alert == True, verdict.anomalies contains detected issues

    # Check session summary
    summary = monitor.summarize()
    print(summary.total_calls)
    print(summary.risk_level)
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from sentinel.core import RiskLevel


@dataclass
class ToolCall:
    """Record of a single tool call in an agentic session."""

    tool_name: str
    arguments: dict
    timestamp: float
    risk: RiskLevel = RiskLevel.NONE
    anomalies: list[str] = field(default_factory=list)


@dataclass
class MonitorVerdict:
    """Result of evaluating a tool call against session patterns."""

    tool_name: str
    risk: RiskLevel
    alert: bool = False
    anomalies: list[Anomaly] = field(default_factory=list)
    session_risk: RiskLevel = RiskLevel.NONE
    total_calls: int = 0


@dataclass
class Anomaly:
    """A detected behavioral anomaly."""

    category: str
    description: str
    risk: RiskLevel
    evidence: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    """Summary of an agentic session's safety profile."""

    total_calls: int
    unique_tools: int
    risk_level: RiskLevel
    anomaly_count: int
    anomalies: list[Anomaly]
    tool_counts: dict[str, int]
    duration_seconds: float
    calls_per_minute: float


# Dangerous command patterns
_DESTRUCTIVE_PATTERNS = [
    re.compile(r'\brm\s+(-[a-zA-Z]*[rf]|--force|--recursive)', re.I),
    re.compile(r'\bgit\s+push\s+.*--force', re.I),
    re.compile(r'\bgit\s+reset\s+--hard', re.I),
    re.compile(r'\bdd\s+.*of=', re.I),
    re.compile(r'\bmkfs\.', re.I),
    re.compile(r'\b(chmod|chown)\s+.*-R\s+/', re.I),
    re.compile(r'--no-verify\b', re.I),
    re.compile(r'\bdrop\s+(table|database)\b', re.I),
    re.compile(r'\btruncate\s+table\b', re.I),
]

_EXFILTRATION_PATTERNS = [
    re.compile(r'\bcurl\s.*\b(POST|PUT)\b.*(-d|--data)', re.I),
    re.compile(r'\bwget\s.*--post', re.I),
    re.compile(r'\bnc\s+-', re.I),
    re.compile(r'>\s*/dev/tcp/', re.I),
    re.compile(r'\bscp\s+.*@', re.I),
]

_CREDENTIAL_PATHS = [
    re.compile(r'\.env\b'),
    re.compile(r'\.ssh/'),
    re.compile(r'\.aws/credentials'),
    re.compile(r'\.gitconfig'),
    re.compile(r'/etc/(passwd|shadow|sudoers)'),
    re.compile(r'\.npmrc'),
    re.compile(r'\.pypirc'),
]

_SENSITIVE_TOOLS = {
    "bash", "shell", "terminal", "execute", "run_command",
    "execute_command", "cmd",
}

_FILE_WRITE_TOOLS = {
    "write_file", "write", "create_file", "edit_file", "edit",
    "patch_file", "append_file",
}

_FILE_READ_TOOLS = {
    "read_file", "read", "cat", "view_file",
}


class AgentMonitor:
    """Monitors tool-use patterns in agentic AI sessions for safety anomalies.

    Detects:
    - Destructive command sequences (rm -rf, git push --force, etc.)
    - Data exfiltration patterns (curl POST, netcat, scp)
    - Credential/secret file access
    - Runaway loops (same tool called repeatedly)
    - Privilege escalation sequences
    - Anomalous spikes in write operations
    - Read-then-exfiltrate patterns

    Args:
        max_repeat_calls: Alert after N identical consecutive tool calls (default: 5).
        write_spike_threshold: Alert when writes/minute exceeds this (default: 20).
        window_seconds: Sliding window for rate calculations (default: 60).
    """

    def __init__(
        self,
        max_repeat_calls: int = 5,
        write_spike_threshold: int = 20,
        window_seconds: int = 60,
    ) -> None:
        self._max_repeat = max_repeat_calls
        self._write_spike = write_spike_threshold
        self._window = window_seconds
        self._calls: list[ToolCall] = []
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._start_time: float | None = None
        self._session_risk = RiskLevel.NONE
        self._recent_reads: list[str] = []  # Track recently read files

    def record(self, tool_name: str, arguments: dict | None = None) -> MonitorVerdict:
        """Record a tool call and check for anomalies.

        Args:
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.

        Returns:
            MonitorVerdict with any detected anomalies.
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        args = arguments or {}
        call = ToolCall(tool_name=tool_name, arguments=args, timestamp=now)
        self._calls.append(call)
        self._tool_counts[tool_name] += 1

        anomalies: list[Anomaly] = []

        # Check all detectors
        anomalies.extend(self._check_destructive(tool_name, args))
        anomalies.extend(self._check_exfiltration(tool_name, args))
        anomalies.extend(self._check_credential_access(tool_name, args))
        anomalies.extend(self._check_runaway_loop(tool_name))
        anomalies.extend(self._check_write_spike(now))
        anomalies.extend(self._check_read_exfiltrate(tool_name, args))

        # Track file reads for read-then-exfiltrate detection
        if tool_name.lower() in _FILE_READ_TOOLS:
            path = args.get("path", args.get("file_path", ""))
            if path:
                self._recent_reads.append(path)
                if len(self._recent_reads) > 50:
                    self._recent_reads = self._recent_reads[-50:]

        # Calculate risk
        call_risk = RiskLevel.NONE
        if anomalies:
            call_risk = max(a.risk for a in anomalies)
        call.risk = call_risk
        call.anomalies = [a.category for a in anomalies]

        if call_risk > self._session_risk:
            self._session_risk = call_risk

        alert = call_risk >= RiskLevel.HIGH

        return MonitorVerdict(
            tool_name=tool_name,
            risk=call_risk,
            alert=alert,
            anomalies=anomalies,
            session_risk=self._session_risk,
            total_calls=len(self._calls),
        )

    def summarize(self) -> SessionSummary:
        """Generate a summary of the current session's safety profile."""
        now = time.time()
        duration = (now - self._start_time) if self._start_time else 0.0
        cpm = (len(self._calls) / (duration / 60.0)) if duration > 0 else 0.0

        all_anomalies: list[Anomaly] = []
        for call in self._calls:
            for cat in call.anomalies:
                all_anomalies.append(Anomaly(
                    category=cat,
                    description=f"Detected in {call.tool_name}",
                    risk=call.risk,
                ))

        return SessionSummary(
            total_calls=len(self._calls),
            unique_tools=len(self._tool_counts),
            risk_level=self._session_risk,
            anomaly_count=sum(len(c.anomalies) for c in self._calls),
            anomalies=all_anomalies,
            tool_counts=dict(self._tool_counts),
            duration_seconds=round(duration, 2),
            calls_per_minute=round(cpm, 2),
        )

    def reset(self) -> None:
        """Reset all tracked state."""
        self._calls.clear()
        self._tool_counts.clear()
        self._start_time = None
        self._session_risk = RiskLevel.NONE
        self._recent_reads.clear()

    def _get_command_text(self, tool_name: str, args: dict) -> str:
        """Extract the command/content string from tool arguments."""
        if tool_name.lower() in _SENSITIVE_TOOLS:
            return str(args.get("command", args.get("cmd", "")))
        return str(args.get("content", args.get("code", "")))

    def _check_destructive(self, tool_name: str, args: dict) -> list[Anomaly]:
        text = self._get_command_text(tool_name, args)
        if not text:
            return []

        results = []
        for pattern in _DESTRUCTIVE_PATTERNS:
            if pattern.search(text):
                results.append(Anomaly(
                    category="destructive_command",
                    description=f"Destructive command pattern detected: {pattern.pattern}",
                    risk=RiskLevel.CRITICAL,
                    evidence=[text[:200]],
                ))
        return results

    def _check_exfiltration(self, tool_name: str, args: dict) -> list[Anomaly]:
        text = self._get_command_text(tool_name, args)
        if not text:
            return []

        results = []
        for pattern in _EXFILTRATION_PATTERNS:
            if pattern.search(text):
                results.append(Anomaly(
                    category="data_exfiltration",
                    description=f"Potential data exfiltration: {pattern.pattern}",
                    risk=RiskLevel.CRITICAL,
                    evidence=[text[:200]],
                ))
        return results

    def _check_credential_access(self, tool_name: str, args: dict) -> list[Anomaly]:
        path = args.get("path", args.get("file_path", args.get("filename", "")))
        if not path:
            # Also check command text for credential paths
            text = self._get_command_text(tool_name, args)
            if not text:
                return []
            path = text

        results = []
        for pattern in _CREDENTIAL_PATHS:
            if pattern.search(path):
                results.append(Anomaly(
                    category="credential_access",
                    description=f"Access to credential/sensitive file: {path}",
                    risk=RiskLevel.HIGH,
                    evidence=[path[:200]],
                ))
        return results

    def _check_runaway_loop(self, tool_name: str) -> list[Anomaly]:
        if len(self._calls) < self._max_repeat:
            return []

        recent = self._calls[-self._max_repeat:]
        if all(c.tool_name == tool_name for c in recent):
            return [Anomaly(
                category="runaway_loop",
                description=(
                    f"Tool '{tool_name}' called {self._max_repeat} times "
                    f"consecutively — possible runaway loop"
                ),
                risk=RiskLevel.MEDIUM,
                evidence=[f"{self._max_repeat} consecutive {tool_name} calls"],
            )]
        return []

    def _check_write_spike(self, now: float) -> list[Anomaly]:
        cutoff = now - self._window
        recent_writes = [
            c for c in self._calls
            if c.timestamp > cutoff
            and c.tool_name.lower() in _FILE_WRITE_TOOLS
        ]

        if len(recent_writes) >= self._write_spike:
            return [Anomaly(
                category="write_spike",
                description=(
                    f"{len(recent_writes)} file writes in {self._window}s "
                    f"(threshold: {self._write_spike})"
                ),
                risk=RiskLevel.HIGH,
                evidence=[f"{len(recent_writes)} writes in window"],
            )]
        return []

    def _check_read_exfiltrate(self, tool_name: str, args: dict) -> list[Anomaly]:
        """Detect read-then-exfiltrate pattern: reading sensitive files then sending data out."""
        if tool_name.lower() not in _SENSITIVE_TOOLS:
            return []

        text = self._get_command_text(tool_name, args)
        if not text:
            return []

        # Check if there's an exfiltration attempt AND we recently read sensitive files
        has_exfil = any(p.search(text) for p in _EXFILTRATION_PATTERNS)
        if not has_exfil:
            return []

        sensitive_reads = [
            r for r in self._recent_reads
            if any(p.search(r) for p in _CREDENTIAL_PATHS)
        ]

        if sensitive_reads:
            return [Anomaly(
                category="read_exfiltrate",
                description=(
                    f"Data exfiltration after reading sensitive files: "
                    f"{', '.join(sensitive_reads[:3])}"
                ),
                risk=RiskLevel.CRITICAL,
                evidence=sensitive_reads[:5],
            )]
        return []
