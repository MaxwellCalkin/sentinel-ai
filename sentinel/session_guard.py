"""Session Guard — unified real-time safety layer for agentic AI sessions.

Combines SessionAudit (tamper-evident logging), AttackChainDetector (multi-step
attack detection), and ThreatFeed (known attack pattern matching) into a single
``guard.check(tool_name, args)`` call that returns allow/block verdicts with
full audit trail.

Usage:
    from sentinel.session_guard import SessionGuard

    guard = SessionGuard(session_id="s-1", user_id="user@org.com")

    verdict = guard.check("bash", {"command": "ls src/"})
    # verdict.allowed == True, verdict.risk == "none"

    verdict = guard.check("bash", {"command": "rm -rf /"})
    # verdict.allowed == False, verdict.risk == "critical"

    verdict = guard.check("read_file", {"path": ".env"})
    # verdict.allowed == True, verdict.risk == "high", verdict.warnings == [...]

    report = guard.export()  # Full audit trail + chain analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sentinel.attack_chain import AttackChainDetector
from sentinel.session_audit import SessionAudit
from sentinel.threat_intel import ThreatFeed, ThreatMatch


# Default block rules: patterns that should be blocked outright
_BLOCK_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    ": (){:|:&};:",  # fork bomb
    "dd if=/dev/zero of=/dev/sda",
    "> /dev/sda",
    "chmod -R 777 /",
]

_BLOCK_PATTERNS = [
    r"rm\s+(-[a-zA-Z]*[rf]){1,2}\s+/\s*$",
    r"rm\s+(-[a-zA-Z]*[rf]){1,2}\s+/\*",
    r"mkfs\.",
    r":\s*\(\s*\)\s*\{",
    r"dd\s+if=/dev/(zero|random)\s+of=/dev/",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+(-R\s+)?777\s+/",
    r"curl\s+.*\|\s*(ba)?sh",
    r"wget\s+.*\|\s*(ba)?sh",
]

_SENSITIVE_PATHS = [
    ".env", ".aws/credentials", ".aws/config",
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/config",
    ".npmrc", ".pypirc", "credentials.json",
    "/etc/shadow", "/etc/passwd",
    ".kube/config", ".docker/config.json",
    ".gitconfig",
]


@dataclass
class GuardVerdict:
    """Result of a safety check on a tool call."""

    allowed: bool
    risk: str  # "none", "low", "medium", "high", "critical"
    tool_name: str
    block_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    threat_matches: list[str] = field(default_factory=list)
    chains_detected: list[str] = field(default_factory=list)
    stage: str = ""  # Attack chain stage if classified

    @property
    def safe(self) -> bool:
        return self.allowed and self.risk in ("none", "low")


import re

_COMPILED_BLOCKS = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]


class SessionGuard:
    """Unified safety guard for agentic AI sessions.

    Combines:
    - SessionAudit: tamper-evident logging of all tool calls
    - AttackChainDetector: multi-step attack sequence detection
    - ThreatFeed: known attack pattern matching

    into a single ``check()`` call.
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str = "",
        agent_id: str = "",
        model: str = "",
        block_on: str = "critical",
        custom_rules: list[Callable[[str, dict], str | None]] | None = None,
        window_seconds: int = 300,
    ):
        self.audit = SessionAudit(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            model=model,
        )
        self.chain_detector = AttackChainDetector(window_seconds=window_seconds)
        self.threat_feed = ThreatFeed.default()
        self._block_threshold = block_on
        self._custom_rules = custom_rules or []
        self._risk_order = ["none", "low", "medium", "high", "critical"]

    @property
    def session_id(self) -> str:
        return self.audit.session_id

    def check(self, tool_name: str, arguments: dict) -> GuardVerdict:
        """Check a tool call for safety. Returns a verdict with allow/block decision."""
        warnings: list[str] = []
        threat_matches: list[str] = []
        risk = "none"
        block_reason = ""
        findings: list[str] = []

        # 1. Check for blocked command patterns
        text = self._extract_text(tool_name, arguments)
        for pattern in _COMPILED_BLOCKS:
            if pattern.search(text):
                block_reason = f"destructive_command: {pattern.pattern}"
                risk = "critical"
                findings.append("destructive_command")
                break

        # 2. Check for sensitive file access
        if not block_reason:
            for path in _SENSITIVE_PATHS:
                if path in text:
                    risk = self._max_risk(risk, "high")
                    findings.append("credential_access")
                    warnings.append(f"Sensitive file access: {path}")
                    break

        # 3. Match against threat intelligence feed
        text_for_threats = text
        if text_for_threats:
            matches = self.threat_feed.match(text_for_threats)
            for m in matches:
                threat_matches.append(f"[{m.severity.value}] {m.technique}")
                risk = self._max_risk(risk, m.severity.value)
                findings.append(f"threat:{m.indicator.id}")
                if m.severity.value == "critical" and not block_reason:
                    warnings.append(f"Threat: {m.technique}")

        # 4. Record in attack chain detector
        chain_verdict = self.chain_detector.record(tool_name, arguments)
        stage = chain_verdict.stage or ""
        if chain_verdict.alert:
            for chain in chain_verdict.chains_detected:
                risk = "critical"
                findings.append(f"chain:{chain.name}")
                warnings.append(f"Attack chain: {chain.description}")

        chains_detected = [c.name for c in chain_verdict.chains_detected]

        # 5. Apply custom rules
        for rule in self._custom_rules:
            result = rule(tool_name, arguments)
            if result:
                block_reason = block_reason or result
                risk = "critical"
                findings.append("custom_rule")

        # 6. Decide: block or allow
        should_block = bool(block_reason) or self._risk_exceeds_threshold(risk)
        if should_block and not block_reason:
            block_reason = f"risk_threshold_exceeded: {risk} >= {self._block_threshold}"

        # 7. Log to audit trail
        if should_block:
            self.audit.log_blocked(
                tool_name, arguments,
                reason=block_reason,
                risk=risk,
                findings=findings,
            )
        else:
            self.audit.log_tool_call(
                tool_name, arguments,
                risk=risk,
                findings=findings,
            )

        return GuardVerdict(
            allowed=not should_block,
            risk=risk,
            tool_name=tool_name,
            block_reason=block_reason,
            warnings=warnings,
            threat_matches=threat_matches,
            chains_detected=chains_detected,
            stage=stage,
        )

    def export(self) -> dict[str, Any]:
        """Export full audit trail with chain analysis."""
        report = self.audit.export()
        report["active_chains"] = [
            {
                "name": c.name,
                "description": c.description,
                "severity": c.severity.value if hasattr(c.severity, 'value') else str(c.severity),
                "confidence": c.confidence,
                "stages": [
                    {"stage": s.stage, "tool": s.tool_name, "timestamp": s.timestamp}
                    for s in c.stages
                ],
            }
            for c in self.chain_detector.active_chains()
        ]
        return report

    def export_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.export(), indent=indent, default=str)

    def verify_integrity(self) -> bool:
        """Verify the audit trail hasn't been tampered with."""
        return self.audit.verify_integrity()

    @property
    def total_checks(self) -> int:
        return self.audit.total_entries

    def _extract_text(self, tool_name: str, arguments: dict) -> str:
        parts: list[str] = []
        for key in ("command", "cmd"):
            if key in arguments and isinstance(arguments[key], str):
                parts.append(arguments[key])
        for key in ("path", "file_path"):
            if key in arguments and isinstance(arguments[key], str):
                parts.append(arguments[key])
        for key in ("content", "new_string"):
            if key in arguments and isinstance(arguments[key], str):
                parts.append(arguments[key])
        if not parts:
            for v in arguments.values():
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts)

    def _max_risk(self, a: str, b: str) -> str:
        ia = self._risk_order.index(a) if a in self._risk_order else 0
        ib = self._risk_order.index(b) if b in self._risk_order else 0
        return self._risk_order[max(ia, ib)]

    def _risk_exceeds_threshold(self, risk: str) -> bool:
        ri = self._risk_order.index(risk) if risk in self._risk_order else 0
        ti = self._risk_order.index(self._block_threshold) if self._block_threshold in self._risk_order else 4
        return ri >= ti
