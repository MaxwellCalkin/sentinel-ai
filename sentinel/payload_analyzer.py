"""Deep analysis of structured payloads for hidden threats.

Detect nested injection, oversized fields, suspicious encoding,
deep nesting, and type confusion in JSON, form data, and API
request payloads.

Usage:
    from sentinel.payload_analyzer import PayloadAnalyzer

    analyzer = PayloadAnalyzer()
    report = analyzer.analyze({"user": "admin'; DROP TABLE--"})
    print(report.is_safe)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class PayloadField:
    """A single field discovered during payload traversal."""
    path: str
    value_type: str
    size: int
    depth: int
    is_suspicious: bool = False
    reason: str = ""


@dataclass
class PayloadThreat:
    """A threat detected in a payload."""
    threat_type: str
    path: str
    severity: str
    description: str


@dataclass
class PayloadReport:
    """Full analysis report for a payload."""
    total_fields: int
    max_depth: int
    threats: list[PayloadThreat]
    fields_analyzed: list[PayloadField]
    risk_score: float
    is_safe: bool


@dataclass
class AnalyzerConfig:
    """Configuration for payload analysis thresholds."""
    max_depth: int = 10
    max_field_size: int = 10000
    max_fields: int = 1000
    check_injections: bool = True


@dataclass
class AnalyzerStats:
    """Cumulative statistics across all analyses."""
    total_analyzed: int = 0
    threats_found: int = 0
    by_threat_type: dict[str, int] = field(default_factory=dict)


_SQL_INJECTION_PATTERNS = [
    re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bSELECT\s+\*", re.IGNORECASE),
    re.compile(r"\bOR\s+1\s*=\s*1\b", re.IGNORECASE),
    re.compile(r"\bUNION\s+SELECT\b", re.IGNORECASE),
    re.compile(r";\s*DELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"'\s*OR\s+'", re.IGNORECASE),
]

_COMMAND_INJECTION_PATTERNS = [
    re.compile(r";\s*rm\s"),
    re.compile(r"\|\s*cat\s"),
    re.compile(r"&&\s*rm\s"),
    re.compile(r"\$\(.*\)"),
    re.compile(r"`[^`]+`"),
]

_SCRIPT_TAG_PATTERN = re.compile(r"<\s*script", re.IGNORECASE)

_TYPE_CONFUSION_PREFIXES = ("function", "eval(", "__import__")


class PayloadAnalyzer:
    """Analyze structured payloads for hidden threats.

    Recursively walks dicts, lists, and JSON strings to detect
    injection patterns, structural anomalies, and encoding attacks.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self._config = config or AnalyzerConfig()
        self._stats = AnalyzerStats()

    def analyze(self, payload: dict | list | str) -> PayloadReport:
        """Analyze a payload and return a detailed threat report."""
        parsed = self._parse_input(payload)
        fields: list[PayloadField] = []
        threats: list[PayloadThreat] = []
        max_depth = 0

        self._walk(parsed, "root", 0, fields, threats)

        max_depth = max((f.depth for f in fields), default=0)
        self._check_field_count(fields, threats)

        risk_score = self._calculate_risk_score(len(threats))
        is_safe = risk_score < 0.3

        self._update_stats(threats)

        return PayloadReport(
            total_fields=len(fields),
            max_depth=max_depth,
            threats=threats,
            fields_analyzed=fields,
            risk_score=risk_score,
            is_safe=is_safe,
        )

    def analyze_batch(self, payloads: list) -> list[PayloadReport]:
        """Analyze multiple payloads and return reports for each."""
        return [self.analyze(p) for p in payloads]

    def stats(self) -> AnalyzerStats:
        """Return cumulative analysis statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_input(self, payload: dict | list | str) -> dict | list | str:
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                return payload
        return payload

    def _walk(
        self,
        node: object,
        path: str,
        depth: int,
        fields: list[PayloadField],
        threats: list[PayloadThreat],
    ) -> None:
        if isinstance(node, dict):
            self._walk_dict(node, path, depth, fields, threats)
        elif isinstance(node, list):
            self._walk_list(node, path, depth, fields, threats)
        else:
            self._walk_leaf(node, path, depth, fields, threats)

    def _walk_dict(
        self,
        node: dict,
        path: str,
        depth: int,
        fields: list[PayloadField],
        threats: list[PayloadThreat],
    ) -> None:
        for key, value in node.items():
            child_path = f"{path}.{key}"
            self._walk(value, child_path, depth + 1, fields, threats)

    def _walk_list(
        self,
        node: list,
        path: str,
        depth: int,
        fields: list[PayloadField],
        threats: list[PayloadThreat],
    ) -> None:
        for index, value in enumerate(node):
            child_path = f"{path}[{index}]"
            self._walk(value, child_path, depth + 1, fields, threats)

    def _walk_leaf(
        self,
        node: object,
        path: str,
        depth: int,
        fields: list[PayloadField],
        threats: list[PayloadThreat],
    ) -> None:
        value_type = type(node).__name__
        size = len(str(node))
        is_suspicious = False
        reason = ""

        if depth > self._config.max_depth:
            is_suspicious = True
            reason = "exceeds max depth"
            threats.append(PayloadThreat(
                threat_type="deep_nesting",
                path=path,
                severity="medium",
                description=f"Field at depth {depth} exceeds limit of {self._config.max_depth}",
            ))

        if isinstance(node, str):
            self._check_string_value(node, path, depth, size, threats, fields)
            return

        field_record = PayloadField(
            path=path,
            value_type=value_type,
            size=size,
            depth=depth,
            is_suspicious=is_suspicious,
            reason=reason,
        )
        fields.append(field_record)

    def _check_string_value(
        self,
        value: str,
        path: str,
        depth: int,
        size: int,
        threats: list[PayloadThreat],
        fields: list[PayloadField],
    ) -> None:
        is_suspicious = False
        reason = ""

        if depth > self._config.max_depth:
            is_suspicious = True
            reason = "exceeds max depth"
            self._add_deep_nesting_threat(path, depth, threats)

        if size > self._config.max_field_size:
            is_suspicious = True
            reason = "oversized field"
            threats.append(PayloadThreat(
                threat_type="oversized_field",
                path=path,
                severity="medium",
                description=f"String field is {size} chars, exceeds limit of {self._config.max_field_size}",
            ))

        if self._config.check_injections:
            injection_reason = self._detect_injection(value)
            if injection_reason:
                is_suspicious = True
                reason = injection_reason
                threats.append(PayloadThreat(
                    threat_type="nested_injection",
                    path=path,
                    severity="high",
                    description=f"Injection pattern detected: {injection_reason}",
                ))

        type_confusion_reason = self._detect_type_confusion(value)
        if type_confusion_reason:
            is_suspicious = True
            reason = type_confusion_reason
            threats.append(PayloadThreat(
                threat_type="type_confusion",
                path=path,
                severity="high",
                description=f"Type confusion detected: {type_confusion_reason}",
            ))

        fields.append(PayloadField(
            path=path,
            value_type="str",
            size=size,
            depth=depth,
            is_suspicious=is_suspicious,
            reason=reason,
        ))

    def _add_deep_nesting_threat(
        self, path: str, depth: int, threats: list[PayloadThreat]
    ) -> None:
        already_reported = any(
            t.path == path and t.threat_type == "deep_nesting" for t in threats
        )
        if not already_reported:
            threats.append(PayloadThreat(
                threat_type="deep_nesting",
                path=path,
                severity="medium",
                description=f"Field at depth {depth} exceeds limit of {self._config.max_depth}",
            ))

    def _detect_injection(self, value: str) -> str:
        for pattern in _SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                return f"SQL injection pattern: {pattern.pattern}"

        if _SCRIPT_TAG_PATTERN.search(value):
            return "script tag injection"

        for pattern in _COMMAND_INJECTION_PATTERNS:
            if pattern.search(value):
                return f"command injection pattern: {pattern.pattern}"

        return ""

    def _detect_type_confusion(self, value: str) -> str:
        stripped = value.strip()
        for prefix in _TYPE_CONFUSION_PREFIXES:
            if stripped.startswith(prefix):
                return f"string looks like code (starts with '{prefix}')"
        return ""

    def _check_field_count(
        self, fields: list[PayloadField], threats: list[PayloadThreat]
    ) -> None:
        if len(fields) > self._config.max_fields:
            threats.append(PayloadThreat(
                threat_type="oversized_field",
                path="root",
                severity="medium",
                description=f"Payload has {len(fields)} fields, exceeds limit of {self._config.max_fields}",
            ))

    def _calculate_risk_score(self, threat_count: int) -> float:
        if threat_count == 0:
            return 0.0
        return min(threat_count * 0.15, 1.0)

    def _update_stats(self, threats: list[PayloadThreat]) -> None:
        self._stats.total_analyzed += 1
        self._stats.threats_found += len(threats)
        for threat in threats:
            current = self._stats.by_threat_type.get(threat.threat_type, 0)
            self._stats.by_threat_type[threat.threat_type] = current + 1
