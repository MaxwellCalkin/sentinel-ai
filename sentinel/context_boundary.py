"""Context boundary enforcement for LLM conversation segments.

Enforces hard boundaries between system/user/assistant context segments.
Prevents user content from bleeding into system context and detects
boundary violation attempts including role injection, privilege escalation,
and context leakage.

Usage:
    from sentinel.context_boundary import ContextBoundary, Segment, BoundaryConfig

    boundary = ContextBoundary()
    segments = [
        Segment(role="system", content="You are helpful.", index=0),
        Segment(role="user", content="Hello!", index=1),
    ]
    report = boundary.check(segments)
    assert report.is_clean
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

_ROLE_TAG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|\n)\s*system\s*:", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"###\s*System", re.IGNORECASE),
    re.compile(r"```system", re.IGNORECASE),
    re.compile(r"<\|?system\|?>", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*assistant\s*:", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*human\s*:", re.IGNORECASE),
]

_BOUNDARY_BLEED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"you\s+must\b", re.IGNORECASE),
    re.compile(r"your\s+instructions\s+are\b", re.IGNORECASE),
    re.compile(r"as\s+a\s+system\b", re.IGNORECASE),
    re.compile(r"your\s+system\s+prompt\b", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"new\s+instructions\b", re.IGNORECASE),
    re.compile(r"override\s+(?:your\s+)?instructions\b", re.IGNORECASE),
]

_PRIVILEGE_ESCALATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"admin\s+mode", re.IGNORECASE),
    re.compile(r"debug\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+(?:mode|override)", re.IGNORECASE),
    re.compile(r"sudo\s+mode", re.IGNORECASE),
    re.compile(r"root\s+access", re.IGNORECASE),
    re.compile(r"maintenance\s+mode", re.IGNORECASE),
    re.compile(r"god\s+mode", re.IGNORECASE),
    re.compile(r"unrestricted\s+mode", re.IGNORECASE),
]

_CONTEXT_LEAK_MIN_LENGTH = 20


@dataclass
class Segment:
    """A single segment in an LLM conversation."""

    role: str
    content: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class BoundaryViolation:
    """A detected boundary violation."""

    violation_type: str
    segment_index: int
    description: str
    severity: str


@dataclass
class BoundaryReport:
    """Report from checking conversation segments for boundary violations."""

    segments_checked: int
    violations: list[BoundaryViolation]
    is_clean: bool
    risk_score: float


@dataclass
class BoundaryConfig:
    """Configuration for boundary enforcement."""

    allow_system_references: bool = False
    allow_role_tags: bool = False
    strict_ordering: bool = True
    max_system_segments: int = 1


@dataclass
class BoundaryStats:
    """Cumulative statistics for a ContextBoundary instance."""

    total_checked: int = 0
    violations_found: int = 0
    clean_count: int = 0


class ContextBoundary:
    """Enforce hard boundaries between conversation context segments.

    Detects role injection, boundary bleed, privilege escalation,
    context leakage, ordering violations, and system segment limits.
    """

    def __init__(self, config: BoundaryConfig | None = None) -> None:
        self._config = config or BoundaryConfig()
        self._total_checked = 0
        self._violations_found = 0
        self._clean_count = 0

    def check(self, segments: list[Segment]) -> BoundaryReport:
        """Check a conversation for boundary violations."""
        violations: list[BoundaryViolation] = []
        system_content = self._extract_system_content(segments)

        self._check_ordering(segments, violations)
        self._check_system_segment_count(segments, violations)

        for segment in segments:
            segment_violations = self._check_segment(segment, system_content)
            violations.extend(segment_violations)

        self._record_stats(len(segments), len(violations))

        risk_score = min(len(violations) * 0.2, 1.0)

        return BoundaryReport(
            segments_checked=len(segments),
            violations=violations,
            is_clean=len(violations) == 0,
            risk_score=risk_score,
        )

    def check_message(self, role: str, content: str) -> list[BoundaryViolation]:
        """Check a single message for boundary violations."""
        segment = Segment(role=role, content=content, index=0)
        violations = self._check_segment(segment, system_content="")
        self._record_stats(1, len(violations))
        return violations

    def sanitize(self, segment: Segment) -> Segment:
        """Remove detected violation patterns from segment content."""
        content = segment.content

        if segment.role in ("user", "assistant"):
            content = self._remove_role_tags(content)
            if segment.role == "user":
                content = self._remove_bleed_patterns(content)
                content = self._remove_escalation_patterns(content)

        content = _collapse_whitespace(content)

        return Segment(
            role=segment.role,
            content=content,
            index=segment.index,
            metadata=dict(segment.metadata),
        )

    def stats(self) -> BoundaryStats:
        """Return cumulative boundary check statistics."""
        return BoundaryStats(
            total_checked=self._total_checked,
            violations_found=self._violations_found,
            clean_count=self._clean_count,
        )

    # ------------------------------------------------------------------
    # Internal: per-segment checks
    # ------------------------------------------------------------------

    def _check_segment(
        self, segment: Segment, system_content: str
    ) -> list[BoundaryViolation]:
        violations: list[BoundaryViolation] = []

        if segment.role in ("user", "assistant"):
            self._detect_role_injection(segment, violations)

        if segment.role == "user":
            self._detect_boundary_bleed(segment, violations)
            self._detect_privilege_escalation(segment, violations)

        if segment.role == "assistant" and system_content:
            self._detect_context_leak(segment, system_content, violations)

        return violations

    def _detect_role_injection(
        self, segment: Segment, violations: list[BoundaryViolation]
    ) -> None:
        if self._config.allow_role_tags:
            return
        for pattern in _ROLE_TAG_PATTERNS:
            if pattern.search(segment.content):
                violations.append(
                    BoundaryViolation(
                        violation_type="role_injection",
                        segment_index=segment.index,
                        description=f"Role tag pattern found in {segment.role} content: {pattern.pattern}",
                        severity="high",
                    )
                )
                return

    def _detect_boundary_bleed(
        self, segment: Segment, violations: list[BoundaryViolation]
    ) -> None:
        if self._config.allow_system_references:
            return
        for pattern in _BOUNDARY_BLEED_PATTERNS:
            if pattern.search(segment.content):
                violations.append(
                    BoundaryViolation(
                        violation_type="boundary_bleed",
                        segment_index=segment.index,
                        description=f"Boundary bleed detected: {pattern.pattern}",
                        severity="medium",
                    )
                )
                return

    def _detect_privilege_escalation(
        self, segment: Segment, violations: list[BoundaryViolation]
    ) -> None:
        for pattern in _PRIVILEGE_ESCALATION_PATTERNS:
            if pattern.search(segment.content):
                violations.append(
                    BoundaryViolation(
                        violation_type="privilege_escalation",
                        segment_index=segment.index,
                        description=f"Privilege escalation attempt: {pattern.pattern}",
                        severity="high",
                    )
                )
                return

    def _detect_context_leak(
        self,
        segment: Segment,
        system_content: str,
        violations: list[BoundaryViolation],
    ) -> None:
        if len(system_content) <= _CONTEXT_LEAK_MIN_LENGTH:
            return
        if _has_overlapping_substring(segment.content, system_content):
            violations.append(
                BoundaryViolation(
                    violation_type="context_leak",
                    segment_index=segment.index,
                    description="Assistant content contains system prompt fragment",
                    severity="high",
                )
            )

    # ------------------------------------------------------------------
    # Internal: conversation-level checks
    # ------------------------------------------------------------------

    def _check_ordering(
        self, segments: list[Segment], violations: list[BoundaryViolation]
    ) -> None:
        if not self._config.strict_ordering:
            return
        if not segments:
            return

        system_indices = [s.index for s in segments if s.role == "system"]
        non_system_indices = [s.index for s in segments if s.role != "system"]

        if not system_indices or not non_system_indices:
            return

        first_system = min(system_indices)
        first_non_system = min(non_system_indices)

        if first_system > first_non_system:
            violations.append(
                BoundaryViolation(
                    violation_type="boundary_bleed",
                    segment_index=first_system,
                    description="System segment must appear before non-system segments",
                    severity="medium",
                )
            )

    def _check_system_segment_count(
        self, segments: list[Segment], violations: list[BoundaryViolation]
    ) -> None:
        system_count = sum(1 for s in segments if s.role == "system")
        if system_count > self._config.max_system_segments:
            violations.append(
                BoundaryViolation(
                    violation_type="boundary_bleed",
                    segment_index=0,
                    description=f"Too many system segments: {system_count} > {self._config.max_system_segments}",
                    severity="medium",
                )
            )

    # ------------------------------------------------------------------
    # Internal: sanitization helpers
    # ------------------------------------------------------------------

    def _remove_role_tags(self, content: str) -> str:
        for pattern in _ROLE_TAG_PATTERNS:
            content = pattern.sub("", content)
        return content

    def _remove_bleed_patterns(self, content: str) -> str:
        for pattern in _BOUNDARY_BLEED_PATTERNS:
            content = pattern.sub("", content)
        return content

    def _remove_escalation_patterns(self, content: str) -> str:
        for pattern in _PRIVILEGE_ESCALATION_PATTERNS:
            content = pattern.sub("", content)
        return content

    # ------------------------------------------------------------------
    # Internal: utilities
    # ------------------------------------------------------------------

    def _extract_system_content(self, segments: list[Segment]) -> str:
        system_parts = [s.content for s in segments if s.role == "system"]
        return " ".join(system_parts)

    def _record_stats(self, checked: int, violation_count: int) -> None:
        self._total_checked += checked
        self._violations_found += violation_count
        if violation_count == 0:
            self._clean_count += checked
        else:
            self._clean_count += max(0, checked - violation_count)


def _has_overlapping_substring(content: str, reference: str) -> bool:
    """Check if content contains a significant substring of reference."""
    content_lower = content.lower()
    reference_lower = reference.lower()
    check_length = _CONTEXT_LEAK_MIN_LENGTH + 1

    for start in range(len(reference_lower) - check_length + 1):
        substring = reference_lower[start : start + check_length]
        if substring in content_lower:
            return True
    return False


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip."""
    return re.sub(r"  +", " ", text).strip()
