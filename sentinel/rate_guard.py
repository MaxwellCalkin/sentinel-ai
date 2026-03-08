"""Rate-based abuse detection for LLM applications.

Tracks per-user safety violation frequency and auto-escalates risk when
users repeatedly trigger safety scanners. Protects against persistent
injection attempts, automated attacks, and abuse patterns.

Usage:
    from sentinel.rate_guard import RateGuard

    rate_guard = RateGuard(window_seconds=300, max_violations=3)

    # On each request
    result = guard.scan(user_input)
    verdict = rate_guard.check(user_id="user-123", scan_result=result)

    if verdict.throttled:
        # User has exceeded violation threshold
        print(f"User throttled: {verdict.reason}")
        # Return error to user, log to SIEM, alert security team

    if verdict.escalated_risk:
        # Risk was auto-escalated due to repeated violations
        print(f"Escalated from {result.risk} to {verdict.risk}")
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from sentinel.core import RiskLevel, ScanResult


@dataclass
class ViolationRecord:
    """A single recorded safety violation."""
    timestamp: float
    risk: RiskLevel
    category: str
    user_id: str


@dataclass
class RateVerdict:
    """Result of rate-based abuse check."""
    user_id: str
    risk: RiskLevel
    throttled: bool = False
    escalated_risk: bool = False
    reason: str | None = None
    violation_count: int = 0
    window_seconds: int = 300


class RateGuard:
    """Rate-based abuse detection and throttling.

    Tracks per-user safety violations within a sliding time window.
    When a user exceeds the violation threshold, subsequent requests
    are auto-throttled and risk is escalated.

    Args:
        window_seconds: Time window for counting violations (default: 300s = 5min).
        max_violations: Max violations in window before throttling (default: 3).
        escalation_threshold: Violations before risk auto-escalation (default: 2).
        cooldown_seconds: How long throttle lasts after threshold hit (default: 60).
    """

    def __init__(
        self,
        window_seconds: int = 300,
        max_violations: int = 3,
        escalation_threshold: int = 2,
        cooldown_seconds: int = 60,
    ) -> None:
        self._window = window_seconds
        self._max_violations = max_violations
        self._escalation_threshold = escalation_threshold
        self._cooldown = cooldown_seconds
        self._violations: dict[str, list[ViolationRecord]] = defaultdict(list)
        self._throttled_until: dict[str, float] = {}

    def check(self, user_id: str, scan_result: ScanResult) -> RateVerdict:
        """Check a scan result against rate limits for a user.

        Args:
            user_id: Unique identifier for the user/session.
            scan_result: The ScanResult from SentinelGuard.scan().

        Returns:
            RateVerdict with throttle/escalation status.
        """
        now = time.time()
        self._prune(user_id, now)

        # Check if currently throttled
        throttled_until = self._throttled_until.get(user_id, 0)
        if now < throttled_until:
            return RateVerdict(
                user_id=user_id,
                risk=RiskLevel.CRITICAL,
                throttled=True,
                reason=f"User throttled for {int(throttled_until - now)}s due to repeated safety violations",
                violation_count=len(self._violations[user_id]),
                window_seconds=self._window,
            )

        # Record violations from this scan
        if scan_result.findings:
            for finding in scan_result.findings:
                if finding.risk >= RiskLevel.MEDIUM:
                    self._violations[user_id].append(ViolationRecord(
                        timestamp=now,
                        risk=finding.risk,
                        category=finding.category,
                        user_id=user_id,
                    ))

        violations = self._violations[user_id]
        count = len(violations)

        # Check throttle threshold
        if count >= self._max_violations:
            self._throttled_until[user_id] = now + self._cooldown
            return RateVerdict(
                user_id=user_id,
                risk=RiskLevel.CRITICAL,
                throttled=True,
                reason=f"{count} safety violations in {self._window}s (threshold: {self._max_violations})",
                violation_count=count,
                window_seconds=self._window,
            )

        # Check escalation threshold
        risk = scan_result.risk
        escalated = False
        if count >= self._escalation_threshold and risk < RiskLevel.CRITICAL:
            if risk == RiskLevel.NONE:
                risk = RiskLevel.LOW
            elif risk == RiskLevel.LOW:
                risk = RiskLevel.MEDIUM
            elif risk == RiskLevel.MEDIUM:
                risk = RiskLevel.HIGH
            elif risk == RiskLevel.HIGH:
                risk = RiskLevel.CRITICAL
            escalated = risk != scan_result.risk

        return RateVerdict(
            user_id=user_id,
            risk=risk,
            throttled=False,
            escalated_risk=escalated,
            reason=f"Risk escalated due to {count} recent violations" if escalated else None,
            violation_count=count,
            window_seconds=self._window,
        )

    def get_violation_count(self, user_id: str) -> int:
        """Get current violation count for a user."""
        self._prune(user_id, time.time())
        return len(self._violations[user_id])

    def reset(self, user_id: str) -> None:
        """Reset violation history for a user."""
        self._violations.pop(user_id, None)
        self._throttled_until.pop(user_id, None)

    def reset_all(self) -> None:
        """Reset all tracked state."""
        self._violations.clear()
        self._throttled_until.clear()

    def _prune(self, user_id: str, now: float) -> None:
        """Remove violations outside the time window."""
        cutoff = now - self._window
        self._violations[user_id] = [
            v for v in self._violations[user_id] if v.timestamp > cutoff
        ]
