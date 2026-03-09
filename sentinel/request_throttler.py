"""Adaptive request throttling based on safety signals.

Adjusts rate limits dynamically when safety incidents increase,
providing automatic protection escalation and cool-down recovery.

Usage:
    from sentinel.request_throttler import RequestThrottler

    throttler = RequestThrottler(base_rate=100, window_seconds=60)
    throttler.report_incident("injection_attempt")
    decision = throttler.check_request()
    if decision.allowed:
        response = llm.complete(prompt)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import List


LEVEL_NORMAL = "normal"
LEVEL_CAUTIOUS = "cautious"
LEVEL_RESTRICTED = "restricted"
LEVEL_BLOCKED = "blocked"

LEVEL_MULTIPLIERS = {
    LEVEL_NORMAL: 1.0,
    LEVEL_CAUTIOUS: 0.5,
    LEVEL_RESTRICTED: 0.25,
    LEVEL_BLOCKED: 0.0,
}


@dataclass
class ThrottleStatus:
    """Current throttle state snapshot."""
    level: str
    allowed_rate: float
    base_rate: int
    window_seconds: float
    incidents_in_window: int


@dataclass
class RequestDecision:
    """Result of a request check."""
    allowed: bool
    throttle_level: str
    reason: str
    wait_seconds: float


@dataclass
class ThrottlerStats:
    """Aggregate throttler statistics."""
    total_requests: int
    denied_count: int
    current_level: str
    total_incidents: int
    deny_rate: float


@dataclass
class _IncidentRecord:
    """Internal record of a single incident."""
    category: str
    timestamp: float


class RequestThrottler:
    """Adaptive request throttler driven by safety incidents.

    Monitors safety incident frequency within a sliding window and
    escalates throttle level when incidents exceed configurable
    thresholds. Automatically recovers to normal after a cool-down
    period with no new incidents.
    """

    def __init__(
        self,
        base_rate: int = 100,
        window_seconds: float = 60.0,
        cautious_threshold: int = 3,
        restricted_threshold: int = 7,
        blocked_threshold: int = 12,
        cooldown_seconds: float = 120.0,
    ) -> None:
        """
        Args:
            base_rate: Maximum requests allowed per window at normal level.
            window_seconds: Sliding window duration in seconds.
            cautious_threshold: Incidents in window to trigger cautious.
            restricted_threshold: Incidents in window to trigger restricted.
            blocked_threshold: Incidents in window to trigger blocked.
            cooldown_seconds: Seconds with no incidents before recovering one level.
        """
        self._base_rate = base_rate
        self._window_seconds = window_seconds
        self._cautious_threshold = cautious_threshold
        self._restricted_threshold = restricted_threshold
        self._blocked_threshold = blocked_threshold
        self._cooldown_seconds = cooldown_seconds

        self._incidents: List[_IncidentRecord] = []
        self._request_timestamps: List[float] = []
        self._total_requests = 0
        self._denied_count = 0
        self._lock = threading.Lock()

    def _current_time(self) -> float:
        return time.monotonic()

    def _prune_old_entries(self, now: float) -> None:
        cutoff = now - self._window_seconds
        self._incidents = [i for i in self._incidents if i.timestamp >= cutoff]
        self._request_timestamps = [t for t in self._request_timestamps if t >= cutoff]

    def _compute_level_from_incidents(self, incident_count: int) -> str:
        if incident_count >= self._blocked_threshold:
            return LEVEL_BLOCKED
        if incident_count >= self._restricted_threshold:
            return LEVEL_RESTRICTED
        if incident_count >= self._cautious_threshold:
            return LEVEL_CAUTIOUS
        return LEVEL_NORMAL

    def _apply_cooldown(self, raw_level: str, now: float) -> str:
        if raw_level == LEVEL_NORMAL:
            return LEVEL_NORMAL
        if not self._incidents:
            return LEVEL_NORMAL
        last_incident_time = max(i.timestamp for i in self._incidents)
        elapsed_since_last = now - last_incident_time
        levels_to_recover = int(elapsed_since_last // self._cooldown_seconds)
        if levels_to_recover <= 0:
            return raw_level
        ordered_levels = [LEVEL_BLOCKED, LEVEL_RESTRICTED, LEVEL_CAUTIOUS, LEVEL_NORMAL]
        current_index = ordered_levels.index(raw_level)
        recovered_index = min(current_index + levels_to_recover, len(ordered_levels) - 1)
        return ordered_levels[recovered_index]

    def _effective_level(self, now: float) -> str:
        incident_count = len(self._incidents)
        raw_level = self._compute_level_from_incidents(incident_count)
        return self._apply_cooldown(raw_level, now)

    def _allowed_rate_for_level(self, level: str) -> float:
        return self._base_rate * LEVEL_MULTIPLIERS[level]

    def report_incident(self, category: str = "general") -> None:
        """Record a safety incident that may escalate throttling.

        Args:
            category: Incident category label.
        """
        with self._lock:
            now = self._current_time()
            self._incidents.append(_IncidentRecord(category=category, timestamp=now))

    def check_request(self) -> RequestDecision:
        """Check whether a new request should be allowed.

        Returns:
            RequestDecision with allowed status and current throttle level.
        """
        with self._lock:
            now = self._current_time()
            self._prune_old_entries(now)
            self._total_requests += 1

            level = self._effective_level(now)
            allowed_rate = self._allowed_rate_for_level(level)

            if level == LEVEL_BLOCKED:
                self._denied_count += 1
                return RequestDecision(
                    allowed=False,
                    throttle_level=level,
                    reason="All requests blocked due to high incident volume",
                    wait_seconds=self._cooldown_seconds,
                )

            requests_in_window = len(self._request_timestamps)
            if requests_in_window >= allowed_rate:
                self._denied_count += 1
                wait = self._estimate_wait_seconds(now)
                return RequestDecision(
                    allowed=False,
                    throttle_level=level,
                    reason=f"Rate limit exceeded at {level} level ({int(allowed_rate)}/{self._window_seconds}s)",
                    wait_seconds=wait,
                )

            self._request_timestamps.append(now)
            return RequestDecision(
                allowed=True,
                throttle_level=level,
                reason=f"Request allowed at {level} level",
                wait_seconds=0.0,
            )

    def _estimate_wait_seconds(self, now: float) -> float:
        if not self._request_timestamps:
            return 0.0
        oldest_in_window = min(self._request_timestamps)
        expires_at = oldest_in_window + self._window_seconds
        wait = expires_at - now
        return max(wait, 0.0)

    def status(self) -> ThrottleStatus:
        """Get current throttle status snapshot."""
        with self._lock:
            now = self._current_time()
            self._prune_old_entries(now)
            level = self._effective_level(now)
            return ThrottleStatus(
                level=level,
                allowed_rate=self._allowed_rate_for_level(level),
                base_rate=self._base_rate,
                window_seconds=self._window_seconds,
                incidents_in_window=len(self._incidents),
            )

    def stats(self) -> ThrottlerStats:
        """Get aggregate throttler statistics."""
        with self._lock:
            now = self._current_time()
            self._prune_old_entries(now)
            total_incidents = len(self._incidents)
            level = self._effective_level(now)
            deny_rate = (
                self._denied_count / self._total_requests
                if self._total_requests > 0
                else 0.0
            )
            return ThrottlerStats(
                total_requests=self._total_requests,
                denied_count=self._denied_count,
                current_level=level,
                total_incidents=total_incidents,
                deny_rate=deny_rate,
            )

    def reset(self) -> None:
        """Reset all state to initial values."""
        with self._lock:
            self._incidents.clear()
            self._request_timestamps.clear()
            self._total_requests = 0
            self._denied_count = 0
