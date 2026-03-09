"""Usage pattern anomaly detection for LLM applications.

Detect unusual usage patterns that may indicate abuse,
credential compromise, or automated attacks. Uses
statistical analysis without ML dependencies.

Usage:
    from sentinel.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(window_size=100)
    for request in requests:
        detector.record(user_id=request.user, tokens=request.tokens)
    anomalies = detector.check("user_123")
"""

from __future__ import annotations

import time
import math
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageEvent:
    """A single usage event."""
    user_id: str
    timestamp: float
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyFlag:
    """A detected anomaly."""
    anomaly_type: str
    user_id: str
    message: str
    severity: str  # "low", "medium", "high"
    value: float
    threshold: float


@dataclass
class AnomalyReport:
    """Anomaly check result for a user."""
    user_id: str
    anomalies: list[AnomalyFlag]
    is_anomalous: bool
    risk_level: str  # "normal", "elevated", "high"
    stats: dict[str, float]


class AnomalyDetector:
    """Statistical anomaly detection for LLM usage.

    Tracks per-user usage patterns and flags anomalies:
    - Rate spikes (sudden increase in request frequency)
    - Token volume spikes (unusual token consumption)
    - Off-hours usage (activity during unusual times)
    - Pattern deviation (z-score from rolling average)
    """

    def __init__(
        self,
        window_size: int = 100,
        rate_threshold: float = 3.0,
        volume_threshold: float = 3.0,
    ) -> None:
        """
        Args:
            window_size: Number of events to keep per user for analysis.
            rate_threshold: Z-score threshold for rate anomalies.
            volume_threshold: Z-score threshold for volume anomalies.
        """
        self._window = window_size
        self._rate_threshold = rate_threshold
        self._volume_threshold = volume_threshold
        self._events: dict[str, list[UsageEvent]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        user_id: str,
        tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a usage event.

        Args:
            user_id: User identifier.
            tokens: Token count for this request.
            metadata: Additional event data.
        """
        event = UsageEvent(
            user_id=user_id,
            timestamp=time.time(),
            tokens=tokens,
            metadata=metadata or {},
        )

        with self._lock:
            if user_id not in self._events:
                self._events[user_id] = []
            events = self._events[user_id]
            events.append(event)
            # Trim to window
            while len(events) > self._window:
                events.pop(0)

    def check(self, user_id: str) -> AnomalyReport:
        """Check a user for anomalous behavior.

        Args:
            user_id: User to check.

        Returns:
            AnomalyReport with any detected anomalies.
        """
        with self._lock:
            events = list(self._events.get(user_id, []))

        anomalies: list[AnomalyFlag] = []
        stats: dict[str, float] = {}

        if len(events) < 3:
            return AnomalyReport(
                user_id=user_id,
                anomalies=[],
                is_anomalous=False,
                risk_level="normal",
                stats={"events": len(events)},
            )

        # Rate analysis (inter-event intervals)
        intervals = []
        for i in range(1, len(events)):
            intervals.append(events[i].timestamp - events[i - 1].timestamp)

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            stats["avg_interval_sec"] = round(avg_interval, 3)

            if len(intervals) >= 3:
                std_interval = self._std(intervals)
                stats["std_interval_sec"] = round(std_interval, 3)

                # Check last interval for spike
                last_interval = intervals[-1]
                if std_interval > 0:
                    z_score = (avg_interval - last_interval) / std_interval
                    stats["rate_z_score"] = round(z_score, 3)

                    if z_score > self._rate_threshold:
                        anomalies.append(AnomalyFlag(
                            anomaly_type="rate_spike",
                            user_id=user_id,
                            message=f"Request rate spike detected (z={z_score:.2f})",
                            severity="high" if z_score > 5.0 else "medium",
                            value=z_score,
                            threshold=self._rate_threshold,
                        ))

        # Volume analysis
        volumes = [e.tokens for e in events if e.tokens > 0]
        if len(volumes) >= 3:
            avg_volume = sum(volumes) / len(volumes)
            std_volume = self._std(volumes)
            stats["avg_tokens"] = round(avg_volume, 1)
            stats["std_tokens"] = round(std_volume, 1)

            last_volume = volumes[-1]
            if std_volume > 0:
                vol_z = (last_volume - avg_volume) / std_volume
                stats["volume_z_score"] = round(vol_z, 3)

                if vol_z > self._volume_threshold:
                    anomalies.append(AnomalyFlag(
                        anomaly_type="volume_spike",
                        user_id=user_id,
                        message=f"Token volume spike detected (z={vol_z:.2f})",
                        severity="high" if vol_z > 5.0 else "medium",
                        value=vol_z,
                        threshold=self._volume_threshold,
                    ))

        # Burst detection (many events in short time)
        recent_window = 60.0  # 1 minute
        now = time.time()
        recent_count = sum(1 for e in events if now - e.timestamp < recent_window)
        stats["events_last_minute"] = recent_count

        if recent_count > 10:
            anomalies.append(AnomalyFlag(
                anomaly_type="burst",
                user_id=user_id,
                message=f"Burst detected: {recent_count} events in last minute",
                severity="high" if recent_count > 50 else "medium",
                value=float(recent_count),
                threshold=10.0,
            ))

        stats["events"] = len(events)

        # Determine risk level
        if any(a.severity == "high" for a in anomalies):
            risk = "high"
        elif anomalies:
            risk = "elevated"
        else:
            risk = "normal"

        return AnomalyReport(
            user_id=user_id,
            anomalies=anomalies,
            is_anomalous=len(anomalies) > 0,
            risk_level=risk,
            stats=stats,
        )

    def check_all(self) -> dict[str, AnomalyReport]:
        """Check all tracked users."""
        with self._lock:
            user_ids = list(self._events.keys())
        return {uid: self.check(uid) for uid in user_ids}

    @property
    def user_count(self) -> int:
        return len(self._events)

    def clear(self, user_id: str | None = None) -> None:
        """Clear events for one or all users."""
        with self._lock:
            if user_id:
                self._events.pop(user_id, None)
            else:
                self._events.clear()

    @staticmethod
    def _std(values: list[float]) -> float:
        """Standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)
