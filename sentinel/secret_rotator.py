"""Credential rotation scheduling and tracking.

Track secret rotation schedules, record rotation history, and alert
when credentials are overdue for rotation. Essential for security
compliance and operational hygiene.

Usage:
    from sentinel.secret_rotator import SecretRotator

    rotator = SecretRotator(default_rotation_days=90)
    rotator.register("db_password")
    rotator.rotate("db_password")

    status = rotator.check("db_password")
    print(status.urgency)  # "ok"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

_SECONDS_PER_DAY = 86400


@dataclass
class SecretEntry:
    """A registered secret with its rotation schedule."""
    name: str
    rotation_days: int
    last_rotated: float
    created_at: float


@dataclass
class RotationStatus:
    """Current rotation status of a single secret."""
    name: str
    days_since_rotation: int
    days_until_due: int
    overdue: bool
    urgency: str  # "ok", "warning", "overdue", "critical"


@dataclass
class RotationEvent:
    """A single rotation event in a secret's history."""
    name: str
    rotated_at: float
    previous_rotation: float | None


@dataclass
class RotationReport:
    """Aggregate rotation report across all tracked secrets."""
    total: int
    overdue_count: int
    warning_count: int
    ok_count: int
    statuses: list[RotationStatus] = field(default_factory=list)
    overdue_secrets: list[str] = field(default_factory=list)


def _compute_urgency(days_until_due: int) -> str:
    if days_until_due > 14:
        return "ok"
    if days_until_due >= 0:
        return "warning"
    if days_until_due >= -30:
        return "overdue"
    return "critical"


class SecretRotator:
    """Manage credential rotation scheduling and tracking.

    Tracks registered secrets, records rotation history, and reports
    on rotation status with urgency levels.
    """

    def __init__(self, default_rotation_days: int = 90) -> None:
        self._default_rotation_days = default_rotation_days
        self._secrets: dict[str, SecretEntry] = {}
        self._history: dict[str, list[RotationEvent]] = {}

    def register(
        self,
        name: str,
        rotation_days: int | None = None,
        last_rotated: float | None = None,
    ) -> None:
        """Register a secret to track.

        Args:
            name: Unique identifier for the secret.
            rotation_days: Days between rotations (uses default if None).
            last_rotated: Timestamp of last rotation (defaults to now).

        Raises:
            ValueError: If the secret is already registered.
        """
        if name in self._secrets:
            raise ValueError(f"Secret already registered: {name}")

        now = time.time()
        entry = SecretEntry(
            name=name,
            rotation_days=rotation_days or self._default_rotation_days,
            last_rotated=last_rotated if last_rotated is not None else now,
            created_at=now,
        )
        self._secrets[name] = entry
        self._history[name] = []

    def rotate(self, name: str) -> None:
        """Mark a secret as rotated now.

        Args:
            name: The secret to rotate.

        Raises:
            KeyError: If the secret is not registered.
        """
        entry = self._get_entry(name)
        previous_rotation = entry.last_rotated
        now = time.time()
        entry.last_rotated = now

        event = RotationEvent(
            name=name,
            rotated_at=now,
            previous_rotation=previous_rotation,
        )
        self._history[name].append(event)

    def check(self, name: str) -> RotationStatus:
        """Check the rotation status of a specific secret.

        Args:
            name: The secret to check.

        Returns:
            RotationStatus with urgency assessment.

        Raises:
            KeyError: If the secret is not registered.
        """
        entry = self._get_entry(name)
        return self._build_status(entry)

    def check_all(self) -> RotationReport:
        """Check rotation status of all registered secrets.

        Returns:
            RotationReport with aggregate counts and per-secret statuses.
        """
        statuses = [self._build_status(entry) for entry in self._secrets.values()]
        overdue_secrets = [s.name for s in statuses if s.overdue]

        overdue_count = sum(1 for s in statuses if s.urgency in ("overdue", "critical"))
        warning_count = sum(1 for s in statuses if s.urgency == "warning")
        ok_count = sum(1 for s in statuses if s.urgency == "ok")

        return RotationReport(
            total=len(statuses),
            overdue_count=overdue_count,
            warning_count=warning_count,
            ok_count=ok_count,
            statuses=statuses,
            overdue_secrets=overdue_secrets,
        )

    def overdue(self) -> list[str]:
        """List names of secrets that are overdue for rotation."""
        return [
            name for name, entry in self._secrets.items()
            if self._build_status(entry).overdue
        ]

    def upcoming(self, days: int = 7) -> list[str]:
        """List names of secrets due for rotation within N days.

        Args:
            days: Look-ahead window in days.

        Returns:
            Names of secrets due within the window (but not yet overdue).
        """
        results = []
        for name, entry in self._secrets.items():
            status = self._build_status(entry)
            if not status.overdue and 0 <= status.days_until_due <= days:
                results.append(name)
        return results

    def history(self, name: str) -> list[RotationEvent]:
        """Get rotation history for a secret.

        Args:
            name: The secret to look up.

        Returns:
            List of RotationEvent records in chronological order.

        Raises:
            KeyError: If the secret is not registered.
        """
        self._get_entry(name)
        return list(self._history[name])

    def remove(self, name: str) -> None:
        """Remove a secret from tracking.

        Args:
            name: The secret to remove.

        Raises:
            KeyError: If the secret is not registered.
        """
        self._get_entry(name)
        del self._secrets[name]
        del self._history[name]

    def list_secrets(self) -> list[str]:
        """List all tracked secret names."""
        return list(self._secrets.keys())

    def _get_entry(self, name: str) -> SecretEntry:
        """Retrieve a secret entry or raise KeyError."""
        try:
            return self._secrets[name]
        except KeyError:
            raise KeyError(f"Secret not registered: {name}")

    def _build_status(self, entry: SecretEntry) -> RotationStatus:
        """Build a RotationStatus from a SecretEntry."""
        now = time.time()
        days_since = (now - entry.last_rotated) / _SECONDS_PER_DAY
        days_until_due = entry.rotation_days - days_since

        days_since_int = int(days_since)
        days_until_due_int = int(days_until_due)

        return RotationStatus(
            name=entry.name,
            days_since_rotation=days_since_int,
            days_until_due=days_until_due_int,
            overdue=days_until_due < 0,
            urgency=_compute_urgency(days_until_due_int),
        )
