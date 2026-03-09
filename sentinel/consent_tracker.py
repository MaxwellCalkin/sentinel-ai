"""User consent tracking for compliance.

Track and enforce user consent for data processing operations,
supporting GDPR, CCPA, and other privacy regulations.

Usage:
    from sentinel.consent_tracker import ConsentTracker

    tracker = ConsentTracker()
    tracker.grant("user_123", "data_collection")
    assert tracker.has_consent("user_123", "data_collection")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConsentRecord:
    """A single consent record."""
    user_id: str
    purpose: str
    granted: bool
    timestamp: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class ConsentStatus:
    """Status of consent for a user."""
    user_id: str
    consents: dict[str, bool]  # purpose -> granted
    missing: list[str]         # Required purposes without consent


class ConsentTracker:
    """Track user consent for data processing.

    Supports granting, revoking, checking, and auditing consent
    with expiration and required purpose enforcement.
    """

    def __init__(
        self,
        required_purposes: list[str] | None = None,
    ) -> None:
        """
        Args:
            required_purposes: Purposes that require explicit consent.
        """
        self._required = set(required_purposes or [])
        self._records: dict[str, dict[str, ConsentRecord]] = {}  # user -> purpose -> record

    def grant(
        self,
        user_id: str,
        purpose: str,
        expires_in: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConsentRecord:
        """Grant consent for a purpose.

        Args:
            user_id: User identifier.
            purpose: Purpose of data processing.
            expires_in: Seconds until consent expires.
            metadata: Additional metadata.

        Returns:
            The ConsentRecord created.
        """
        expires_at = time.time() + expires_in if expires_in else None
        record = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            granted=True,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        if user_id not in self._records:
            self._records[user_id] = {}
        self._records[user_id][purpose] = record
        return record

    def revoke(self, user_id: str, purpose: str) -> bool:
        """Revoke consent for a purpose.

        Returns:
            True if consent was found and revoked.
        """
        if user_id in self._records and purpose in self._records[user_id]:
            self._records[user_id][purpose] = ConsentRecord(
                user_id=user_id,
                purpose=purpose,
                granted=False,
            )
            return True
        return False

    def revoke_all(self, user_id: str) -> int:
        """Revoke all consent for a user. Returns count revoked."""
        if user_id not in self._records:
            return 0
        count = 0
        for purpose in list(self._records[user_id].keys()):
            self._records[user_id][purpose] = ConsentRecord(
                user_id=user_id,
                purpose=purpose,
                granted=False,
            )
            count += 1
        return count

    def has_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has active consent for a purpose."""
        records = self._records.get(user_id, {})
        record = records.get(purpose)
        if record is None:
            return False
        if not record.granted:
            return False
        if record.expired:
            return False
        return True

    def check_required(self, user_id: str) -> ConsentStatus:
        """Check if user has all required consents.

        Returns:
            ConsentStatus with missing required purposes.
        """
        consents: dict[str, bool] = {}
        missing: list[str] = []

        for purpose in self._required:
            has = self.has_consent(user_id, purpose)
            consents[purpose] = has
            if not has:
                missing.append(purpose)

        return ConsentStatus(
            user_id=user_id,
            consents=consents,
            missing=missing,
        )

    def get_records(self, user_id: str) -> list[ConsentRecord]:
        """Get all consent records for a user."""
        return list(self._records.get(user_id, {}).values())

    @property
    def user_count(self) -> int:
        """Number of users with records."""
        return len(self._records)

    def delete_user(self, user_id: str) -> bool:
        """Delete all records for a user (right to erasure)."""
        if user_id in self._records:
            del self._records[user_id]
            return True
        return False

    def audit(self) -> dict[str, Any]:
        """Generate audit report."""
        total_records = sum(len(r) for r in self._records.values())
        granted = sum(
            1 for records in self._records.values()
            for r in records.values()
            if r.granted and not r.expired
        )
        return {
            "total_users": self.user_count,
            "total_records": total_records,
            "active_consents": granted,
            "revoked_or_expired": total_records - granted,
            "required_purposes": list(self._required),
        }
