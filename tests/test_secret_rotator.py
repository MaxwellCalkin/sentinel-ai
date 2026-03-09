"""Tests for secret rotation scheduling and tracking."""

import time
from unittest.mock import patch

import pytest
from sentinel.secret_rotator import (
    SecretRotator,
    RotationEvent,
    RotationReport,
    RotationStatus,
    SecretEntry,
)

_SECONDS_PER_DAY = 86400


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_secret(self):
        rotator = SecretRotator()
        rotator.register("db_password")
        assert "db_password" in rotator.list_secrets()

    def test_register_custom_period(self):
        rotator = SecretRotator()
        rotator.register("api_key", rotation_days=30)
        status = rotator.check("api_key")
        assert status.days_until_due <= 30

    def test_register_duplicate(self):
        rotator = SecretRotator()
        rotator.register("db_password")
        with pytest.raises(ValueError, match="already registered"):
            rotator.register("db_password")


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class TestRotate:
    def test_rotate_updates_timestamp(self):
        rotator = SecretRotator()
        old_time = time.time() - 50 * _SECONDS_PER_DAY
        rotator.register("token", last_rotated=old_time)

        status_before = rotator.check("token")
        assert status_before.days_since_rotation >= 49

        rotator.rotate("token")
        status_after = rotator.check("token")
        assert status_after.days_since_rotation == 0

    def test_rotate_records_history(self):
        rotator = SecretRotator()
        rotator.register("token")
        rotator.rotate("token")

        events = rotator.history("token")
        assert len(events) == 1
        assert isinstance(events[0], RotationEvent)
        assert events[0].name == "token"
        assert events[0].previous_rotation is not None


# ---------------------------------------------------------------------------
# Status checking
# ---------------------------------------------------------------------------

class TestCheck:
    def test_check_fresh(self):
        rotator = SecretRotator(default_rotation_days=90)
        rotator.register("key")
        status = rotator.check("key")
        assert isinstance(status, RotationStatus)
        assert status.days_since_rotation == 0
        assert status.days_until_due >= 89
        assert status.overdue is False
        assert status.urgency == "ok"

    def test_check_overdue(self):
        rotator = SecretRotator(default_rotation_days=30)
        past = time.time() - 45 * _SECONDS_PER_DAY
        rotator.register("old_key", last_rotated=past)

        status = rotator.check("old_key")
        assert status.overdue is True
        assert status.days_until_due < 0

    def test_urgency_levels(self):
        now = time.time()
        rotator = SecretRotator(default_rotation_days=90)

        # "ok": >14 days remaining
        rotator.register("ok_key", last_rotated=now - 50 * _SECONDS_PER_DAY)
        assert rotator.check("ok_key").urgency == "ok"

        # "warning": 0-14 days remaining
        rotator.register("warn_key", last_rotated=now - 80 * _SECONDS_PER_DAY)
        assert rotator.check("warn_key").urgency == "warning"

        # "overdue": 0 to -30 days past due
        rotator.register("overdue_key", last_rotated=now - 100 * _SECONDS_PER_DAY)
        assert rotator.check("overdue_key").urgency == "overdue"

        # "critical": more than 30 days past due
        rotator.register("critical_key", last_rotated=now - 130 * _SECONDS_PER_DAY)
        assert rotator.check("critical_key").urgency == "critical"


# ---------------------------------------------------------------------------
# Aggregate reports
# ---------------------------------------------------------------------------

class TestCheckAll:
    def test_check_all_report(self):
        now = time.time()
        rotator = SecretRotator(default_rotation_days=30)
        rotator.register("fresh", last_rotated=now)
        rotator.register("stale", last_rotated=now - 45 * _SECONDS_PER_DAY)

        report = rotator.check_all()
        assert isinstance(report, RotationReport)
        assert report.total == 2
        assert report.overdue_count == 1
        assert report.ok_count == 1
        assert len(report.statuses) == 2
        assert "stale" in report.overdue_secrets

    def test_overdue_list(self):
        now = time.time()
        rotator = SecretRotator(default_rotation_days=30)
        rotator.register("good", last_rotated=now)
        rotator.register("bad", last_rotated=now - 60 * _SECONDS_PER_DAY)

        overdue = rotator.overdue()
        assert "bad" in overdue
        assert "good" not in overdue

    def test_upcoming_list(self):
        now = time.time()
        rotator = SecretRotator(default_rotation_days=30)
        # Due in ~5 days
        rotator.register("soon", last_rotated=now - 25 * _SECONDS_PER_DAY)
        # Due in ~20 days
        rotator.register("later", last_rotated=now - 10 * _SECONDS_PER_DAY)

        upcoming = rotator.upcoming(days=7)
        assert "soon" in upcoming
        assert "later" not in upcoming


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_empty(self):
        rotator = SecretRotator()
        rotator.register("key")
        events = rotator.history("key")
        assert events == []

    def test_history_after_rotations(self):
        rotator = SecretRotator()
        rotator.register("key")
        rotator.rotate("key")
        rotator.rotate("key")
        rotator.rotate("key")

        events = rotator.history("key")
        assert len(events) == 3

        for event in events:
            assert event.name == "key"
            assert event.rotated_at > 0

        # Each event after the first should reference the previous rotation
        assert events[0].previous_rotation is not None
        assert events[1].previous_rotation == events[0].rotated_at


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_secret(self):
        rotator = SecretRotator()
        rotator.register("temp_key")
        assert "temp_key" in rotator.list_secrets()

        rotator.remove("temp_key")
        assert "temp_key" not in rotator.list_secrets()

    def test_remove_missing_raises(self):
        rotator = SecretRotator()
        with pytest.raises(KeyError, match="not registered"):
            rotator.remove("nonexistent")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

class TestList:
    def test_list_secrets(self):
        rotator = SecretRotator()
        rotator.register("alpha")
        rotator.register("beta")
        rotator.register("gamma")

        names = rotator.list_secrets()
        assert len(names) == 3
        assert set(names) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_check_missing_raises(self):
        rotator = SecretRotator()
        with pytest.raises(KeyError, match="not registered"):
            rotator.check("ghost")

    def test_rotate_missing_raises(self):
        rotator = SecretRotator()
        with pytest.raises(KeyError, match="not registered"):
            rotator.rotate("ghost")

    def test_history_missing_raises(self):
        rotator = SecretRotator()
        with pytest.raises(KeyError, match="not registered"):
            rotator.history("ghost")

    def test_default_rotation_days(self):
        rotator = SecretRotator(default_rotation_days=60)
        rotator.register("key")
        status = rotator.check("key")
        assert status.days_until_due <= 60
        assert status.days_until_due >= 58
