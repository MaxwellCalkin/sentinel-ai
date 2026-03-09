"""Tests for consent tracking."""

import time
import pytest
from sentinel.consent_tracker import ConsentTracker, ConsentRecord, ConsentStatus


# ---------------------------------------------------------------------------
# Granting consent
# ---------------------------------------------------------------------------

class TestGrant:
    def test_grant_consent(self):
        t = ConsentTracker()
        rec = t.grant("user1", "data_collection")
        assert rec.granted
        assert rec.user_id == "user1"
        assert rec.purpose == "data_collection"

    def test_has_consent(self):
        t = ConsentTracker()
        t.grant("user1", "analytics")
        assert t.has_consent("user1", "analytics")

    def test_no_consent(self):
        t = ConsentTracker()
        assert not t.has_consent("user1", "analytics")

    def test_grant_with_metadata(self):
        t = ConsentTracker()
        rec = t.grant("user1", "marketing", metadata={"source": "signup"})
        assert rec.metadata["source"] == "signup"


# ---------------------------------------------------------------------------
# Revoking consent
# ---------------------------------------------------------------------------

class TestRevoke:
    def test_revoke(self):
        t = ConsentTracker()
        t.grant("user1", "analytics")
        assert t.revoke("user1", "analytics")
        assert not t.has_consent("user1", "analytics")

    def test_revoke_nonexistent(self):
        t = ConsentTracker()
        assert not t.revoke("user1", "analytics")

    def test_revoke_all(self):
        t = ConsentTracker()
        t.grant("user1", "analytics")
        t.grant("user1", "marketing")
        count = t.revoke_all("user1")
        assert count == 2
        assert not t.has_consent("user1", "analytics")
        assert not t.has_consent("user1", "marketing")

    def test_revoke_all_empty(self):
        t = ConsentTracker()
        assert t.revoke_all("nonexistent") == 0


# ---------------------------------------------------------------------------
# Expiration
# ---------------------------------------------------------------------------

class TestExpiration:
    def test_not_expired(self):
        t = ConsentTracker()
        t.grant("user1", "analytics", expires_in=3600)
        assert t.has_consent("user1", "analytics")

    def test_expired(self):
        t = ConsentTracker()
        t.grant("user1", "analytics", expires_in=0.001)
        time.sleep(0.01)
        assert not t.has_consent("user1", "analytics")

    def test_no_expiry(self):
        t = ConsentTracker()
        rec = t.grant("user1", "analytics")
        assert not rec.expired


# ---------------------------------------------------------------------------
# Required purposes
# ---------------------------------------------------------------------------

class TestRequired:
    def test_all_required_met(self):
        t = ConsentTracker(required_purposes=["analytics", "data_collection"])
        t.grant("user1", "analytics")
        t.grant("user1", "data_collection")
        status = t.check_required("user1")
        assert len(status.missing) == 0

    def test_missing_required(self):
        t = ConsentTracker(required_purposes=["analytics", "data_collection"])
        t.grant("user1", "analytics")
        status = t.check_required("user1")
        assert "data_collection" in status.missing

    def test_no_required(self):
        t = ConsentTracker()
        status = t.check_required("user1")
        assert len(status.missing) == 0


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

class TestUserManagement:
    def test_user_count(self):
        t = ConsentTracker()
        t.grant("user1", "a")
        t.grant("user2", "a")
        assert t.user_count == 2

    def test_get_records(self):
        t = ConsentTracker()
        t.grant("user1", "a")
        t.grant("user1", "b")
        records = t.get_records("user1")
        assert len(records) == 2

    def test_get_records_empty(self):
        t = ConsentTracker()
        assert t.get_records("nonexistent") == []

    def test_delete_user(self):
        t = ConsentTracker()
        t.grant("user1", "analytics")
        assert t.delete_user("user1")
        assert t.user_count == 0
        assert not t.has_consent("user1", "analytics")

    def test_delete_nonexistent(self):
        t = ConsentTracker()
        assert not t.delete_user("nope")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class TestAudit:
    def test_audit_report(self):
        t = ConsentTracker(required_purposes=["analytics"])
        t.grant("user1", "analytics")
        t.grant("user1", "marketing")
        t.revoke("user1", "marketing")
        report = t.audit()
        assert report["total_users"] == 1
        assert report["total_records"] == 2
        assert report["active_consents"] == 1
        assert report["required_purposes"] == ["analytics"]

    def test_audit_empty(self):
        t = ConsentTracker()
        report = t.audit()
        assert report["total_users"] == 0
