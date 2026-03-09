"""Tests for usage quota."""

import pytest
from sentinel.usage_quota import UsageQuota, QuotaStatus


class TestLimits:
    def test_set_limit(self):
        q = UsageQuota()
        q.set_limit("user-1", max_requests=100, window_hours=24)
        status = q.check("user-1")
        assert status.within_quota
        assert status.requests_remaining == 100

    def test_default_limits(self):
        q = UsageQuota(default_max_requests=50)
        status = q.check("user-1")
        assert status.requests_remaining == 50


class TestUsageTracking:
    def test_record_usage(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=10)
        q.record("u1", requests=3)
        status = q.check("u1")
        assert status.requests_used == 3
        assert status.requests_remaining == 7

    def test_token_tracking(self):
        q = UsageQuota()
        q.set_limit("u1", max_tokens=1000)
        q.record("u1", tokens=500)
        status = q.check("u1")
        assert status.tokens_used == 500
        assert status.tokens_remaining == 500

    def test_consume(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=5)
        status = q.consume("u1")
        assert status.requests_used == 1
        assert status.within_quota


class TestQuotaEnforcement:
    def test_quota_exceeded_requests(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=3)
        q.record("u1", requests=3)
        status = q.check("u1")
        assert not status.within_quota

    def test_quota_exceeded_tokens(self):
        q = UsageQuota()
        q.set_limit("u1", max_tokens=100)
        q.record("u1", tokens=150)
        status = q.check("u1")
        assert not status.within_quota

    def test_within_quota(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=10, max_tokens=1000)
        q.record("u1", requests=5, tokens=500)
        status = q.check("u1")
        assert status.within_quota


class TestUtilization:
    def test_utilization(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=10)
        q.record("u1", requests=5)
        status = q.check("u1")
        assert status.utilization == 0.5

    def test_zero_utilization(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=10)
        status = q.check("u1")
        assert status.utilization == 0.0


class TestReset:
    def test_reset(self):
        q = UsageQuota()
        q.record("u1", requests=5)
        assert q.reset("u1")
        status = q.check("u1")
        assert status.requests_used == 0

    def test_reset_missing(self):
        q = UsageQuota()
        assert not q.reset("nonexistent")


class TestEntities:
    def test_list_entities(self):
        q = UsageQuota()
        q.set_limit("a")
        q.record("b")
        entities = q.list_entities()
        assert "a" in entities
        assert "b" in entities


class TestStructure:
    def test_status_structure(self):
        q = UsageQuota()
        q.set_limit("u1", max_requests=10)
        status = q.check("u1")
        assert isinstance(status, QuotaStatus)
        assert status.entity_id == "u1"
        assert isinstance(status.within_quota, bool)
        assert 0.0 <= status.utilization <= 1.0
