"""Tests for token bucket rate limiter."""

import time
import pytest
from sentinel.ratelimit_guard import RateLimitGuard, RateLimitResult, BucketStats


# ---------------------------------------------------------------------------
# Basic allow/deny
# ---------------------------------------------------------------------------

class TestBasicAllow:
    def test_allow_within_limit(self):
        g = RateLimitGuard(requests_per_minute=60)
        result = g.allow("user1")
        assert result.allowed
        assert result.remaining >= 0

    def test_deny_over_limit(self):
        g = RateLimitGuard(requests_per_minute=2, burst=2)
        g.allow("user1")
        g.allow("user1")
        result = g.allow("user1")
        assert not result.allowed
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_remaining_decreases(self):
        g = RateLimitGuard(requests_per_minute=10, burst=10)
        r1 = g.allow("u")
        r2 = g.allow("u")
        assert r2.remaining < r1.remaining

    def test_bucket_name_in_result(self):
        g = RateLimitGuard(requests_per_minute=60)
        result = g.allow("mykey")
        assert result.bucket == "mykey"

    def test_default_global_bucket(self):
        g = RateLimitGuard(requests_per_minute=60)
        result = g.allow()
        assert result.bucket == "global"


# ---------------------------------------------------------------------------
# Per-key isolation
# ---------------------------------------------------------------------------

class TestPerKey:
    def test_separate_buckets(self):
        g = RateLimitGuard(requests_per_minute=2, burst=2)
        g.allow("user1")
        g.allow("user1")
        # user1 exhausted, user2 should still be allowed
        assert not g.allow("user1").allowed
        assert g.allow("user2").allowed

    def test_bucket_count(self):
        g = RateLimitGuard(requests_per_minute=60)
        g.allow("a")
        g.allow("b")
        g.allow("c")
        assert g.bucket_count == 3


# ---------------------------------------------------------------------------
# Custom limits
# ---------------------------------------------------------------------------

class TestCustomLimits:
    def test_custom_limit_per_key(self):
        g = RateLimitGuard(requests_per_minute=100, burst=100)
        g.set_limit("vip", requests_per_minute=1000, burst=1000)
        # Default users get 100 burst
        for _ in range(100):
            g.allow("normal")
        assert not g.allow("normal").allowed
        # VIP gets 1000
        for _ in range(500):
            assert g.allow("vip").allowed

    def test_custom_limit_reset(self):
        g = RateLimitGuard(requests_per_minute=10, burst=2)
        g.allow("k")
        g.set_limit("k", requests_per_minute=100, burst=100)
        # After set_limit, bucket resets
        result = g.allow("k")
        assert result.allowed
        assert result.remaining >= 98  # 100 - 1


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class TestCost:
    def test_high_cost_request(self):
        g = RateLimitGuard(requests_per_minute=10, burst=10)
        result = g.allow("u", cost=5)
        assert result.allowed
        assert result.remaining == 5

    def test_cost_exceeds_remaining(self):
        g = RateLimitGuard(requests_per_minute=10, burst=10)
        g.allow("u", cost=8)
        result = g.allow("u", cost=5)
        assert not result.allowed


# ---------------------------------------------------------------------------
# Refill
# ---------------------------------------------------------------------------

class TestRefill:
    def test_tokens_refill(self):
        g = RateLimitGuard(requests_per_minute=6000, burst=2)
        g.allow("u")
        g.allow("u")
        assert not g.allow("u").allowed
        # Wait for refill (6000/min = 100/sec, so 0.02s = 2 tokens)
        time.sleep(0.03)
        assert g.allow("u").allowed


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_basic(self):
        g = RateLimitGuard(requests_per_minute=60, burst=10)
        g.allow("u")
        g.allow("u")
        stats = g.stats("u")
        assert isinstance(stats, BucketStats)
        assert stats.requests_total == 2
        assert stats.capacity == 10

    def test_stats_nonexistent(self):
        g = RateLimitGuard(requests_per_minute=60)
        assert g.stats("nope") is None

    def test_denial_rate(self):
        g = RateLimitGuard(requests_per_minute=10, burst=1)
        g.allow("u")  # allowed
        g.allow("u")  # denied
        g.allow("u")  # denied
        stats = g.stats("u")
        assert stats.requests_denied == 2
        assert stats.denial_rate == pytest.approx(2 / 3, abs=0.01)

    def test_all_stats(self):
        g = RateLimitGuard(requests_per_minute=60)
        g.allow("a")
        g.allow("b")
        all_s = g.all_stats()
        assert "a" in all_s
        assert "b" in all_s


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_single(self):
        g = RateLimitGuard(requests_per_minute=10, burst=1)
        g.allow("u")
        g.reset("u")
        assert g.stats("u") is None
        assert g.allow("u").allowed

    def test_reset_all(self):
        g = RateLimitGuard(requests_per_minute=60)
        g.allow("a")
        g.allow("b")
        g.reset()
        assert g.bucket_count == 0


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------

class TestWait:
    def test_wait_gets_token(self):
        g = RateLimitGuard(requests_per_minute=6000, burst=1)
        g.allow("u")  # exhaust
        result = g.wait("u", timeout=0.5)
        assert result.allowed

    def test_wait_timeout(self):
        g = RateLimitGuard(requests_per_minute=1, burst=1)
        g.allow("u")
        result = g.wait("u", timeout=0.05)
        assert not result.allowed
