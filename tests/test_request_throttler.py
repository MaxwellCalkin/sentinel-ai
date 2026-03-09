"""Tests for adaptive request throttler."""

import pytest
from sentinel.request_throttler import (
    RequestThrottler,
    ThrottleStatus,
    RequestDecision,
    ThrottlerStats,
    LEVEL_NORMAL,
    LEVEL_CAUTIOUS,
    LEVEL_RESTRICTED,
    LEVEL_BLOCKED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Deterministic clock for tests that avoids real sleeps."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_throttler(clock: FakeClock, **kwargs) -> RequestThrottler:
    throttler = RequestThrottler(**kwargs)
    throttler._current_time = clock
    return throttler


# ---------------------------------------------------------------------------
# Basic request allow/deny
# ---------------------------------------------------------------------------

class TestBasicRequests:
    def test_allow_within_limit(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=10, window_seconds=60)
        decision = t.check_request()
        assert isinstance(decision, RequestDecision)
        assert decision.allowed
        assert decision.throttle_level == LEVEL_NORMAL
        assert isinstance(decision.reason, str)
        assert decision.wait_seconds == 0.0

    def test_deny_when_rate_exceeded(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=3, window_seconds=60)
        for _ in range(3):
            assert t.check_request().allowed
        decision = t.check_request()
        assert not decision.allowed
        assert decision.wait_seconds > 0


# ---------------------------------------------------------------------------
# Throttle levels from incidents
# ---------------------------------------------------------------------------

class TestThrottleLevels:
    def test_normal_with_no_incidents(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=100, window_seconds=60)
        assert t.status().level == LEVEL_NORMAL

    def test_cautious_at_threshold(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, cautious_threshold=3
        )
        for _ in range(3):
            t.report_incident("injection")
        assert t.status().level == LEVEL_CAUTIOUS
        assert t.status().allowed_rate == 50.0

    def test_restricted_at_threshold(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, restricted_threshold=7
        )
        for _ in range(7):
            t.report_incident("abuse")
        assert t.status().level == LEVEL_RESTRICTED
        assert t.status().allowed_rate == 25.0

    def test_blocked_at_threshold(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, blocked_threshold=12
        )
        for _ in range(12):
            t.report_incident("critical")
        status = t.status()
        assert status.level == LEVEL_BLOCKED
        assert status.allowed_rate == 0.0

    def test_blocked_denies_all_requests(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, blocked_threshold=5
        )
        for _ in range(5):
            t.report_incident("attack")
        decision = t.check_request()
        assert not decision.allowed
        assert decision.throttle_level == LEVEL_BLOCKED


# ---------------------------------------------------------------------------
# Reduced rate under cautious/restricted
# ---------------------------------------------------------------------------

class TestReducedRate:
    def test_cautious_halves_allowed_rate(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=10, window_seconds=60, cautious_threshold=2
        )
        t.report_incident("a")
        t.report_incident("b")
        # Allowed rate is 5 now; consume 5
        for _ in range(5):
            assert t.check_request().allowed
        decision = t.check_request()
        assert not decision.allowed
        assert decision.throttle_level == LEVEL_CAUTIOUS

    def test_restricted_quarters_allowed_rate(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=8, window_seconds=60, restricted_threshold=2
        )
        t.report_incident("x")
        t.report_incident("y")
        # Allowed rate is 2 (8 * 0.25)
        assert t.check_request().allowed
        assert t.check_request().allowed
        decision = t.check_request()
        assert not decision.allowed
        assert decision.throttle_level == LEVEL_RESTRICTED


# ---------------------------------------------------------------------------
# Auto-recovery via cool-down
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_recover_one_level_after_cooldown(self):
        clock = FakeClock()
        t = _make_throttler(
            clock,
            base_rate=100,
            window_seconds=300,
            restricted_threshold=3,
            cooldown_seconds=60,
        )
        for _ in range(3):
            t.report_incident("test")
        assert t.status().level == LEVEL_RESTRICTED
        clock.advance(61)
        assert t.status().level == LEVEL_CAUTIOUS

    def test_recover_to_normal_after_enough_cooldown(self):
        clock = FakeClock()
        t = _make_throttler(
            clock,
            base_rate=100,
            window_seconds=600,
            blocked_threshold=5,
            cooldown_seconds=30,
        )
        for _ in range(5):
            t.report_incident("severe")
        assert t.status().level == LEVEL_BLOCKED
        # 3 cooldown periods to go from blocked -> restricted -> cautious -> normal
        clock.advance(91)
        assert t.status().level == LEVEL_NORMAL

    def test_no_recovery_before_cooldown_elapsed(self):
        clock = FakeClock()
        t = _make_throttler(
            clock,
            base_rate=100,
            window_seconds=300,
            cautious_threshold=2,
            cooldown_seconds=60,
        )
        t.report_incident("a")
        t.report_incident("b")
        assert t.status().level == LEVEL_CAUTIOUS
        clock.advance(30)
        assert t.status().level == LEVEL_CAUTIOUS


# ---------------------------------------------------------------------------
# Sliding window expiry
# ---------------------------------------------------------------------------

class TestWindowExpiry:
    def test_incidents_expire_outside_window(self):
        clock = FakeClock()
        t = _make_throttler(
            clock,
            base_rate=100,
            window_seconds=60,
            cautious_threshold=3,
        )
        for _ in range(3):
            t.report_incident("test")
        assert t.status().level == LEVEL_CAUTIOUS
        clock.advance(61)
        assert t.status().level == LEVEL_NORMAL
        assert t.status().incidents_in_window == 0

    def test_requests_expire_outside_window(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=2, window_seconds=10)
        t.check_request()
        t.check_request()
        assert not t.check_request().allowed
        clock.advance(11)
        assert t.check_request().allowed


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_dataclass(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=10, window_seconds=60)
        t.check_request()
        stats = t.stats()
        assert isinstance(stats, ThrottlerStats)

    def test_stats_tracks_totals(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=2, window_seconds=60)
        t.report_incident("a")
        t.check_request()
        t.check_request()
        t.check_request()  # denied (rate=2, no incidents escalation since threshold=3)
        stats = t.stats()
        assert stats.total_requests == 3
        assert stats.denied_count == 1
        assert stats.total_incidents == 1
        assert stats.deny_rate == pytest.approx(1 / 3, abs=0.01)

    def test_stats_current_level(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, cautious_threshold=1
        )
        t.report_incident("x")
        assert t.stats().current_level == LEVEL_CAUTIOUS


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_returns_dataclass(self):
        clock = FakeClock()
        t = _make_throttler(clock, base_rate=50, window_seconds=30)
        status = t.status()
        assert isinstance(status, ThrottleStatus)
        assert status.base_rate == 50
        assert status.window_seconds == 30.0

    def test_status_reflects_incidents(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, cautious_threshold=2
        )
        t.report_incident("a")
        t.report_incident("b")
        status = t.status()
        assert status.incidents_in_window == 2
        assert status.level == LEVEL_CAUTIOUS
        assert status.allowed_rate == 50.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_state(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=10, window_seconds=60, cautious_threshold=1
        )
        t.report_incident("x")
        t.check_request()
        t.check_request()
        t.reset()
        stats = t.stats()
        assert stats.total_requests == 0
        assert stats.denied_count == 0
        assert stats.total_incidents == 0
        assert t.status().level == LEVEL_NORMAL


# ---------------------------------------------------------------------------
# Incident categories
# ---------------------------------------------------------------------------

class TestIncidentCategories:
    def test_different_categories_all_count(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, cautious_threshold=3
        )
        t.report_incident("injection")
        t.report_incident("abuse")
        t.report_incident("pii_leak")
        assert t.status().level == LEVEL_CAUTIOUS

    def test_default_category(self):
        clock = FakeClock()
        t = _make_throttler(
            clock, base_rate=100, window_seconds=60, cautious_threshold=1
        )
        t.report_incident()
        assert t.status().level == LEVEL_CAUTIOUS
