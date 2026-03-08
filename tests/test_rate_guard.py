"""Tests for rate-based abuse detection."""

import time
from unittest.mock import patch

from sentinel.rate_guard import RateGuard, RateVerdict
from sentinel.core import ScanResult, Finding, RiskLevel


def _make_result(risk: RiskLevel = RiskLevel.NONE, findings: list | None = None) -> ScanResult:
    return ScanResult(
        text="test",
        risk=risk,
        findings=findings or [],
        blocked=risk >= RiskLevel.HIGH,
    )


def _make_finding(risk: RiskLevel = RiskLevel.HIGH, category: str = "prompt_injection") -> Finding:
    return Finding(
        scanner="test",
        category=category,
        description="test finding",
        risk=risk,
    )


class TestRateGuardBasic:
    def test_clean_input_not_throttled(self):
        rg = RateGuard()
        result = _make_result()
        verdict = rg.check("user-1", result)
        assert not verdict.throttled
        assert verdict.violation_count == 0

    def test_single_violation_not_throttled(self):
        rg = RateGuard(max_violations=3)
        result = _make_result(RiskLevel.HIGH, [_make_finding()])
        verdict = rg.check("user-1", result)
        assert not verdict.throttled
        assert verdict.violation_count == 1

    def test_throttled_after_max_violations(self):
        rg = RateGuard(max_violations=3)
        finding = _make_finding()
        for _ in range(3):
            rg.check("user-1", _make_result(RiskLevel.HIGH, [finding]))
        verdict = rg.check("user-1", _make_result(RiskLevel.HIGH, [finding]))
        # After 3 violations, the 3rd check triggers throttle
        # (3 findings from first 3 checks)
        assert verdict.throttled
        assert verdict.risk == RiskLevel.CRITICAL

    def test_different_users_independent(self):
        rg = RateGuard(max_violations=2)
        finding = _make_finding()
        rg.check("user-1", _make_result(RiskLevel.HIGH, [finding]))
        rg.check("user-1", _make_result(RiskLevel.HIGH, [finding]))
        v1 = rg.check("user-1", _make_result(RiskLevel.HIGH, [finding]))
        v2 = rg.check("user-2", _make_result(RiskLevel.HIGH, [finding]))
        assert v1.throttled
        assert not v2.throttled


class TestRiskEscalation:
    def test_escalation_after_threshold(self):
        rg = RateGuard(escalation_threshold=2, max_violations=10)
        finding = _make_finding()
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        # Now at 2 violations, low-risk input gets escalated
        verdict = rg.check("u1", _make_result(RiskLevel.LOW))
        assert verdict.escalated_risk
        assert verdict.risk == RiskLevel.MEDIUM

    def test_no_escalation_below_threshold(self):
        rg = RateGuard(escalation_threshold=3, max_violations=10)
        finding = _make_finding()
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        verdict = rg.check("u1", _make_result(RiskLevel.LOW))
        assert not verdict.escalated_risk
        assert verdict.risk == RiskLevel.LOW

    def test_escalation_chain(self):
        rg = RateGuard(escalation_threshold=2, max_violations=10)
        finding = _make_finding()
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        # NONE -> LOW
        v = rg.check("u1", _make_result(RiskLevel.NONE))
        assert v.risk == RiskLevel.LOW
        assert v.escalated_risk


class TestWindowExpiry:
    def test_violations_expire_after_window(self):
        rg = RateGuard(window_seconds=10, max_violations=2, cooldown_seconds=5)
        finding = _make_finding()

        with patch("sentinel.rate_guard.time") as mock_time:
            mock_time.time.return_value = 1000.0
            rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
            rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))

            # Move past both window (10s) and cooldown (5s)
            mock_time.time.return_value = 1020.0
            verdict = rg.check("u1", _make_result(RiskLevel.LOW))
            assert not verdict.throttled
            assert verdict.violation_count == 0


class TestThrottleCooldown:
    def test_throttle_persists_during_cooldown(self):
        rg = RateGuard(max_violations=2, cooldown_seconds=30)
        finding = _make_finding()

        with patch("sentinel.rate_guard.time") as mock_time:
            mock_time.time.return_value = 1000.0
            rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
            rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))

            # Still within cooldown
            mock_time.time.return_value = 1010.0
            verdict = rg.check("u1", _make_result(RiskLevel.NONE))
            assert verdict.throttled

    def test_throttle_lifts_after_cooldown(self):
        rg = RateGuard(max_violations=2, cooldown_seconds=30, window_seconds=5)
        finding = _make_finding()

        with patch("sentinel.rate_guard.time") as mock_time:
            mock_time.time.return_value = 1000.0
            rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
            rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))

            # Past cooldown AND window
            mock_time.time.return_value = 1040.0
            verdict = rg.check("u1", _make_result(RiskLevel.NONE))
            assert not verdict.throttled


class TestReset:
    def test_reset_user(self):
        rg = RateGuard(max_violations=2)
        finding = _make_finding()
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        rg.reset("u1")
        verdict = rg.check("u1", _make_result(RiskLevel.NONE))
        assert not verdict.throttled
        assert verdict.violation_count == 0

    def test_reset_all(self):
        rg = RateGuard(max_violations=2)
        finding = _make_finding()
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        rg.check("u2", _make_result(RiskLevel.HIGH, [finding]))
        rg.reset_all()
        assert rg.get_violation_count("u1") == 0
        assert rg.get_violation_count("u2") == 0

    def test_get_violation_count(self):
        rg = RateGuard()
        finding = _make_finding()
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        rg.check("u1", _make_result(RiskLevel.HIGH, [finding]))
        assert rg.get_violation_count("u1") == 2
        assert rg.get_violation_count("u2") == 0


class TestLowRiskNotCounted:
    def test_low_risk_findings_not_counted(self):
        rg = RateGuard(max_violations=2)
        low_finding = _make_finding(risk=RiskLevel.LOW)
        rg.check("u1", _make_result(RiskLevel.LOW, [low_finding]))
        rg.check("u1", _make_result(RiskLevel.LOW, [low_finding]))
        rg.check("u1", _make_result(RiskLevel.LOW, [low_finding]))
        # Low risk findings don't count toward violations
        assert rg.get_violation_count("u1") == 0
