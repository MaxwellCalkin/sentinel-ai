"""Tests for the risk-aware model router."""

import pytest
from unittest.mock import MagicMock
from sentinel.risk_router import RiskRouter, ModelTier, RouteDecision, RoutingRule
from sentinel.core import ScanResult, RiskLevel, Finding


MODELS = {
    ModelTier.LOW: "claude-haiku-4-5-20251001",
    ModelTier.MEDIUM: "claude-sonnet-4-6",
    ModelTier.HIGH: "claude-opus-4-6",
}


def make_guard(risk_level=RiskLevel.NONE, findings=None):
    """Create a mock guard that returns a preset scan result."""
    guard = MagicMock()
    result = ScanResult(
        text="",
        risk=risk_level,
        blocked=risk_level >= RiskLevel.HIGH,
        findings=findings or [],
    )
    guard.scan.return_value = result
    return guard


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

class TestBasicRouting:
    def test_safe_content_routes_low(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("What's the weather?")
        assert decision.tier == ModelTier.LOW
        assert decision.model == "claude-haiku-4-5-20251001"
        assert not decision.escalated

    def test_low_risk_routes_low(self):
        guard = make_guard(RiskLevel.LOW)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("some text")
        assert decision.tier == ModelTier.LOW

    def test_medium_risk_routes_medium(self):
        guard = make_guard(RiskLevel.MEDIUM)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("some text")
        assert decision.tier == ModelTier.MEDIUM
        assert decision.model == "claude-sonnet-4-6"
        assert decision.escalated

    def test_high_risk_routes_high(self):
        guard = make_guard(RiskLevel.HIGH)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("risky text")
        assert decision.tier == ModelTier.HIGH
        assert decision.model == "claude-opus-4-6"
        assert decision.escalated

    def test_critical_risk_routes_high(self):
        guard = make_guard(RiskLevel.CRITICAL)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("very risky")
        assert decision.tier == ModelTier.HIGH
        assert decision.escalated


# ---------------------------------------------------------------------------
# Route decision properties
# ---------------------------------------------------------------------------

class TestRouteDecision:
    def test_risk_level_preserved(self):
        guard = make_guard(RiskLevel.MEDIUM)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("test")
        assert decision.risk == RiskLevel.MEDIUM

    def test_scan_result_preserved(self):
        guard = make_guard(RiskLevel.HIGH)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("test")
        assert decision.scan_result is not None
        assert decision.scan_result.risk == RiskLevel.HIGH

    def test_reasons_populated(self):
        guard = make_guard(RiskLevel.HIGH)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("test")
        assert len(decision.reasons) > 0
        assert "high" in decision.reasons[0].lower()

    def test_no_reasons_for_safe(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        decision = router.route("safe text")
        assert len(decision.reasons) == 0


# ---------------------------------------------------------------------------
# Custom risk mapping
# ---------------------------------------------------------------------------

class TestCustomRiskMapping:
    def test_custom_map(self):
        guard = make_guard(RiskLevel.LOW)
        custom_map = {
            RiskLevel.CRITICAL: ModelTier.HIGH,
            RiskLevel.HIGH: ModelTier.HIGH,
            RiskLevel.MEDIUM: ModelTier.HIGH,  # escalate medium to high
            RiskLevel.LOW: ModelTier.MEDIUM,    # escalate low to medium
            RiskLevel.NONE: ModelTier.LOW,
        }
        router = RiskRouter(MODELS, guard=guard, risk_map=custom_map)
        decision = router.route("test")
        assert decision.tier == ModelTier.MEDIUM

    def test_all_to_high(self):
        guard = make_guard(RiskLevel.NONE)
        all_high = {level: ModelTier.HIGH for level in RiskLevel}
        router = RiskRouter(MODELS, guard=guard, risk_map=all_high)
        decision = router.route("safe text")
        assert decision.tier == ModelTier.HIGH


# ---------------------------------------------------------------------------
# Custom routing rules
# ---------------------------------------------------------------------------

class TestCustomRules:
    def test_rule_escalates(self):
        guard = make_guard(RiskLevel.NONE)
        rule = RoutingRule(
            name="long_text",
            check=lambda text, scan: len(text) > 10,
            tier=ModelTier.MEDIUM,
            reason="Text is long",
        )
        router = RiskRouter(MODELS, guard=guard, rules=[rule])
        decision = router.route("This is a longer text that triggers the rule")
        assert decision.tier == ModelTier.MEDIUM
        assert any("long_text" in r for r in decision.reasons)

    def test_rule_does_not_downgrade(self):
        guard = make_guard(RiskLevel.HIGH)
        rule = RoutingRule(
            name="downgrade",
            check=lambda text, scan: True,
            tier=ModelTier.LOW,  # tries to downgrade
            reason="Should not work",
        )
        router = RiskRouter(MODELS, guard=guard, rules=[rule])
        decision = router.route("risky text")
        assert decision.tier == ModelTier.HIGH  # stays high

    def test_rule_not_triggered(self):
        guard = make_guard(RiskLevel.NONE)
        rule = RoutingRule(
            name="never",
            check=lambda text, scan: False,
            tier=ModelTier.HIGH,
            reason="Never triggers",
        )
        router = RiskRouter(MODELS, guard=guard, rules=[rule])
        decision = router.route("test")
        assert decision.tier == ModelTier.LOW

    def test_multiple_rules(self):
        guard = make_guard(RiskLevel.NONE)
        rules = [
            RoutingRule("r1", lambda t, s: True, ModelTier.MEDIUM, "R1"),
            RoutingRule("r2", lambda t, s: True, ModelTier.HIGH, "R2"),
        ]
        router = RiskRouter(MODELS, guard=guard, rules=rules)
        decision = router.route("test")
        assert decision.tier == ModelTier.HIGH
        assert len(decision.reasons) >= 2


# ---------------------------------------------------------------------------
# Batch routing
# ---------------------------------------------------------------------------

class TestBatchRouting:
    def test_batch_route(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        decisions = router.route_batch(["hello", "world", "test"])
        assert len(decisions) == 3
        assert all(d.tier == ModelTier.LOW for d in decisions)

    def test_batch_empty(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        decisions = router.route_batch([])
        assert len(decisions) == 0


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------

class TestCostSummary:
    def test_cost_summary(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        decisions = [
            RouteDecision("m1", ModelTier.LOW, RiskLevel.NONE, MagicMock()),
            RouteDecision("m2", ModelTier.LOW, RiskLevel.NONE, MagicMock()),
            RouteDecision("m3", ModelTier.MEDIUM, RiskLevel.MEDIUM, MagicMock()),
            RouteDecision("m4", ModelTier.HIGH, RiskLevel.HIGH, MagicMock()),
        ]
        summary = router.cost_summary(decisions)
        assert summary["total"] == 4
        assert summary["by_tier"]["low"] == 2
        assert summary["by_tier"]["medium"] == 1
        assert summary["by_tier"]["high"] == 1
        assert summary["escalation_rate"] == 0.5

    def test_cost_summary_empty(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        summary = router.cost_summary([])
        assert summary["total"] == 0
        assert summary["escalation_rate"] == 0.0

    def test_all_escalated(self):
        guard = make_guard(RiskLevel.NONE)
        router = RiskRouter(MODELS, guard=guard)
        decisions = [
            RouteDecision("m1", ModelTier.HIGH, RiskLevel.HIGH, MagicMock()),
            RouteDecision("m2", ModelTier.MEDIUM, RiskLevel.MEDIUM, MagicMock()),
        ]
        summary = router.cost_summary(decisions)
        assert summary["escalation_rate"] == 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_tier_raises(self):
        with pytest.raises(ValueError, match="No model configured"):
            RiskRouter({ModelTier.LOW: "haiku", ModelTier.MEDIUM: "sonnet"})

    def test_all_tiers_required(self):
        with pytest.raises(ValueError):
            RiskRouter({ModelTier.HIGH: "opus"})
