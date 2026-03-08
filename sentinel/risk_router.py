"""Risk-aware model routing for LLM applications.

Routes prompts to appropriate models based on safety risk assessment.
High-risk prompts are routed to more capable/safer models, while low-risk
prompts can use cheaper/faster models, optimizing cost without compromising
safety.

Usage:
    from sentinel.risk_router import RiskRouter, ModelTier

    router = RiskRouter({
        ModelTier.HIGH: "claude-opus-4-6",
        ModelTier.MEDIUM: "claude-sonnet-4-6",
        ModelTier.LOW: "claude-haiku-4-5-20251001",
    })

    route = router.route("What's the weather today?")
    print(route.model)  # "claude-haiku-4-5-20251001"
    print(route.tier)   # ModelTier.LOW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from sentinel.core import SentinelGuard, ScanResult, RiskLevel


class ModelTier(Enum):
    """Model routing tiers."""
    LOW = 0       # Fast/cheap models for safe content
    MEDIUM = 1    # Balanced models for moderate risk
    HIGH = 2      # Most capable models for high-risk content


@dataclass
class RouteDecision:
    """Result of routing a prompt to a model."""
    model: str
    tier: ModelTier
    risk: RiskLevel
    scan_result: ScanResult
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def escalated(self) -> bool:
        """True if the prompt was routed to a higher tier than LOW."""
        return self.tier != ModelTier.LOW


@dataclass
class RoutingRule:
    """A rule that can escalate routing to a higher tier."""
    name: str
    check: callable
    tier: ModelTier
    reason: str


class RiskRouter:
    """Route prompts to models based on risk assessment.

    Uses SentinelGuard to assess content risk, then maps risk levels
    to model tiers. Supports custom routing rules for domain-specific
    escalation logic.
    """

    # Default risk-to-tier mapping
    _DEFAULT_RISK_MAP: dict[RiskLevel, ModelTier] = {
        RiskLevel.CRITICAL: ModelTier.HIGH,
        RiskLevel.HIGH: ModelTier.HIGH,
        RiskLevel.MEDIUM: ModelTier.MEDIUM,
        RiskLevel.LOW: ModelTier.LOW,
        RiskLevel.NONE: ModelTier.LOW,
    }

    def __init__(
        self,
        models: dict[ModelTier, str],
        guard: SentinelGuard | None = None,
        risk_map: dict[RiskLevel, ModelTier] | None = None,
        default_tier: ModelTier = ModelTier.LOW,
        rules: Sequence[RoutingRule] | None = None,
    ):
        """
        Args:
            models: Mapping of model tiers to model identifiers.
            guard: SentinelGuard instance for risk assessment.
            risk_map: Custom risk-level to tier mapping.
            default_tier: Default tier when no risk is detected.
            rules: Additional routing rules for custom escalation.
        """
        self._models = dict(models)
        self._guard = guard or SentinelGuard.default()
        self._risk_map = risk_map or dict(self._DEFAULT_RISK_MAP)
        self._default_tier = default_tier
        self._rules = list(rules) if rules else []

        # Validate all tiers have models
        for tier in ModelTier:
            if tier not in self._models:
                raise ValueError(f"No model configured for tier {tier.name}")

    def route(self, text: str) -> RouteDecision:
        """Assess risk and route to appropriate model."""
        scan = self._guard.scan(text)

        # Determine tier from risk level
        tier = self._risk_map.get(scan.risk, self._default_tier)
        reasons: list[str] = []

        if scan.risk != RiskLevel.NONE:
            reasons.append(
                f"Risk level {scan.risk.value} detected"
            )

        # Apply custom rules (can only escalate, never downgrade)
        for rule in self._rules:
            if rule.tier.value > tier.value:
                # Custom rule check
                if rule.check(text, scan):
                    tier = rule.tier
                    reasons.append(f"Rule '{rule.name}': {rule.reason}")

        model = self._models[tier]

        return RouteDecision(
            model=model,
            tier=tier,
            risk=scan.risk,
            scan_result=scan,
            reasons=reasons,
        )

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        """Route multiple prompts and return decisions."""
        return [self.route(text) for text in texts]

    def cost_summary(self, decisions: list[RouteDecision]) -> dict[str, Any]:
        """Summarize routing decisions for cost analysis."""
        tier_counts: dict[str, int] = {t.name.lower(): 0 for t in ModelTier}
        for d in decisions:
            tier_counts[d.tier.name.lower()] += 1

        return {
            "total": len(decisions),
            "by_tier": tier_counts,
            "escalation_rate": (
                sum(1 for d in decisions if d.escalated) / len(decisions)
                if decisions
                else 0.0
            ),
        }
