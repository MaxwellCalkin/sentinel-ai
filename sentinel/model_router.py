"""Intelligent model routing based on content sensitivity.

Route prompts to appropriate models based on detected risk level,
content sensitivity, cost constraints, and custom rules.

Usage:
    from sentinel.model_router import ModelRouter

    router = ModelRouter()
    router.add_route("high_risk", model="claude-opus", condition=lambda t: "secret" in t)
    decision = router.route("Tell me about public data")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RouteRule:
    """A single routing rule."""
    name: str
    model: str
    condition: Callable[[str], bool]
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    """Result of routing a prompt."""
    model: str
    rule_name: str
    matched: bool
    fallback: bool = False
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RouterStats:
    """Statistics for the router."""
    total_routed: int
    route_counts: dict[str, int]
    fallback_count: int


class ModelRouter:
    """Route prompts to appropriate models based on rules.

    Evaluates rules in priority order (highest first) and
    routes to the first matching model. Falls back to a
    default model if no rules match.
    """

    def __init__(self, default_model: str = "claude-sonnet") -> None:
        """
        Args:
            default_model: Model to use when no rules match.
        """
        self._default = default_model
        self._rules: list[RouteRule] = []
        self._keyword_rules: list[tuple[str, str, list[str]]] = []  # (name, model, keywords)
        self._stats: dict[str, int] = {}
        self._fallback_count = 0
        self._total = 0

    def add_route(
        self,
        name: str,
        model: str,
        condition: Callable[[str], bool],
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a routing rule.

        Args:
            name: Rule name.
            model: Target model identifier.
            condition: Function(text) -> True if rule matches.
            priority: Higher = evaluated first.
            metadata: Additional rule metadata.
        """
        self._rules.append(RouteRule(
            name=name,
            model=model,
            condition=condition,
            priority=priority,
            metadata=metadata or {},
        ))
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_keyword_route(
        self,
        name: str,
        model: str,
        keywords: list[str],
        case_sensitive: bool = False,
    ) -> None:
        """Add a keyword-based routing rule.

        Args:
            name: Rule name.
            model: Target model.
            keywords: Keywords that trigger this route.
            case_sensitive: Whether matching is case-sensitive.
        """
        if not case_sensitive:
            keywords = [k.lower() for k in keywords]

        def condition(text: str) -> bool:
            t = text if case_sensitive else text.lower()
            return any(k in t for k in keywords)

        self.add_route(name, model, condition)

    def add_sensitivity_route(
        self,
        name: str,
        model: str,
        patterns: list[str],
    ) -> None:
        """Route based on sensitive content patterns (regex).

        Args:
            name: Rule name.
            model: Target model for sensitive content.
            patterns: Regex patterns indicating sensitivity.
        """
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        def condition(text: str) -> bool:
            return any(r.search(text) for r in compiled)

        self.add_route(name, model, condition, priority=10)

    def route(self, text: str) -> RouteDecision:
        """Route a prompt to the appropriate model.

        Args:
            text: The prompt text to route.

        Returns:
            RouteDecision with selected model and matching rule.
        """
        self._total += 1

        for rule in self._rules:
            try:
                if rule.condition(text):
                    self._stats[rule.name] = self._stats.get(rule.name, 0) + 1
                    return RouteDecision(
                        model=rule.model,
                        rule_name=rule.name,
                        matched=True,
                    )
            except Exception:
                continue

        self._fallback_count += 1
        return RouteDecision(
            model=self._default,
            rule_name="default",
            matched=False,
            fallback=True,
        )

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        """Route multiple prompts."""
        return [self.route(t) for t in texts]

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def stats(self) -> RouterStats:
        return RouterStats(
            total_routed=self._total,
            route_counts=dict(self._stats),
            fallback_count=self._fallback_count,
        )

    def reset_stats(self) -> None:
        self._stats.clear()
        self._fallback_count = 0
        self._total = 0
