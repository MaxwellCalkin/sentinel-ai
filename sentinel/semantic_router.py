"""Semantic intent router for LLM prompts.

Route prompts to appropriate handlers based on semantic
intent detection using weighted keyword and pattern matching.

Usage:
    from sentinel.semantic_router import SemanticRouter

    router = SemanticRouter()
    router.add_route("coding", handler="code_model",
                     keywords=["code", "program", "function", "debug"])
    result = router.route("Write a Python function to sort a list")
    print(result.route)  # "coding"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouteConfig:
    """Configuration for a semantic route."""
    name: str
    handler: str
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResult:
    """Result of routing a prompt to a handler."""
    route: str
    handler: str
    confidence: float
    scores: dict[str, float]
    matched_keywords: list[str]


class SemanticRouter:
    """Route prompts to handlers based on semantic intent.

    Uses weighted keyword and pattern matching to determine the
    best route for a given prompt. Supports batch routing and
    configurable default fallback.
    """

    def __init__(
        self,
        default_route: str = "general",
        default_handler: str = "default",
    ) -> None:
        self._routes: list[RouteConfig] = []
        self._default_route = default_route
        self._default_handler = default_handler

    def add_route(
        self,
        name: str,
        handler: str,
        keywords: list[str] | None = None,
        patterns: list[str] | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a route with keywords, patterns, and optional weight."""
        self._routes.append(
            RouteConfig(
                name=name,
                handler=handler,
                keywords=keywords or [],
                patterns=patterns or [],
                weight=weight,
                metadata=metadata or {},
            )
        )

    def route(self, text: str) -> RouteResult:
        """Route a prompt to the best matching handler."""
        if not self._routes:
            return RouteResult(
                route=self._default_route,
                handler=self._default_handler,
                confidence=0.0,
                scores={},
                matched_keywords=[],
            )

        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        scores: dict[str, float] = {}
        all_matched: dict[str, list[str]] = {}

        for rc in self._routes:
            score = 0.0
            matched: list[str] = []

            # Keyword matching (1 point each)
            for kw in rc.keywords:
                if kw.lower() in words:
                    score += 1.0
                    matched.append(kw)

            # Pattern matching (2 points each)
            for pat in rc.patterns:
                if re.search(pat, text, re.IGNORECASE):
                    score += 2.0
                    matched.append(f"pattern:{pat}")

            score *= rc.weight

            # Normalize by total number of matchers
            total_matchers = len(rc.keywords) + len(rc.patterns)
            if total_matchers > 0:
                score = score / total_matchers

            scores[rc.name] = round(score, 4)
            all_matched[rc.name] = matched

        best_name = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_name]

        if best_score == 0:
            return RouteResult(
                route=self._default_route,
                handler=self._default_handler,
                confidence=0.0,
                scores=scores,
                matched_keywords=[],
            )

        best_config = next(r for r in self._routes if r.name == best_name)
        confidence = min(1.0, best_score)

        return RouteResult(
            route=best_name,
            handler=best_config.handler,
            confidence=round(confidence, 4),
            scores=scores,
            matched_keywords=all_matched[best_name],
        )

    def route_batch(self, texts: list[str]) -> list[RouteResult]:
        """Route multiple prompts and return results."""
        return [self.route(t) for t in texts]

    def list_routes(self) -> list[str]:
        """Return names of all configured routes."""
        return [r.name for r in self._routes]
