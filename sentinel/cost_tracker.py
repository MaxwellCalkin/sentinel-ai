"""LLM API cost tracking and budget enforcement.

Tracks token usage and estimated costs per model, enforces budgets,
and provides spend analytics. Essential for production LLM deployments.

Usage:
    from sentinel.cost_tracker import CostTracker

    tracker = CostTracker()
    tracker.set_pricing("claude-sonnet-4-20250514", input_per_1k=0.003, output_per_1k=0.015)
    tracker.record(model="claude-sonnet-4-20250514", input_tokens=500, output_tokens=100)

    print(tracker.total_cost)       # 0.003
    print(tracker.by_model())       # {'claude-sonnet-4-20250514': ModelCost(...)}
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class UsageRecord:
    """A single API call usage record."""
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCost:
    """Aggregated cost for a single model."""
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0


@dataclass
class BudgetAlert:
    """Alert triggered when budget threshold is reached."""
    model: str | None  # None = global
    threshold: float
    current_spend: float
    timestamp: float = field(default_factory=time.time)


class BudgetExceeded(Exception):
    """Raised when a budget limit is exceeded."""
    def __init__(self, model: str | None, limit: float, current: float):
        self.model = model
        self.limit = limit
        self.current = current
        scope = model or "global"
        super().__init__(
            f"Budget exceeded for {scope}: ${current:.4f} >= ${limit:.4f}"
        )


# Default pricing per 1K tokens (approximate as of early 2026)
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-20250514": (0.015, 0.075),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-haiku-3-5-20241022": (0.0008, 0.004),
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
}


class CostTracker:
    """Track LLM API costs and enforce budgets.

    Thread-safe cost tracking with per-model pricing, budget limits,
    alert callbacks, and spend analytics.
    """

    def __init__(self, enforce_budgets: bool = True) -> None:
        self._pricing: dict[str, tuple[float, float]] = dict(_DEFAULT_PRICING)
        self._records: list[UsageRecord] = []
        self._budgets: dict[str | None, float] = {}  # model (or None=global) -> limit
        self._alert_thresholds: dict[str | None, list[float]] = {}
        self._fired_alerts: set[tuple[str | None, float]] = set()
        self._on_alert: Callable[[BudgetAlert], Any] | None = None
        self._enforce = enforce_budgets
        self._lock = threading.Lock()

    def set_pricing(
        self,
        model: str,
        input_per_1k: float,
        output_per_1k: float,
    ) -> None:
        """Set pricing for a model (per 1K tokens)."""
        self._pricing[model] = (input_per_1k, output_per_1k)

    def set_budget(self, limit: float, model: str | None = None) -> None:
        """Set a spending limit. model=None for global budget."""
        self._budgets[model] = limit

    def set_alert_thresholds(
        self,
        thresholds: list[float],
        model: str | None = None,
    ) -> None:
        """Set alert thresholds (e.g., [0.5, 0.8, 0.9] for 50/80/90%)."""
        self._alert_thresholds[model] = sorted(thresholds)

    def on_alert(self, callback: Callable[[BudgetAlert], Any]) -> None:
        """Register a callback for budget alerts."""
        self._on_alert = callback

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record an API call's token usage.

        Returns:
            The UsageRecord created.

        Raises:
            BudgetExceeded: If enforce_budgets=True and a limit would be exceeded.
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        with self._lock:
            # Check budgets before recording
            if self._enforce:
                self._check_budget(model, cost)

            rec = UsageRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                metadata=metadata or {},
            )
            self._records.append(rec)

            # Check alert thresholds
            self._check_alerts(model)

        return rec

    @property
    def total_cost(self) -> float:
        """Total spend across all models."""
        return sum(r.cost for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all calls."""
        return sum(r.input_tokens + r.output_tokens for r in self._records)

    @property
    def call_count(self) -> int:
        """Total number of API calls recorded."""
        return len(self._records)

    def model_cost(self, model: str) -> ModelCost:
        """Get aggregated cost for a specific model."""
        mc = ModelCost(model=model)
        for r in self._records:
            if r.model == model:
                mc.total_input_tokens += r.input_tokens
                mc.total_output_tokens += r.output_tokens
                mc.total_cost += r.cost
                mc.call_count += 1
        return mc

    def by_model(self) -> dict[str, ModelCost]:
        """Get cost breakdown by model."""
        models: dict[str, ModelCost] = {}
        for r in self._records:
            if r.model not in models:
                models[r.model] = ModelCost(model=r.model)
            mc = models[r.model]
            mc.total_input_tokens += r.input_tokens
            mc.total_output_tokens += r.output_tokens
            mc.total_cost += r.cost
            mc.call_count += 1
        return models

    def cost_since(self, since: float) -> float:
        """Total cost since a timestamp."""
        return sum(r.cost for r in self._records if r.timestamp >= since)

    def cost_between(self, start: float, end: float) -> float:
        """Total cost between two timestamps."""
        return sum(
            r.cost for r in self._records
            if start <= r.timestamp <= end
        )

    def budget_remaining(self, model: str | None = None) -> float | None:
        """Remaining budget. Returns None if no budget set."""
        if model not in self._budgets:
            return None
        limit = self._budgets[model]
        if model is None:
            spent = self.total_cost
        else:
            spent = self.model_cost(model).total_cost
        return max(0.0, limit - spent)

    def budget_utilization(self, model: str | None = None) -> float | None:
        """Budget utilization as fraction (0.0 to 1.0+). None if no budget."""
        if model not in self._budgets:
            return None
        limit = self._budgets[model]
        if limit == 0:
            return 1.0
        if model is None:
            spent = self.total_cost
        else:
            spent = self.model_cost(model).total_cost
        return spent / limit

    def summary(self) -> dict[str, Any]:
        """Get a full cost summary."""
        by_model = self.by_model()
        return {
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "models": {
                name: {
                    "input_tokens": mc.total_input_tokens,
                    "output_tokens": mc.total_output_tokens,
                    "cost": round(mc.total_cost, 6),
                    "calls": mc.call_count,
                }
                for name, mc in by_model.items()
            },
            "budgets": {
                (k or "global"): {
                    "limit": v,
                    "remaining": self.budget_remaining(k),
                    "utilization": self.budget_utilization(k),
                }
                for k, v in self._budgets.items()
            },
        }

    def reset(self) -> None:
        """Clear all records and fired alerts."""
        with self._lock:
            self._records.clear()
            self._fired_alerts.clear()

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        pricing = self._pricing.get(model)
        if pricing is None:
            return 0.0
        input_rate, output_rate = pricing
        return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate

    def _check_budget(self, model: str, additional_cost: float) -> None:
        # Check model-specific budget
        if model in self._budgets:
            current = self.model_cost(model).total_cost
            if current + additional_cost > self._budgets[model]:
                raise BudgetExceeded(model, self._budgets[model], current + additional_cost)

        # Check global budget
        if None in self._budgets:
            current = self.total_cost
            if current + additional_cost > self._budgets[None]:
                raise BudgetExceeded(None, self._budgets[None], current + additional_cost)

    def _check_alerts(self, model: str) -> None:
        for scope in [model, None]:
            if scope not in self._alert_thresholds or scope not in self._budgets:
                continue
            utilization = self.budget_utilization(scope)
            if utilization is None:
                continue
            for threshold in self._alert_thresholds[scope]:
                key = (scope, threshold)
                if key not in self._fired_alerts and utilization >= threshold:
                    self._fired_alerts.add(key)
                    if self._on_alert:
                        alert = BudgetAlert(
                            model=scope,
                            threshold=threshold,
                            current_spend=(
                                self.total_cost if scope is None
                                else self.model_cost(scope).total_cost
                            ),
                        )
                        self._on_alert(alert)
