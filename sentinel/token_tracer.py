"""Token usage tracing with per-operation breakdown and cost attribution.

Traces token usage across LLM requests, groups by operation and model,
estimates costs, and enforces token budgets.

Usage:
    from sentinel.token_tracer import TokenTracer

    tracer = TokenTracer(budget_limit=100_000)
    tracer.trace("summarize", input_tokens=500, output_tokens=200, model="claude-sonnet-4-6")
    print(tracer.total_tokens())       # 700
    print(tracer.budget_remaining())   # 99300
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


_DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "default": (3.0, 15.0),
}


@dataclass
class TraceEntry:
    """A single token usage trace."""
    operation: str
    input_tokens: int
    output_tokens: int
    model: str
    total_tokens: int
    timestamp: float


@dataclass
class TracerReport:
    """Aggregate report of all traced token usage."""
    total_traces: int
    total_input: int
    total_output: int
    total_tokens: int
    by_operation: dict[str, int]
    by_model: dict[str, int]
    budget_limit: int
    budget_remaining: int | None
    over_budget: bool


class TokenTracer:
    """Trace token usage across LLM requests.

    Tracks per-operation and per-model breakdowns, estimates costs
    using configurable rates, and enforces optional token budgets.
    """

    def __init__(self, budget_limit: int = 0) -> None:
        """
        Args:
            budget_limit: Maximum total tokens allowed. 0 means no limit.
        """
        self._budget_limit = budget_limit
        self._traces: list[TraceEntry] = []

    def trace(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
    ) -> TraceEntry:
        """Record a token usage trace.

        Args:
            operation: Name of the operation (e.g., "summarize", "classify").
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens produced.
            model: Model identifier used for the request.

        Returns:
            The recorded TraceEntry.
        """
        entry = TraceEntry(
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            total_tokens=input_tokens + output_tokens,
            timestamp=time.time(),
        )
        self._traces.append(entry)
        return entry

    def get_traces(self, operation: str | None = None) -> list[TraceEntry]:
        """Get recorded traces, optionally filtered by operation.

        Args:
            operation: If provided, only return traces for this operation.

        Returns:
            List of matching TraceEntry objects.
        """
        if operation is None:
            return list(self._traces)
        return [t for t in self._traces if t.operation == operation]

    def total_tokens(self) -> int:
        """Sum of all input and output tokens across all traces."""
        return sum(t.total_tokens for t in self._traces)

    def total_by_operation(self) -> dict[str, int]:
        """Total tokens grouped by operation name."""
        result: dict[str, int] = {}
        for entry in self._traces:
            result[entry.operation] = result.get(entry.operation, 0) + entry.total_tokens
        return result

    def total_by_model(self) -> dict[str, int]:
        """Total tokens grouped by model identifier."""
        result: dict[str, int] = {}
        for entry in self._traces:
            result[entry.model] = result.get(entry.model, 0) + entry.total_tokens
        return result

    def cost_estimate(
        self,
        rates: dict[str, tuple[float, float]] | None = None,
    ) -> float:
        """Estimate total cost based on per-model token rates.

        Args:
            rates: Mapping of model name to (input_rate, output_rate) per 1M tokens.
                   Falls back to "default" key for unknown models.
                   Uses {\"default\": (3.0, 15.0)} when no rates are provided.

        Returns:
            Estimated cost in dollars.
        """
        effective_rates = rates if rates is not None else _DEFAULT_RATES
        total_cost = 0.0
        for entry in self._traces:
            input_rate, output_rate = _resolve_rates(entry.model, effective_rates)
            total_cost += (entry.input_tokens * input_rate + entry.output_tokens * output_rate) / 1_000_000
        return total_cost

    def budget_remaining(self) -> int | None:
        """Remaining token budget, or None if no limit is set."""
        if self._budget_limit == 0:
            return None
        return self._budget_limit - self.total_tokens()

    def over_budget(self) -> bool:
        """True if total tokens exceed the budget limit."""
        if self._budget_limit == 0:
            return False
        return self.total_tokens() > self._budget_limit

    def report(self) -> TracerReport:
        """Generate an aggregate report of all traced token usage."""
        return TracerReport(
            total_traces=len(self._traces),
            total_input=sum(t.input_tokens for t in self._traces),
            total_output=sum(t.output_tokens for t in self._traces),
            total_tokens=self.total_tokens(),
            by_operation=self.total_by_operation(),
            by_model=self.total_by_model(),
            budget_limit=self._budget_limit,
            budget_remaining=self.budget_remaining(),
            over_budget=self.over_budget(),
        )

    def clear(self) -> None:
        """Reset all traces."""
        self._traces.clear()


def _resolve_rates(
    model: str,
    rates: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Look up rates for a model, falling back to the default key."""
    if model in rates:
        return rates[model]
    return rates.get("default", (0.0, 0.0))
