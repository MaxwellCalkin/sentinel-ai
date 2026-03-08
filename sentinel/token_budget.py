"""Token budget guard — prevents runaway LLM costs.

Tracks cumulative token usage across API calls and enforces limits.
Supports per-call limits, session limits, and rate limiting.

Usage:
    from sentinel.token_budget import TokenBudget

    budget = TokenBudget(max_tokens_per_call=4096, max_tokens_per_session=100_000)
    budget.check_request(estimated_tokens=500)    # raises if over budget
    budget.record_usage(input_tokens=100, output_tokens=400)
    print(budget.remaining)  # 99500
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when a token budget limit is exceeded."""

    def __init__(self, message: str, usage: TokenUsage):
        super().__init__(message)
        self.usage = usage


@dataclass
class TokenUsage:
    """Cumulative token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0

    @property
    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


# Approximate costs per 1M tokens (input/output) as of March 2026
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
    "o1-mini": (1.10, 4.40),
    "o3": (10.0, 40.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-6",
) -> float:
    """Estimate cost in USD for a given token count."""
    costs = _MODEL_COSTS.get(model, (3.0, 15.0))  # default to Sonnet pricing
    input_cost = (input_tokens / 1_000_000) * costs[0]
    output_cost = (output_tokens / 1_000_000) * costs[1]
    return input_cost + output_cost


class TokenBudget:
    """Track and limit token usage across LLM calls.

    Args:
        max_tokens_per_call: Maximum tokens allowed per single API call.
        max_tokens_per_session: Maximum cumulative tokens for the session.
        max_cost_usd: Maximum cumulative cost in USD.
        max_calls: Maximum number of API calls.
        model: Default model for cost estimation.
    """

    def __init__(
        self,
        max_tokens_per_call: int | None = None,
        max_tokens_per_session: int | None = None,
        max_cost_usd: float | None = None,
        max_calls: int | None = None,
        model: str = "claude-sonnet-4-6",
    ):
        self._max_per_call = max_tokens_per_call
        self._max_per_session = max_tokens_per_session
        self._max_cost = max_cost_usd
        self._max_calls = max_calls
        self._model = model
        self._usage = TokenUsage()
        self._lock = threading.Lock()
        self._call_history: list[dict] = []

    def check_request(self, estimated_tokens: int, model: str | None = None) -> None:
        """Check if a request is within budget. Raises BudgetExceededError if not."""
        with self._lock:
            if self._max_per_call and estimated_tokens > self._max_per_call:
                raise BudgetExceededError(
                    f"Request exceeds per-call limit: {estimated_tokens} > {self._max_per_call}",
                    usage=self._usage,
                )

            if self._max_per_session:
                projected = self._usage.total_tokens + estimated_tokens
                if projected > self._max_per_session:
                    raise BudgetExceededError(
                        f"Request would exceed session limit: "
                        f"{projected} > {self._max_per_session} "
                        f"(remaining: {self._max_per_session - self._usage.total_tokens})",
                        usage=self._usage,
                    )

            if self._max_calls and self._usage.total_calls >= self._max_calls:
                raise BudgetExceededError(
                    f"Call limit reached: {self._usage.total_calls} >= {self._max_calls}",
                    usage=self._usage,
                )

            if self._max_cost:
                m = model or self._model
                est_cost = estimate_cost(estimated_tokens, estimated_tokens, m)
                if self._usage.total_cost_usd + est_cost > self._max_cost:
                    raise BudgetExceededError(
                        f"Cost limit would be exceeded: "
                        f"${self._usage.total_cost_usd + est_cost:.4f} > ${self._max_cost:.4f}",
                        usage=self._usage,
                    )

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> TokenUsage:
        """Record token usage from an API call. Returns updated usage."""
        m = model or self._model
        cost = estimate_cost(input_tokens, output_tokens, m)

        with self._lock:
            self._usage.input_tokens += input_tokens
            self._usage.output_tokens += output_tokens
            self._usage.total_tokens += input_tokens + output_tokens
            self._usage.total_calls += 1
            self._usage.total_cost_usd += cost
            self._call_history.append({
                "timestamp": time.time(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": m,
                "cost_usd": cost,
            })
            return TokenUsage(
                input_tokens=self._usage.input_tokens,
                output_tokens=self._usage.output_tokens,
                total_tokens=self._usage.total_tokens,
                total_calls=self._usage.total_calls,
                total_cost_usd=self._usage.total_cost_usd,
            )

    @property
    def usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                input_tokens=self._usage.input_tokens,
                output_tokens=self._usage.output_tokens,
                total_tokens=self._usage.total_tokens,
                total_calls=self._usage.total_calls,
                total_cost_usd=self._usage.total_cost_usd,
            )

    @property
    def remaining(self) -> int | None:
        """Remaining token budget, or None if no session limit."""
        if self._max_per_session is None:
            return None
        with self._lock:
            return max(0, self._max_per_session - self._usage.total_tokens)

    @property
    def remaining_cost(self) -> float | None:
        """Remaining cost budget in USD, or None if no cost limit."""
        if self._max_cost is None:
            return None
        with self._lock:
            return max(0.0, self._max_cost - self._usage.total_cost_usd)

    @property
    def remaining_calls(self) -> int | None:
        """Remaining call budget, or None if no call limit."""
        if self._max_calls is None:
            return None
        with self._lock:
            return max(0, self._max_calls - self._usage.total_calls)

    @property
    def history(self) -> list[dict]:
        """Call history (copy)."""
        with self._lock:
            return list(self._call_history)

    def reset(self) -> None:
        """Reset all usage counters."""
        with self._lock:
            self._usage = TokenUsage()
            self._call_history.clear()
