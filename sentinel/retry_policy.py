"""Configurable retry policies for LLM API calls.

Provides exponential backoff, jitter, and per-exception retry logic
for resilient LLM applications.

Usage:
    from sentinel.retry_policy import RetryPolicy

    policy = RetryPolicy(max_retries=3, backoff_base=1.0)
    result = policy.execute(lambda: call_llm("prompt"))
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryAttempt:
    """Record of a single attempt."""
    attempt: int
    success: bool
    error: Exception | None = None
    latency_ms: float = 0.0
    waited_ms: float = 0.0


@dataclass
class RetryResult:
    """Result of a retried operation."""
    value: Any = None
    success: bool = False
    attempts: int = 0
    total_latency_ms: float = 0.0
    history: list[RetryAttempt] = field(default_factory=list)

    @property
    def retried(self) -> bool:
        return self.attempts > 1 and self.success

    @property
    def final_error(self) -> Exception | None:
        if self.history and not self.success:
            return self.history[-1].error
        return None


class RetryPolicy:
    """Configurable retry with exponential backoff.

    Supports max retries, backoff base/multiplier, jitter,
    max delay cap, and per-exception retry filtering.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_delay: float = 30.0,
        jitter: bool = True,
        retry_on: tuple[type[Exception], ...] | None = None,
        on_retry: Callable[[RetryAttempt], Any] | None = None,
    ) -> None:
        """
        Args:
            max_retries: Maximum number of retries (0 = no retries).
            backoff_base: Initial delay in seconds.
            backoff_multiplier: Multiplier for each subsequent retry.
            max_delay: Maximum delay cap in seconds.
            jitter: Add random jitter to delays.
            retry_on: Exception types to retry on. None = retry all.
            on_retry: Callback invoked before each retry.
        """
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_multiplier = backoff_multiplier
        self._max_delay = max_delay
        self._jitter = jitter
        self._retry_on = retry_on
        self._on_retry = on_retry

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for a given attempt number (0-indexed)."""
        delay = self._backoff_base * (self._backoff_multiplier ** attempt)
        delay = min(delay, self._max_delay)
        if self._jitter:
            delay = delay * random.uniform(0.5, 1.0)
        return delay

    def should_retry(self, error: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        if self._retry_on is None:
            return True
        return isinstance(error, self._retry_on)

    def execute(self, fn: Callable[[], T]) -> RetryResult:
        """Execute a function with retry logic.

        Args:
            fn: Function to execute. Takes no arguments, returns any value.

        Returns:
            RetryResult with value (if successful) and attempt history.
        """
        history: list[RetryAttempt] = []
        total_waited = 0.0

        for attempt_num in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                value = fn()
                elapsed = (time.perf_counter() - start) * 1000
                history.append(RetryAttempt(
                    attempt=attempt_num,
                    success=True,
                    latency_ms=elapsed,
                    waited_ms=total_waited,
                ))
                return RetryResult(
                    value=value,
                    success=True,
                    attempts=attempt_num + 1,
                    total_latency_ms=sum(a.latency_ms for a in history) + total_waited,
                    history=history,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                attempt_rec = RetryAttempt(
                    attempt=attempt_num,
                    success=False,
                    error=e,
                    latency_ms=elapsed,
                )
                history.append(attempt_rec)

                # Check if we should retry this exception type
                if not self.should_retry(e):
                    return RetryResult(
                        success=False,
                        attempts=attempt_num + 1,
                        total_latency_ms=sum(a.latency_ms for a in history),
                        history=history,
                    )

                # Check if we've exhausted retries
                if attempt_num >= self._max_retries:
                    return RetryResult(
                        success=False,
                        attempts=attempt_num + 1,
                        total_latency_ms=sum(a.latency_ms for a in history) + total_waited,
                        history=history,
                    )

                # Compute and apply backoff
                delay = self.compute_delay(attempt_num)
                if self._on_retry:
                    self._on_retry(attempt_rec)
                time.sleep(delay)
                total_waited += delay * 1000

        # Should not reach here, but just in case
        return RetryResult(
            success=False,
            attempts=len(history),
            total_latency_ms=sum(a.latency_ms for a in history),
            history=history,
        )

    def execute_dry_run(self, fn: Callable[[], T]) -> RetryResult:
        """Execute without actually sleeping (for testing).

        Same as execute but skips the sleep calls.
        """
        history: list[RetryAttempt] = []
        simulated_wait = 0.0

        for attempt_num in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                value = fn()
                elapsed = (time.perf_counter() - start) * 1000
                history.append(RetryAttempt(
                    attempt=attempt_num,
                    success=True,
                    latency_ms=elapsed,
                    waited_ms=simulated_wait,
                ))
                return RetryResult(
                    value=value,
                    success=True,
                    attempts=attempt_num + 1,
                    total_latency_ms=sum(a.latency_ms for a in history) + simulated_wait,
                    history=history,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                attempt_rec = RetryAttempt(
                    attempt=attempt_num,
                    success=False,
                    error=e,
                    latency_ms=elapsed,
                )
                history.append(attempt_rec)

                if not self.should_retry(e):
                    return RetryResult(
                        success=False,
                        attempts=attempt_num + 1,
                        total_latency_ms=sum(a.latency_ms for a in history),
                        history=history,
                    )

                if attempt_num >= self._max_retries:
                    return RetryResult(
                        success=False,
                        attempts=attempt_num + 1,
                        total_latency_ms=sum(a.latency_ms for a in history) + simulated_wait,
                        history=history,
                    )

                delay = self.compute_delay(attempt_num)
                simulated_wait += delay * 1000
                if self._on_retry:
                    self._on_retry(attempt_rec)

        return RetryResult(
            success=False,
            attempts=len(history),
            total_latency_ms=sum(a.latency_ms for a in history),
            history=history,
        )
