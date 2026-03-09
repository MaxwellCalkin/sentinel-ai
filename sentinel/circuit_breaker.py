"""Circuit breaker for LLM API calls.

Prevents cascading failures by tracking error rates and
temporarily halting requests when a service is unhealthy.

Usage:
    from sentinel.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    try:
        result = cb.call(lambda: call_llm("prompt"))
    except CircuitOpenError:
        # Service is down, use fallback
        ...
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit is open and requests are blocked."""
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{name}' is open. Retry after {retry_after:.1f}s"
        )


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    last_failure_time: float | None
    last_success_time: float | None
    times_opened: int


class CircuitBreaker:
    """Circuit breaker for resilient API calls.

    Tracks failures and opens the circuit to prevent cascading
    failures when a service is unhealthy. Automatically tests
    recovery after a timeout period.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        on_state_change: Callable[[CircuitState, CircuitState], Any] | None = None,
    ) -> None:
        """
        Args:
            name: Circuit identifier.
            failure_threshold: Consecutive failures to open circuit.
            recovery_timeout: Seconds to wait before testing recovery.
            success_threshold: Successes in half-open to close circuit.
            on_state_change: Callback when state changes.
        """
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._total_calls = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._opened_at: float | None = None
        self._times_opened = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_recovery()
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def call(self, fn: Callable[[], T]) -> T:
        """Execute a function through the circuit breaker.

        Args:
            fn: Function to execute.

        Returns:
            The function's return value.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        with self._lock:
            self._check_recovery()
            self._total_calls += 1

            if self._state == CircuitState.OPEN:
                retry_after = self._recovery_timeout
                if self._opened_at:
                    elapsed = time.time() - self._opened_at
                    retry_after = max(0, self._recovery_timeout - elapsed)
                raise CircuitOpenError(self._name, retry_after)

        # Execute outside lock
        try:
            result = fn()
        except Exception as e:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._half_open_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                self._transition(CircuitState.OPEN)

    def _record_success(self) -> None:
        with self._lock:
            self._success_count += 1
            self._last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self._success_threshold:
                    self._transition(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset on success

    def _check_recovery(self) -> None:
        """Check if open circuit should transition to half-open."""
        if self._state == CircuitState.OPEN and self._opened_at:
            if time.time() - self._opened_at >= self._recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._times_opened += 1
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0

        if self._on_state_change and old_state != new_state:
            self._on_state_change(old_state, new_state)

    def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        with self._lock:
            self._transition(CircuitState.CLOSED)

    def stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        with self._lock:
            self._check_recovery()
            return CircuitStats(
                name=self._name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=self._total_calls,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                times_opened=self._times_opened,
            )
