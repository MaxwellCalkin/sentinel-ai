"""Tests for circuit breaker."""

import time
import pytest
from sentinel.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError, CircuitStats,
)


# ---------------------------------------------------------------------------
# Normal operation (closed)
# ---------------------------------------------------------------------------

class TestClosedState:
    def test_success(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: 42)
        assert result == 42

    def test_stays_closed_on_success(self):
        cb = CircuitBreaker()
        cb.call(lambda: "ok")
        assert cb.is_closed

    def test_exception_propagates(self):
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(ValueError, match="test"):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("test")))

    def test_failure_count_increments(self):
        cb = CircuitBreaker(failure_threshold=10)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
            except RuntimeError:
                pass
        assert cb.stats().failure_count == 3

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=10)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        cb.call(lambda: "ok")
        assert cb.stats().failure_count == 0


# ---------------------------------------------------------------------------
# Opening the circuit
# ---------------------------------------------------------------------------

class TestOpening:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
            except RuntimeError:
                pass
        assert cb.is_open

    def test_blocks_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "should not execute")

    def test_circuit_open_error_has_retry(self):
        cb = CircuitBreaker(name="test_cb", failure_threshold=1, recovery_timeout=30.0)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        try:
            cb.call(lambda: "x")
        except CircuitOpenError as e:
            assert e.name == "test_cb"
            assert e.retry_after > 0


# ---------------------------------------------------------------------------
# Recovery (half-open)
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        assert cb.is_open
        time.sleep(0.02)
        # State should transition to half-open on next check
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        time.sleep(0.02)
        # Two successes in half-open should close
        cb.call(lambda: "ok")
        cb.call(lambda: "ok")
        assert cb.is_closed

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        time.sleep(0.02)
        # Fail during half-open
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        assert cb.is_open


# ---------------------------------------------------------------------------
# State change callback
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_on_state_change(self):
        changes = []
        cb = CircuitBreaker(
            failure_threshold=1,
            on_state_change=lambda old, new: changes.append((old, new)),
        )
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        assert len(changes) == 1
        assert changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_structure(self):
        cb = CircuitBreaker(name="test")
        cb.call(lambda: "ok")
        s = cb.stats()
        assert isinstance(s, CircuitStats)
        assert s.name == "test"
        assert s.state == CircuitState.CLOSED
        assert s.success_count == 1
        assert s.total_calls == 1

    def test_times_opened(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        time.sleep(0.02)
        cb.call(lambda: "recover")
        cb.call(lambda: "recover")
        # Now closed again, trip it again
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        assert cb.stats().times_opened == 2


# ---------------------------------------------------------------------------
# Manual reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        assert cb.is_open
        cb.reset()
        assert cb.is_closed

    def test_works_after_reset(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass
        cb.reset()
        result = cb.call(lambda: 99)
        assert result == 99


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_name_property(self):
        cb = CircuitBreaker(name="my_circuit")
        assert cb.name == "my_circuit"

    def test_zero_failure_threshold(self):
        # Threshold of 0 means it opens immediately on any failure
        # But since we check >= threshold, 0 would always be open
        # This is an unusual config but should not crash
        cb = CircuitBreaker(failure_threshold=0)
        # First failure should open it
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass

    def test_concurrent_calls(self):
        import threading
        cb = CircuitBreaker(failure_threshold=100)
        results = []

        def worker():
            try:
                r = cb.call(lambda: 1)
                results.append(r)
            except Exception:
                pass

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 20
