"""Tests for retry policy."""

import pytest
from sentinel.retry_policy import RetryPolicy, RetryResult, RetryAttempt


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestSuccess:
    def test_immediate_success(self):
        policy = RetryPolicy(max_retries=3)
        result = policy.execute_dry_run(lambda: 42)
        assert result.success
        assert result.value == 42
        assert result.attempts == 1
        assert not result.retried

    def test_success_after_retry(self):
        calls = [0]
        def flaky():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("not yet")
            return "done"

        policy = RetryPolicy(max_retries=3, backoff_base=0.001)
        result = policy.execute_dry_run(flaky)
        assert result.success
        assert result.value == "done"
        assert result.attempts == 3
        assert result.retried


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------

class TestExhausted:
    def test_all_fail(self):
        policy = RetryPolicy(max_retries=2, backoff_base=0.001)
        result = policy.execute_dry_run(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert not result.success
        assert result.attempts == 3  # 1 original + 2 retries
        assert result.final_error is not None

    def test_no_retries(self):
        policy = RetryPolicy(max_retries=0)
        result = policy.execute_dry_run(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert not result.success
        assert result.attempts == 1


# ---------------------------------------------------------------------------
# Exception filtering
# ---------------------------------------------------------------------------

class TestExceptionFiltering:
    def test_retry_on_specific(self):
        calls = [0]
        def fn():
            calls[0] += 1
            if calls[0] == 1:
                raise ConnectionError("timeout")
            return "ok"

        policy = RetryPolicy(
            max_retries=3,
            retry_on=(ConnectionError,),
        )
        result = policy.execute_dry_run(fn)
        assert result.success
        assert result.attempts == 2

    def test_no_retry_wrong_exception(self):
        policy = RetryPolicy(
            max_retries=3,
            retry_on=(ConnectionError,),
        )
        result = policy.execute_dry_run(
            lambda: (_ for _ in ()).throw(ValueError("wrong"))
        )
        assert not result.success
        assert result.attempts == 1  # Did not retry


# ---------------------------------------------------------------------------
# Backoff computation
# ---------------------------------------------------------------------------

class TestBackoff:
    def test_exponential_growth(self):
        policy = RetryPolicy(
            backoff_base=1.0,
            backoff_multiplier=2.0,
            max_delay=100.0,
            jitter=False,
        )
        assert policy.compute_delay(0) == 1.0
        assert policy.compute_delay(1) == 2.0
        assert policy.compute_delay(2) == 4.0
        assert policy.compute_delay(3) == 8.0

    def test_max_delay_cap(self):
        policy = RetryPolicy(
            backoff_base=1.0,
            backoff_multiplier=2.0,
            max_delay=5.0,
            jitter=False,
        )
        assert policy.compute_delay(10) == 5.0

    def test_jitter_reduces_delay(self):
        policy = RetryPolicy(
            backoff_base=10.0,
            backoff_multiplier=1.0,
            jitter=True,
        )
        delays = [policy.compute_delay(0) for _ in range(100)]
        assert all(5.0 <= d <= 10.0 for d in delays)
        # Should have some variance
        assert min(delays) < max(delays)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_on_retry_called(self):
        retries = []
        policy = RetryPolicy(
            max_retries=2,
            on_retry=lambda attempt: retries.append(attempt),
        )
        result = policy.execute_dry_run(
            lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        assert len(retries) == 2
        assert all(isinstance(r, RetryAttempt) for r in retries)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_recorded(self):
        calls = [0]
        def fn():
            calls[0] += 1
            if calls[0] < 2:
                raise ValueError("first fail")
            return "ok"

        policy = RetryPolicy(max_retries=3)
        result = policy.execute_dry_run(fn)
        assert len(result.history) == 2
        assert not result.history[0].success
        assert result.history[0].error is not None
        assert result.history[1].success

    def test_latency_positive(self):
        policy = RetryPolicy(max_retries=0)
        result = policy.execute_dry_run(lambda: "fast")
        assert result.history[0].latency_ms >= 0
        assert result.total_latency_ms >= 0


# ---------------------------------------------------------------------------
# RetryResult properties
# ---------------------------------------------------------------------------

class TestRetryResult:
    def test_retried_property(self):
        r = RetryResult(success=True, attempts=3)
        assert r.retried

    def test_not_retried(self):
        r = RetryResult(success=True, attempts=1)
        assert not r.retried

    def test_failed_not_retried(self):
        r = RetryResult(success=False, attempts=3)
        assert not r.retried

    def test_final_error_on_failure(self):
        err = RuntimeError("boom")
        r = RetryResult(
            success=False,
            attempts=1,
            history=[RetryAttempt(attempt=0, success=False, error=err)],
        )
        assert r.final_error is err

    def test_final_error_none_on_success(self):
        r = RetryResult(success=True, attempts=1)
        assert r.final_error is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_should_retry_all(self):
        policy = RetryPolicy(retry_on=None)
        assert policy.should_retry(ValueError("x"))
        assert policy.should_retry(RuntimeError("x"))

    def test_should_retry_specific(self):
        policy = RetryPolicy(retry_on=(ValueError,))
        assert policy.should_retry(ValueError("x"))
        assert not policy.should_retry(RuntimeError("x"))

    def test_max_retries_property(self):
        policy = RetryPolicy(max_retries=5)
        assert policy.max_retries == 5
