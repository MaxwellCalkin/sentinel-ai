"""Tests for the token budget guard."""

import pytest
from sentinel.token_budget import (
    TokenBudget,
    TokenUsage,
    BudgetExceededError,
    estimate_cost,
)


class TestEstimateCost:
    def test_sonnet_pricing(self):
        cost = estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4-6")
        assert cost == 3.0 + 15.0  # $3/M input + $15/M output

    def test_opus_pricing(self):
        cost = estimate_cost(1_000_000, 1_000_000, "claude-opus-4-6")
        assert cost == 15.0 + 75.0

    def test_gpt4o_pricing(self):
        cost = estimate_cost(1_000_000, 1_000_000, "gpt-4o")
        assert cost == 2.50 + 10.0

    def test_small_usage(self):
        cost = estimate_cost(100, 200, "claude-sonnet-4-6")
        assert cost == pytest.approx(0.0003 + 0.003, abs=1e-6)

    def test_unknown_model_defaults(self):
        cost = estimate_cost(1_000_000, 1_000_000, "unknown-model")
        assert cost == 3.0 + 15.0  # defaults to Sonnet pricing

    def test_zero_tokens(self):
        assert estimate_cost(0, 0) == 0.0


class TestTokenUsage:
    def test_default(self):
        u = TokenUsage()
        assert u.total_tokens == 0
        assert u.total_calls == 0
        assert u.total_cost_usd == 0.0

    def test_summary(self):
        u = TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300, total_calls=1)
        s = u.summary
        assert s["input_tokens"] == 100
        assert s["output_tokens"] == 200
        assert s["total_tokens"] == 300


class TestTokenBudgetRecording:
    def test_record_usage(self):
        budget = TokenBudget()
        budget.record_usage(100, 200)
        assert budget.usage.input_tokens == 100
        assert budget.usage.output_tokens == 200
        assert budget.usage.total_tokens == 300
        assert budget.usage.total_calls == 1

    def test_cumulative_usage(self):
        budget = TokenBudget()
        budget.record_usage(100, 200)
        budget.record_usage(300, 400)
        assert budget.usage.total_tokens == 1000
        assert budget.usage.total_calls == 2

    def test_cost_tracking(self):
        budget = TokenBudget(model="claude-sonnet-4-6")
        budget.record_usage(1_000_000, 1_000_000)
        assert budget.usage.total_cost_usd == pytest.approx(18.0, abs=0.01)

    def test_history(self):
        budget = TokenBudget()
        budget.record_usage(100, 200)
        budget.record_usage(300, 400)
        assert len(budget.history) == 2
        assert budget.history[0]["input_tokens"] == 100
        assert budget.history[1]["input_tokens"] == 300

    def test_reset(self):
        budget = TokenBudget()
        budget.record_usage(100, 200)
        budget.reset()
        assert budget.usage.total_tokens == 0
        assert len(budget.history) == 0


class TestTokenBudgetPerCallLimit:
    def test_within_limit(self):
        budget = TokenBudget(max_tokens_per_call=1000)
        budget.check_request(500)  # should not raise

    def test_exceeds_limit(self):
        budget = TokenBudget(max_tokens_per_call=1000)
        with pytest.raises(BudgetExceededError, match="per-call"):
            budget.check_request(1500)

    def test_exact_limit(self):
        budget = TokenBudget(max_tokens_per_call=1000)
        budget.check_request(1000)  # should not raise


class TestTokenBudgetSessionLimit:
    def test_within_session_limit(self):
        budget = TokenBudget(max_tokens_per_session=10_000)
        budget.record_usage(3000, 2000)
        budget.check_request(4000)  # 5000 + 4000 = 9000 < 10000

    def test_exceeds_session_limit(self):
        budget = TokenBudget(max_tokens_per_session=10_000)
        budget.record_usage(5000, 4000)
        with pytest.raises(BudgetExceededError, match="session limit"):
            budget.check_request(2000)  # 9000 + 2000 = 11000 > 10000

    def test_remaining(self):
        budget = TokenBudget(max_tokens_per_session=10_000)
        assert budget.remaining == 10_000
        budget.record_usage(3000, 2000)
        assert budget.remaining == 5000

    def test_remaining_no_limit(self):
        budget = TokenBudget()
        assert budget.remaining is None


class TestTokenBudgetCostLimit:
    def test_within_cost_limit(self):
        budget = TokenBudget(max_cost_usd=1.0, model="claude-sonnet-4-6")
        budget.check_request(1000)  # tiny cost

    def test_exceeds_cost_limit(self):
        budget = TokenBudget(max_cost_usd=0.01, model="claude-sonnet-4-6")
        budget.record_usage(500_000, 500_000)  # ~$9
        with pytest.raises(BudgetExceededError, match="Cost limit"):
            budget.check_request(100_000)

    def test_remaining_cost(self):
        budget = TokenBudget(max_cost_usd=10.0)
        assert budget.remaining_cost == pytest.approx(10.0)
        budget.record_usage(1_000_000, 0, model="claude-sonnet-4-6")  # $3
        assert budget.remaining_cost == pytest.approx(7.0, abs=0.01)

    def test_remaining_cost_no_limit(self):
        budget = TokenBudget()
        assert budget.remaining_cost is None


class TestTokenBudgetCallLimit:
    def test_within_call_limit(self):
        budget = TokenBudget(max_calls=10)
        for _ in range(9):
            budget.record_usage(100, 100)
        budget.check_request(100)  # 9 calls, 1 more allowed

    def test_exceeds_call_limit(self):
        budget = TokenBudget(max_calls=3)
        for _ in range(3):
            budget.record_usage(100, 100)
        with pytest.raises(BudgetExceededError, match="Call limit"):
            budget.check_request(100)

    def test_remaining_calls(self):
        budget = TokenBudget(max_calls=5)
        assert budget.remaining_calls == 5
        budget.record_usage(100, 100)
        assert budget.remaining_calls == 4

    def test_remaining_calls_no_limit(self):
        budget = TokenBudget()
        assert budget.remaining_calls is None


class TestBudgetExceededError:
    def test_has_usage(self):
        budget = TokenBudget(max_calls=1)
        budget.record_usage(100, 100)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check_request(100)
        assert exc_info.value.usage.total_calls == 1

    def test_is_exception(self):
        assert issubclass(BudgetExceededError, Exception)


class TestTokenBudgetCombined:
    def test_multiple_limits(self):
        budget = TokenBudget(
            max_tokens_per_call=5000,
            max_tokens_per_session=20_000,
            max_calls=10,
        )
        # All within limits
        budget.check_request(3000)
        budget.record_usage(1500, 1500)

        # Per-call limit exceeded
        with pytest.raises(BudgetExceededError, match="per-call"):
            budget.check_request(6000)

    def test_thread_safety(self):
        import threading
        budget = TokenBudget(max_tokens_per_session=1_000_000)
        errors = []

        def record():
            try:
                for _ in range(100):
                    budget.record_usage(10, 10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert budget.usage.total_tokens == 20_000  # 10 threads * 100 * 20
        assert budget.usage.total_calls == 1000
