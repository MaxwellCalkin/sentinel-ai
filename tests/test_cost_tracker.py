"""Tests for LLM API cost tracking."""

import pytest
import time
from sentinel.cost_tracker import (
    CostTracker, UsageRecord, ModelCost, BudgetAlert, BudgetExceeded,
)


# ---------------------------------------------------------------------------
# Basic recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_single(self):
        t = CostTracker()
        rec = t.record("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        assert isinstance(rec, UsageRecord)
        assert rec.model == "claude-sonnet-4-20250514"
        assert rec.input_tokens == 1000
        assert rec.output_tokens == 500
        assert rec.cost > 0

    def test_record_with_metadata(self):
        t = CostTracker()
        rec = t.record(
            "claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            metadata={"request_id": "abc123"},
        )
        assert rec.metadata["request_id"] == "abc123"

    def test_call_count(self):
        t = CostTracker()
        t.record("claude-sonnet-4-20250514", input_tokens=100, output_tokens=50)
        t.record("claude-sonnet-4-20250514", input_tokens=200, output_tokens=100)
        assert t.call_count == 2

    def test_unknown_model_zero_cost(self):
        t = CostTracker()
        rec = t.record("unknown-model", input_tokens=1000, output_tokens=500)
        assert rec.cost == 0.0


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_sonnet_pricing(self):
        t = CostTracker()
        rec = t.record("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=1000)
        # 1K input * 0.003 + 1K output * 0.015 = 0.018
        assert abs(rec.cost - 0.018) < 1e-9

    def test_opus_pricing(self):
        t = CostTracker()
        rec = t.record("claude-opus-4-20250514", input_tokens=1000, output_tokens=1000)
        # 1K input * 0.015 + 1K output * 0.075 = 0.09
        assert abs(rec.cost - 0.09) < 1e-9

    def test_custom_pricing(self):
        t = CostTracker()
        t.set_pricing("my-model", input_per_1k=0.01, output_per_1k=0.02)
        rec = t.record("my-model", input_tokens=2000, output_tokens=1000)
        # 2K * 0.01 + 1K * 0.02 = 0.04
        assert abs(rec.cost - 0.04) < 1e-9

    def test_total_cost(self):
        t = CostTracker()
        t.set_pricing("m", input_per_1k=0.01, output_per_1k=0.01)
        t.record("m", input_tokens=1000, output_tokens=0)  # 0.01
        t.record("m", input_tokens=0, output_tokens=1000)  # 0.01
        assert abs(t.total_cost - 0.02) < 1e-9

    def test_total_tokens(self):
        t = CostTracker()
        t.record("claude-sonnet-4-20250514", input_tokens=500, output_tokens=200)
        t.record("claude-sonnet-4-20250514", input_tokens=300, output_tokens=100)
        assert t.total_tokens == 1100


# ---------------------------------------------------------------------------
# Per-model breakdown
# ---------------------------------------------------------------------------

class TestModelBreakdown:
    def test_model_cost(self):
        t = CostTracker()
        t.record("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        t.record("claude-sonnet-4-20250514", input_tokens=2000, output_tokens=1000)
        mc = t.model_cost("claude-sonnet-4-20250514")
        assert mc.call_count == 2
        assert mc.total_input_tokens == 3000
        assert mc.total_output_tokens == 1500

    def test_by_model(self):
        t = CostTracker()
        t.record("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        t.record("claude-opus-4-20250514", input_tokens=500, output_tokens=200)
        breakdown = t.by_model()
        assert len(breakdown) == 2
        assert "claude-sonnet-4-20250514" in breakdown
        assert "claude-opus-4-20250514" in breakdown

    def test_model_cost_empty(self):
        t = CostTracker()
        mc = t.model_cost("nonexistent")
        assert mc.call_count == 0
        assert mc.total_cost == 0.0


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgets:
    def test_global_budget(self):
        t = CostTracker(enforce_budgets=True)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(0.05)  # global $0.05
        t.record("m", input_tokens=40, output_tokens=0)  # $0.04
        with pytest.raises(BudgetExceeded):
            t.record("m", input_tokens=20, output_tokens=0)  # would be $0.06

    def test_model_budget(self):
        t = CostTracker(enforce_budgets=True)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(0.01, model="m")
        with pytest.raises(BudgetExceeded):
            t.record("m", input_tokens=20, output_tokens=0)  # $0.02 > $0.01

    def test_budget_not_enforced(self):
        t = CostTracker(enforce_budgets=False)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(0.01)
        # Should not raise even though over budget
        t.record("m", input_tokens=1000, output_tokens=0)
        assert t.total_cost > 0.01

    def test_budget_remaining(self):
        t = CostTracker()
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(1.0)
        t.record("m", input_tokens=500, output_tokens=0)  # $0.50
        remaining = t.budget_remaining()
        assert remaining is not None
        assert abs(remaining - 0.5) < 1e-6

    def test_budget_remaining_no_budget(self):
        t = CostTracker()
        assert t.budget_remaining() is None

    def test_budget_utilization(self):
        t = CostTracker()
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(1.0)
        t.record("m", input_tokens=250, output_tokens=0)  # $0.25
        util = t.budget_utilization()
        assert util is not None
        assert abs(util - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_alert_fires(self):
        alerts = []
        t = CostTracker(enforce_budgets=False)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(1.0)
        t.set_alert_thresholds([0.5, 0.8])
        t.on_alert(lambda a: alerts.append(a))

        t.record("m", input_tokens=600, output_tokens=0)  # $0.60 -> 60% -> fires 0.5
        assert len(alerts) == 1
        assert alerts[0].threshold == 0.5

    def test_alert_fires_multiple(self):
        alerts = []
        t = CostTracker(enforce_budgets=False)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(1.0)
        t.set_alert_thresholds([0.5, 0.8])
        t.on_alert(lambda a: alerts.append(a))

        t.record("m", input_tokens=900, output_tokens=0)  # $0.90 -> fires both
        assert len(alerts) == 2

    def test_alert_fires_once(self):
        alerts = []
        t = CostTracker(enforce_budgets=False)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(1.0)
        t.set_alert_thresholds([0.5])
        t.on_alert(lambda a: alerts.append(a))

        t.record("m", input_tokens=600, output_tokens=0)
        t.record("m", input_tokens=100, output_tokens=0)
        assert len(alerts) == 1  # Should not fire again


# ---------------------------------------------------------------------------
# Time-based queries
# ---------------------------------------------------------------------------

class TestTimeQueries:
    def test_cost_since(self):
        t = CostTracker()
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.record("m", input_tokens=1000, output_tokens=0)
        mark = time.time()
        t.record("m", input_tokens=2000, output_tokens=0)
        cost = t.cost_since(mark)
        assert abs(cost - 2.0) < 0.1  # Approximate due to timing

    def test_cost_between(self):
        t = CostTracker()
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        start = time.time()
        t.record("m", input_tokens=1000, output_tokens=0)
        end = time.time()
        cost = t.cost_between(start, end)
        assert abs(cost - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_structure(self):
        t = CostTracker()
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(10.0)
        t.record("m", input_tokens=1000, output_tokens=0)

        s = t.summary()
        assert "total_cost" in s
        assert "total_tokens" in s
        assert "call_count" in s
        assert "models" in s
        assert "budgets" in s
        assert "m" in s["models"]
        assert s["call_count"] == 1

    def test_summary_empty(self):
        t = CostTracker()
        s = t.summary()
        assert s["total_cost"] == 0
        assert s["call_count"] == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears(self):
        t = CostTracker()
        t.record("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        t.reset()
        assert t.total_cost == 0
        assert t.call_count == 0

    def test_reset_clears_alerts(self):
        alerts = []
        t = CostTracker(enforce_budgets=False)
        t.set_pricing("m", input_per_1k=1.0, output_per_1k=0.0)
        t.set_budget(1.0)
        t.set_alert_thresholds([0.5])
        t.on_alert(lambda a: alerts.append(a))

        t.record("m", input_tokens=600, output_tokens=0)
        assert len(alerts) == 1
        t.reset()
        t.record("m", input_tokens=600, output_tokens=0)
        assert len(alerts) == 2  # Alert fires again after reset


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_tokens(self):
        t = CostTracker()
        rec = t.record("claude-sonnet-4-20250514", input_tokens=0, output_tokens=0)
        assert rec.cost == 0.0

    def test_budget_exceeded_message(self):
        exc = BudgetExceeded("my-model", 1.0, 1.5)
        assert "my-model" in str(exc)
        assert "1.5" in str(exc)

    def test_global_budget_exceeded_message(self):
        exc = BudgetExceeded(None, 1.0, 1.5)
        assert "global" in str(exc)
