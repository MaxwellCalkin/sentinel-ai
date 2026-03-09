"""Tests for token usage tracing with per-operation breakdown."""

from sentinel.token_tracer import TokenTracer, TraceEntry, TracerReport


# ---------------------------------------------------------------------------
# Trace recording
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace_basic(self):
        tracer = TokenTracer()
        entry = tracer.trace("summarize", input_tokens=100, output_tokens=50)

        assert isinstance(entry, TraceEntry)
        assert entry.operation == "summarize"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.model == ""
        assert entry.total_tokens == 150
        assert entry.timestamp > 0

    def test_trace_with_model(self):
        tracer = TokenTracer()
        entry = tracer.trace("classify", input_tokens=200, output_tokens=30, model="claude-sonnet-4-6")

        assert entry.model == "claude-sonnet-4-6"
        assert entry.total_tokens == 230

    def test_multiple_traces(self):
        tracer = TokenTracer()
        tracer.trace("summarize", input_tokens=100, output_tokens=50)
        tracer.trace("classify", input_tokens=200, output_tokens=30)
        tracer.trace("summarize", input_tokens=150, output_tokens=60)

        assert len(tracer.get_traces()) == 3


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

class TestTotals:
    def test_total_tokens(self):
        tracer = TokenTracer()
        tracer.trace("op_a", input_tokens=100, output_tokens=50)
        tracer.trace("op_b", input_tokens=200, output_tokens=100)

        assert tracer.total_tokens() == 450

    def test_by_operation(self):
        tracer = TokenTracer()
        tracer.trace("summarize", input_tokens=100, output_tokens=50)
        tracer.trace("classify", input_tokens=200, output_tokens=30)
        tracer.trace("summarize", input_tokens=150, output_tokens=60)

        by_op = tracer.total_by_operation()
        assert by_op["summarize"] == 360  # (100+50) + (150+60)
        assert by_op["classify"] == 230   # 200+30

    def test_by_model(self):
        tracer = TokenTracer()
        tracer.trace("op", input_tokens=100, output_tokens=50, model="model-a")
        tracer.trace("op", input_tokens=200, output_tokens=100, model="model-b")
        tracer.trace("op", input_tokens=300, output_tokens=50, model="model-a")

        by_model = tracer.total_by_model()
        assert by_model["model-a"] == 500  # (100+50) + (300+50)
        assert by_model["model-b"] == 300  # 200+100


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCost:
    def test_cost_default_rate(self):
        tracer = TokenTracer()
        tracer.trace("op", input_tokens=1_000_000, output_tokens=0)

        # Default: input rate 3.0 per 1M -> cost = 3.0
        cost = tracer.cost_estimate()
        assert abs(cost - 3.0) < 1e-9

    def test_cost_custom_rates(self):
        tracer = TokenTracer()
        tracer.trace("op", input_tokens=500_000, output_tokens=500_000, model="custom")

        rates = {"custom": (2.0, 10.0)}
        # (500_000 * 2.0 + 500_000 * 10.0) / 1_000_000 = 1.0 + 5.0 = 6.0
        cost = tracer.cost_estimate(rates=rates)
        assert abs(cost - 6.0) < 1e-9

    def test_cost_model_specific(self):
        tracer = TokenTracer()
        tracer.trace("op", input_tokens=1_000_000, output_tokens=0, model="cheap")
        tracer.trace("op", input_tokens=1_000_000, output_tokens=0, model="expensive")

        rates = {
            "cheap": (1.0, 5.0),
            "expensive": (10.0, 50.0),
            "default": (3.0, 15.0),
        }
        # cheap: 1_000_000 * 1.0 / 1M = 1.0
        # expensive: 1_000_000 * 10.0 / 1M = 10.0
        cost = tracer.cost_estimate(rates=rates)
        assert abs(cost - 11.0) < 1e-9


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

class TestBudget:
    def test_no_budget(self):
        tracer = TokenTracer(budget_limit=0)
        tracer.trace("op", input_tokens=999_999, output_tokens=999_999)

        assert tracer.budget_remaining() is None
        assert tracer.over_budget() is False

    def test_under_budget(self):
        tracer = TokenTracer(budget_limit=1000)
        tracer.trace("op", input_tokens=200, output_tokens=100)

        assert tracer.budget_remaining() == 700
        assert tracer.over_budget() is False

    def test_over_budget(self):
        tracer = TokenTracer(budget_limit=500)
        tracer.trace("op", input_tokens=300, output_tokens=300)

        assert tracer.budget_remaining() == -100
        assert tracer.over_budget() is True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        tracer = TokenTracer(budget_limit=10000)
        tracer.trace("summarize", input_tokens=100, output_tokens=50, model="m")

        report = tracer.report()
        assert isinstance(report, TracerReport)
        assert hasattr(report, "total_traces")
        assert hasattr(report, "total_input")
        assert hasattr(report, "total_output")
        assert hasattr(report, "total_tokens")
        assert hasattr(report, "by_operation")
        assert hasattr(report, "by_model")
        assert hasattr(report, "budget_limit")
        assert hasattr(report, "budget_remaining")
        assert hasattr(report, "over_budget")

    def test_report_values(self):
        tracer = TokenTracer(budget_limit=5000)
        tracer.trace("summarize", input_tokens=400, output_tokens=200, model="model-a")
        tracer.trace("classify", input_tokens=100, output_tokens=50, model="model-b")

        report = tracer.report()
        assert report.total_traces == 2
        assert report.total_input == 500
        assert report.total_output == 250
        assert report.total_tokens == 750
        assert report.by_operation == {"summarize": 600, "classify": 150}
        assert report.by_model == {"model-a": 600, "model-b": 150}
        assert report.budget_limit == 5000
        assert report.budget_remaining == 4250
        assert report.over_budget is False


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilter:
    def test_get_by_operation(self):
        tracer = TokenTracer()
        tracer.trace("summarize", input_tokens=100, output_tokens=50)
        tracer.trace("classify", input_tokens=200, output_tokens=30)
        tracer.trace("summarize", input_tokens=150, output_tokens=60)

        filtered = tracer.get_traces(operation="summarize")
        assert len(filtered) == 2
        assert all(t.operation == "summarize" for t in filtered)

    def test_get_all(self):
        tracer = TokenTracer()
        tracer.trace("op_a", input_tokens=100, output_tokens=50)
        tracer.trace("op_b", input_tokens=200, output_tokens=30)

        all_traces = tracer.get_traces()
        assert len(all_traces) == 2


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets(self):
        tracer = TokenTracer(budget_limit=1000)
        tracer.trace("op", input_tokens=300, output_tokens=200)
        assert tracer.total_tokens() == 500

        tracer.clear()
        assert tracer.total_tokens() == 0
        assert tracer.get_traces() == []
        assert tracer.budget_remaining() == 1000
        assert tracer.over_budget() is False
