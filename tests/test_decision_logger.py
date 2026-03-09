"""Tests for the structured decision logger."""

import threading
import time

import pytest

from sentinel.decision_logger import DecisionLogger, Decision, DecisionSummary


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

class TestDecision:
    def test_to_dict_contains_all_fields(self):
        d = Decision(
            action="block",
            reason="Injection detected",
            component="PromptInjectionScanner",
            timestamp=1000.0,
            metadata={"score": 0.95},
            input_hash="abc123",
        )
        result = d.to_dict()
        assert result["action"] == "block"
        assert result["reason"] == "Injection detected"
        assert result["component"] == "PromptInjectionScanner"
        assert result["timestamp"] == 1000.0
        assert result["metadata"] == {"score": 0.95}
        assert result["input_hash"] == "abc123"

    def test_defaults(self):
        d = Decision(action="allow", reason="", component="", timestamp=0.0)
        assert d.metadata == {}
        assert d.input_hash == ""


# ---------------------------------------------------------------------------
# Logging decisions
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_single_decision(self):
        logger = DecisionLogger()
        decision = logger.log("block", reason="Harmful content", component="ToxicityScanner")
        assert logger.count == 1
        assert decision.action == "block"
        assert decision.reason == "Harmful content"
        assert decision.component == "ToxicityScanner"
        assert isinstance(decision, Decision)

    def test_log_assigns_timestamp(self):
        logger = DecisionLogger()
        before = time.time()
        decision = logger.log("allow")
        after = time.time()
        assert before <= decision.timestamp <= after

    def test_log_input_hash(self):
        logger = DecisionLogger()
        with_text = logger.log("flag", input_text="sensitive input")
        without_text = logger.log("allow")
        assert len(with_text.input_hash) == 16
        assert without_text.input_hash == ""

    def test_log_with_metadata(self):
        logger = DecisionLogger()
        decision = logger.log("redact", metadata={"field": "ssn", "count": 3})
        assert decision.metadata == {"field": "ssn", "count": 3}


# ---------------------------------------------------------------------------
# Max size eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_max_size_evicts_oldest(self):
        logger = DecisionLogger(max_size=3)
        for i in range(5):
            logger.log("allow", reason=str(i))
        assert logger.count == 3
        reasons = [d.reason for d in logger.decisions]
        assert reasons == ["2", "3", "4"]

    def test_max_size_exactly_at_limit(self):
        logger = DecisionLogger(max_size=5)
        for i in range(5):
            logger.log("allow")
        assert logger.count == 5


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

class TestQuerying:
    def setup_method(self):
        self.logger = DecisionLogger()
        self.logger.log("allow", component="PIIScanner")
        self.logger.log("block", component="PromptInjectionScanner")
        self.logger.log("flag", component="ToxicityScanner")
        self.logger.log("block", component="PromptInjectionScanner")
        self.logger.log("redact", component="PIIScanner")

    def test_query_by_action(self):
        results = self.logger.query(action="block")
        assert len(results) == 2
        assert all(d.action == "block" for d in results)

    def test_query_by_component(self):
        results = self.logger.query(component="PIIScanner")
        assert len(results) == 2

    def test_query_by_time_range_since_and_until(self):
        future = time.time() + 1000
        past = time.time() - 1000
        assert len(self.logger.query(since=future)) == 0
        assert len(self.logger.query(until=past)) == 0

    def test_query_combined_filters(self):
        results = self.logger.query(action="block", component="PromptInjectionScanner")
        assert len(results) == 2

    def test_query_no_filters_returns_all(self):
        results = self.logger.query()
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_as_list_of_dicts(self):
        logger = DecisionLogger()
        logger.log("block", reason="Bad input", component="Scanner")
        exported = logger.export()
        assert len(exported) == 1
        assert isinstance(exported[0], dict)
        assert exported[0]["action"] == "block"
        assert exported[0]["reason"] == "Bad input"

    def test_export_empty_log(self):
        logger = DecisionLogger()
        assert logger.export() == []


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_on_empty_log(self):
        logger = DecisionLogger()
        summary = logger.summary()
        assert isinstance(summary, DecisionSummary)
        assert summary.total == 0
        assert summary.by_action == {}
        assert summary.by_component == {}
        assert summary.block_rate == 0.0

    def test_summary_counts_and_block_rate(self):
        logger = DecisionLogger()
        logger.log("allow", component="A")
        logger.log("block", component="B")
        logger.log("block", component="A")
        logger.log("flag", component="B")

        summary = logger.summary()
        assert summary.total == 4
        assert summary.by_action == {"allow": 1, "block": 2, "flag": 1}
        assert summary.by_component == {"A": 2, "B": 2}
        assert summary.block_rate == pytest.approx(0.5)

    def test_summary_block_rate_zero_when_no_blocks(self):
        logger = DecisionLogger()
        logger.log("allow")
        logger.log("flag")
        summary = logger.summary()
        assert summary.block_rate == 0.0


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all(self):
        logger = DecisionLogger()
        logger.log("allow")
        logger.log("block")
        logger.clear()
        assert logger.count == 0
        assert logger.decisions == []


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_logging(self):
        logger = DecisionLogger()
        num_threads = 10
        logs_per_thread = 100

        def log_many():
            for _ in range(logs_per_thread):
                logger.log("allow", component="concurrent")

        threads = [threading.Thread(target=log_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert logger.count == num_threads * logs_per_thread


# ---------------------------------------------------------------------------
# Defensive copy
# ---------------------------------------------------------------------------

class TestDefensiveCopy:
    def test_decisions_property_returns_copy(self):
        logger = DecisionLogger()
        logger.log("allow")
        snapshot = logger.decisions
        snapshot.append(Decision(action="fake", reason="", component="", timestamp=0.0))
        assert logger.count == 1
