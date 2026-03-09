"""Tests for interaction logger."""

import time
import threading
import pytest
from sentinel.interaction_logger import InteractionLogger, Interaction, InteractionSummary


# ---------------------------------------------------------------------------
# Logging interactions
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_returns_interaction(self):
        logger = InteractionLogger()
        result = logger.log("Hello", "Hi there", model="claude-3")
        assert isinstance(result, Interaction)
        assert result.input_text == "Hello"
        assert result.output_text == "Hi there"
        assert result.model == "claude-3"

    def test_log_assigns_unique_id(self):
        logger = InteractionLogger()
        first = logger.log("a", "b")
        second = logger.log("c", "d")
        assert first.id != second.id

    def test_log_assigns_timestamp(self):
        before = time.time()
        logger = InteractionLogger()
        result = logger.log("input", "output")
        after = time.time()
        assert before <= result.timestamp <= after

    def test_log_stores_safety_score_and_latency(self):
        logger = InteractionLogger()
        result = logger.log("q", "a", safety_score=0.85, latency_ms=42.5)
        assert result.safety_score == 0.85
        assert result.latency_ms == 42.5

    def test_log_stores_metadata(self):
        logger = InteractionLogger()
        result = logger.log("q", "a", metadata={"env": "prod", "version": 2})
        assert result.metadata["env"] == "prod"
        assert result.metadata["version"] == 2

    def test_log_defaults(self):
        logger = InteractionLogger()
        result = logger.log("q", "a")
        assert result.model == "unknown"
        assert result.safety_score == 1.0
        assert result.latency_ms == 0.0
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Querying by model
# ---------------------------------------------------------------------------

class TestQueryByModel:
    def test_query_by_model(self):
        logger = InteractionLogger()
        logger.log("a", "b", model="claude-3")
        logger.log("c", "d", model="gpt-4")
        logger.log("e", "f", model="claude-3")
        results = logger.query_by_model("claude-3")
        assert len(results) == 2
        assert all(r.model == "claude-3" for r in results)

    def test_query_by_model_no_match(self):
        logger = InteractionLogger()
        logger.log("a", "b", model="claude-3")
        results = logger.query_by_model("llama")
        assert results == []


# ---------------------------------------------------------------------------
# Querying by safety score
# ---------------------------------------------------------------------------

class TestQueryBySafetyScore:
    def test_query_by_safety_score_range(self):
        logger = InteractionLogger()
        logger.log("a", "b", safety_score=0.3)
        logger.log("c", "d", safety_score=0.7)
        logger.log("e", "f", safety_score=0.9)
        results = logger.query_by_safety_score(min_score=0.5, max_score=0.8)
        assert len(results) == 1
        assert results[0].safety_score == 0.7

    def test_query_by_safety_score_inclusive_bounds(self):
        logger = InteractionLogger()
        logger.log("a", "b", safety_score=0.5)
        logger.log("c", "d", safety_score=0.8)
        results = logger.query_by_safety_score(min_score=0.5, max_score=0.8)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Querying by time range
# ---------------------------------------------------------------------------

class TestQueryByTimeRange:
    def test_query_by_time_range(self):
        logger = InteractionLogger()
        before = time.time()
        logger.log("a", "b")
        after = time.time()
        time.sleep(0.01)
        logger.log("c", "d")
        results = logger.query_by_time_range(before, after)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_returns_list_of_dicts(self):
        logger = InteractionLogger()
        logger.log("q", "a", model="claude-3", safety_score=0.9, latency_ms=50.0)
        exported = logger.export()
        assert len(exported) == 1
        record = exported[0]
        assert isinstance(record, dict)
        assert record["input_text"] == "q"
        assert record["output_text"] == "a"
        assert record["model"] == "claude-3"
        assert record["safety_score"] == 0.9
        assert record["latency_ms"] == 50.0
        assert "id" in record
        assert "timestamp" in record
        assert "metadata" in record

    def test_export_empty(self):
        logger = InteractionLogger()
        assert logger.export() == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_basic(self):
        logger = InteractionLogger()
        logger.log("a", "b", model="claude-3", safety_score=0.8, latency_ms=100.0)
        logger.log("c", "d", model="gpt-4", safety_score=0.6, latency_ms=200.0)
        summary = logger.summary()
        assert isinstance(summary, InteractionSummary)
        assert summary.total == 2
        assert summary.avg_safety_score == 0.7
        assert summary.avg_latency_ms == 150.0
        assert summary.by_model == {"claude-3": 1, "gpt-4": 1}

    def test_summary_low_safety_count(self):
        logger = InteractionLogger()
        logger.log("a", "b", safety_score=0.3)
        logger.log("c", "d", safety_score=0.4)
        logger.log("e", "f", safety_score=0.8)
        summary = logger.summary()
        assert summary.low_safety_count == 2

    def test_summary_empty(self):
        logger = InteractionLogger()
        summary = logger.summary()
        assert summary.total == 0
        assert summary.avg_safety_score == 0.0
        assert summary.avg_latency_ms == 0.0
        assert summary.by_model == {}
        assert summary.low_safety_count == 0


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_max_size_eviction(self):
        logger = InteractionLogger(max_size=3)
        for i in range(5):
            logger.log(f"input-{i}", f"output-{i}", model=f"model-{i}")
        assert logger.count() == 3
        exported = logger.export()
        assert exported[0]["input_text"] == "input-2"
        assert exported[2]["input_text"] == "input-4"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_logging(self):
        logger = InteractionLogger(max_size=500)
        errors = []

        def log_batch(start):
            try:
                for i in range(100):
                    logger.log(f"input-{start}-{i}", f"output-{start}-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_batch, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert logger.count() == 500


# ---------------------------------------------------------------------------
# Clear and count
# ---------------------------------------------------------------------------

class TestClearAndCount:
    def test_clear_returns_count(self):
        logger = InteractionLogger()
        logger.log("a", "b")
        logger.log("c", "d")
        cleared = logger.clear()
        assert cleared == 2
        assert logger.count() == 0

    def test_clear_empty(self):
        logger = InteractionLogger()
        assert logger.clear() == 0
