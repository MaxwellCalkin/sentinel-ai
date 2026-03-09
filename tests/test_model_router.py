"""Tests for intelligent model routing."""

import pytest
from sentinel.model_router import ModelRouter, RouteDecision, RouterStats


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

class TestBasicRouting:
    def test_default_route(self):
        r = ModelRouter(default_model="claude-sonnet")
        decision = r.route("Hello world")
        assert decision.model == "claude-sonnet"
        assert decision.fallback
        assert not decision.matched

    def test_custom_route_match(self):
        r = ModelRouter()
        r.add_route("sensitive", "claude-opus", lambda t: "secret" in t)
        decision = r.route("This is secret data")
        assert decision.model == "claude-opus"
        assert decision.rule_name == "sensitive"
        assert decision.matched

    def test_no_match_falls_back(self):
        r = ModelRouter(default_model="haiku")
        r.add_route("long", "opus", lambda t: len(t) > 1000)
        decision = r.route("short")
        assert decision.model == "haiku"
        assert decision.fallback


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class TestPriority:
    def test_higher_priority_wins(self):
        r = ModelRouter()
        r.add_route("low", "model-a", lambda t: True, priority=1)
        r.add_route("high", "model-b", lambda t: True, priority=10)
        decision = r.route("test")
        assert decision.model == "model-b"
        assert decision.rule_name == "high"

    def test_equal_priority_first_added(self):
        r = ModelRouter()
        r.add_route("first", "model-a", lambda t: True, priority=5)
        r.add_route("second", "model-b", lambda t: True, priority=5)
        decision = r.route("test")
        assert decision.model == "model-a"


# ---------------------------------------------------------------------------
# Keyword routing
# ---------------------------------------------------------------------------

class TestKeywordRouting:
    def test_keyword_match(self):
        r = ModelRouter()
        r.add_keyword_route("pii", "opus", ["ssn", "social security", "credit card"])
        assert r.route("What is my SSN?").model == "opus"

    def test_keyword_case_insensitive(self):
        r = ModelRouter()
        r.add_keyword_route("pii", "opus", ["password"])
        assert r.route("Tell me the PASSWORD").model == "opus"

    def test_keyword_no_match(self):
        r = ModelRouter(default_model="haiku")
        r.add_keyword_route("pii", "opus", ["ssn"])
        assert r.route("Hello world").model == "haiku"


# ---------------------------------------------------------------------------
# Sensitivity routing
# ---------------------------------------------------------------------------

class TestSensitivityRouting:
    def test_regex_pattern_match(self):
        r = ModelRouter()
        r.add_sensitivity_route("pii", "opus", [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b\d{16}\b",              # Credit card
        ])
        assert r.route("My SSN is 123-45-6789").model == "opus"

    def test_regex_no_match(self):
        r = ModelRouter(default_model="haiku")
        r.add_sensitivity_route("pii", "opus", [r"\b\d{3}-\d{2}-\d{4}\b"])
        assert r.route("Just regular text").model == "haiku"

    def test_sensitivity_high_priority(self):
        r = ModelRouter()
        r.add_route("generic", "sonnet", lambda t: True, priority=0)
        r.add_sensitivity_route("pii", "opus", [r"secret"])
        # Sensitivity routes get priority=10 by default
        assert r.route("This is secret").model == "opus"


# ---------------------------------------------------------------------------
# Batch routing
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_routing(self):
        r = ModelRouter(default_model="haiku")
        r.add_keyword_route("sensitive", "opus", ["secret"])
        results = r.route_batch(["hello", "secret info", "normal"])
        assert results[0].model == "haiku"
        assert results[1].model == "opus"
        assert results[2].model == "haiku"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_tracking(self):
        r = ModelRouter(default_model="haiku")
        r.add_route("custom", "opus", lambda t: "x" in t)
        r.route("x")
        r.route("y")
        r.route("x again")
        stats = r.stats()
        assert stats.total_routed == 3
        assert stats.route_counts["custom"] == 2
        assert stats.fallback_count == 1

    def test_reset_stats(self):
        r = ModelRouter()
        r.route("test")
        r.reset_stats()
        stats = r.stats()
        assert stats.total_routed == 0
        assert stats.fallback_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_rule_count(self):
        r = ModelRouter()
        assert r.rule_count == 0
        r.add_route("a", "m", lambda t: True)
        assert r.rule_count == 1

    def test_error_in_condition_skipped(self):
        r = ModelRouter(default_model="fallback")
        r.add_route("bad", "opus", lambda t: 1 / 0)  # Will raise
        decision = r.route("test")
        assert decision.model == "fallback"
        assert decision.fallback

    def test_empty_text(self):
        r = ModelRouter()
        decision = r.route("")
        assert decision.model == "claude-sonnet"
