"""Tests for intent classifier."""

import pytest
from sentinel.intent_classifier import (
    IntentClassifier,
    ClassificationOutput,
    ClassifierStats,
    IntentDef,
)


def _make_classifier() -> IntentClassifier:
    """Build a classifier with standard test intents."""
    classifier = IntentClassifier()
    classifier.add_intent(
        "code_gen",
        keywords=["write", "code", "function", "implement"],
        patterns=[r"\b(?:python|javascript|java)\b"],
        risk_level="medium",
    )
    classifier.add_intent(
        "data_query",
        keywords=["select", "database", "query", "table"],
        risk_level="low",
    )
    classifier.add_intent(
        "admin",
        keywords=["delete", "drop", "shutdown", "admin"],
        patterns=[r"rm\s+-rf", r"sudo\b"],
        risk_level="critical",
    )
    return classifier


class TestAddIntent:
    def test_add_intent(self):
        classifier = IntentClassifier()
        classifier.add_intent("greeting", keywords=["hello", "hi"])
        assert "greeting" in classifier.list_intents()

    def test_invalid_risk(self):
        classifier = IntentClassifier()
        with pytest.raises(ValueError, match="Invalid risk_level"):
            classifier.add_intent("bad", keywords=["x"], risk_level="extreme")

    def test_remove_intent(self):
        classifier = IntentClassifier()
        classifier.add_intent("temp", keywords=["a"])
        classifier.remove_intent("temp")
        assert "temp" not in classifier.list_intents()

    def test_list_intents(self):
        classifier = _make_classifier()
        intents = classifier.list_intents()
        assert intents == ["admin", "code_gen", "data_query"]


class TestClassify:
    def test_keyword_match(self):
        classifier = _make_classifier()
        result = classifier.classify("Write a function to sort numbers")
        assert result.intent == "code_gen"
        assert "write" in result.matched_keywords
        assert "function" in result.matched_keywords

    def test_pattern_match(self):
        classifier = _make_classifier()
        result = classifier.classify("Run sudo command now")
        assert result.intent == "admin"
        assert any("sudo" in p for p in result.matched_patterns)

    def test_no_match_uses_default(self):
        classifier = _make_classifier()
        result = classifier.classify("Tell me a joke about cats")
        assert result.intent == "general"
        assert result.confidence == 0.0

    def test_highest_score_wins(self):
        classifier = _make_classifier()
        # "write code function" hits 3 keywords in code_gen (3.0)
        # versus only "query" in data_query (1.0)
        result = classifier.classify("write code function query")
        assert result.intent == "code_gen"


class TestConfidence:
    def test_confidence_range(self):
        classifier = _make_classifier()
        result = classifier.classify("Write python code")
        assert 0.0 < result.confidence <= 1.0

    def test_zero_confidence_on_default(self):
        classifier = _make_classifier()
        result = classifier.classify("completely unrelated topic")
        assert result.confidence == 0.0
        assert result.intent == "general"


class TestBatch:
    def test_classify_batch(self):
        classifier = _make_classifier()
        results = classifier.classify_batch([
            "Write a python function",
            "Select from database table",
            "Hello world",
        ])
        assert len(results) == 3
        assert results[0].intent == "code_gen"
        assert results[1].intent == "data_query"
        assert results[2].intent == "general"


class TestRisk:
    def test_get_risk(self):
        classifier = _make_classifier()
        assert classifier.get_risk("admin") == "critical"
        assert classifier.get_risk("code_gen") == "medium"
        assert classifier.get_risk("data_query") == "low"

    def test_missing_intent_raises(self):
        classifier = _make_classifier()
        with pytest.raises(KeyError, match="not found"):
            classifier.get_risk("nonexistent")


class TestStats:
    def test_stats_tracking(self):
        classifier = _make_classifier()
        classifier.classify("Write python code")
        classifier.classify("Hello there")
        stats = classifier.stats()
        assert stats.total_classified == 2
        assert isinstance(stats.avg_confidence, float)

    def test_stats_by_intent(self):
        classifier = _make_classifier()
        classifier.classify("Write code function")
        classifier.classify("Write python code")
        classifier.classify("Select from database")
        stats = classifier.stats()
        assert stats.by_intent.get("code_gen") == 2
        assert stats.by_intent.get("data_query") == 1


class TestEdge:
    def test_empty_text(self):
        classifier = _make_classifier()
        result = classifier.classify("")
        assert result.intent == "general"
        assert result.confidence == 0.0
        assert result.matched_keywords == []

    def test_overlapping_intents(self):
        """When two intents share a keyword, score decides the winner."""
        classifier = IntentClassifier()
        classifier.add_intent("intent_a", keywords=["deploy", "server", "run"])
        classifier.add_intent("intent_b", keywords=["deploy", "container"])
        result = classifier.classify("deploy the server and run it")
        assert result.intent == "intent_a"
        assert result.confidence > 0.0
