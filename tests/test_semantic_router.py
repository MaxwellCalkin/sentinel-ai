"""Tests for semantic router."""

import pytest

from sentinel.semantic_router import SemanticRouter, RouteResult


class TestRouting:
    def test_keyword_routing(self):
        r = SemanticRouter()
        r.add_route("coding", handler="code_model", keywords=["code", "program", "function"])
        r.add_route("creative", handler="creative_model", keywords=["story", "poem", "write"])
        result = r.route("Write a function to sort a list")
        assert result.route == "coding"

    def test_creative_routing(self):
        r = SemanticRouter()
        r.add_route("coding", handler="code_model", keywords=["code", "debug", "function"])
        r.add_route("creative", handler="creative_model", keywords=["story", "poem", "creative"])
        result = r.route("Write a creative poem about nature")
        assert result.route == "creative"

    def test_default_route(self):
        r = SemanticRouter(default_route="fallback", default_handler="default_model")
        r.add_route("coding", handler="code_model", keywords=["code"])
        result = r.route("What is the weather today?")
        assert result.route == "fallback"
        assert result.confidence == 0.0

    def test_no_routes(self):
        r = SemanticRouter()
        result = r.route("Hello")
        assert result.route == "general"

    def test_pattern_routing(self):
        r = SemanticRouter()
        r.add_route("sql", handler="db_model", patterns=[r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b"])
        result = r.route("Help me write a SELECT query for users")
        assert result.route == "sql"


class TestScoring:
    def test_confidence_range(self):
        r = SemanticRouter()
        r.add_route("test", handler="h", keywords=["hello", "world"])
        result = r.route("hello world")
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_present(self):
        r = SemanticRouter()
        r.add_route("a", handler="h1", keywords=["alpha"])
        r.add_route("b", handler="h2", keywords=["beta"])
        result = r.route("alpha test")
        assert "a" in result.scores
        assert "b" in result.scores

    def test_matched_keywords(self):
        r = SemanticRouter()
        r.add_route("test", handler="h", keywords=["code", "python", "debug"])
        result = r.route("Debug my Python code")
        assert len(result.matched_keywords) >= 2

    def test_weighted_route(self):
        r = SemanticRouter()
        r.add_route("low", handler="h1", keywords=["test"], weight=0.1)
        r.add_route("high", handler="h2", keywords=["test"], weight=10.0)
        result = r.route("This is a test")
        assert result.route == "high"


class TestBatch:
    def test_batch_routing(self):
        r = SemanticRouter()
        r.add_route("code", handler="h", keywords=["code", "program"])
        r.add_route("chat", handler="h", keywords=["hello", "hi"])
        results = r.route_batch(["Write code for me", "Hello there"])
        assert len(results) == 2
        assert results[0].route == "code"
        assert results[1].route == "chat"


class TestManagement:
    def test_list_routes(self):
        r = SemanticRouter()
        r.add_route("a", handler="h1")
        r.add_route("b", handler="h2")
        assert r.list_routes() == ["a", "b"]


class TestStructure:
    def test_result_structure(self):
        r = SemanticRouter()
        r.add_route("test", handler="model", keywords=["test"])
        result = r.route("test input")
        assert isinstance(result, RouteResult)
        assert isinstance(result.handler, str)
        assert isinstance(result.scores, dict)
        assert isinstance(result.matched_keywords, list)
