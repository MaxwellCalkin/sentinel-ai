"""Tests for response cache."""

import pytest
import time
from sentinel.response_cache import ResponseCache, CacheResponse, CacheMetrics


class TestBasicOps:
    def test_set_and_get(self):
        c = ResponseCache()
        c.set("What is Python?", "A programming language")
        result = c.get("What is Python?")
        assert result.hit
        assert result.response == "A programming language"

    def test_cache_miss(self):
        c = ResponseCache()
        result = c.get("Unknown query")
        assert not result.hit
        assert result.response == ""

    def test_overwrite(self):
        c = ResponseCache()
        c.set("q", "old answer")
        c.set("q", "new answer")
        result = c.get("q")
        assert result.response == "new answer"

    def test_size(self):
        c = ResponseCache()
        c.set("a", "1")
        c.set("b", "2")
        assert c.size() == 2


class TestTTL:
    def test_no_ttl(self):
        c = ResponseCache()
        c.set("q", "answer")
        result = c.get("q")
        assert result.hit

    def test_expired_entry(self):
        c = ResponseCache(default_ttl=0.01)
        c.set("q", "answer")
        time.sleep(0.02)
        result = c.get("q")
        assert not result.hit

    def test_custom_ttl(self):
        c = ResponseCache()
        c.set("q", "answer", ttl=0.01)
        time.sleep(0.02)
        result = c.get("q")
        assert not result.hit


class TestEviction:
    def test_lru_eviction(self):
        c = ResponseCache(max_entries=2)
        c.set("a", "1")
        c.set("b", "2")
        c.set("c", "3")  # should evict "a"
        assert c.get("a").hit is False
        assert c.get("b").hit is True
        assert c.get("c").hit is True

    def test_eviction_count(self):
        c = ResponseCache(max_entries=2)
        c.set("a", "1")
        c.set("b", "2")
        c.set("c", "3")
        assert c.metrics().evictions >= 1


class TestNamespace:
    def test_namespace_isolation(self):
        c = ResponseCache()
        c.set("q", "answer-a", namespace="ns-a")
        c.set("q", "answer-b", namespace="ns-b")
        assert c.get("q", namespace="ns-a").response == "answer-a"
        assert c.get("q", namespace="ns-b").response == "answer-b"

    def test_clear_namespace(self):
        c = ResponseCache()
        c.set("a", "1", namespace="ns1")
        c.set("b", "2", namespace="ns2")
        cleared = c.clear(namespace="ns1")
        assert cleared == 1
        assert c.size() == 1


class TestInvalidate:
    def test_invalidate(self):
        c = ResponseCache()
        c.set("q", "answer")
        assert c.invalidate("q")
        assert not c.get("q").hit

    def test_invalidate_missing(self):
        c = ResponseCache()
        assert not c.invalidate("nonexistent")


class TestMetrics:
    def test_hit_rate(self):
        c = ResponseCache()
        c.set("q", "answer")
        c.get("q")  # hit
        c.get("q")  # hit
        c.get("missing")  # miss
        m = c.metrics()
        assert m.total_hits == 2
        assert m.total_misses == 1
        assert abs(m.hit_rate - 0.6667) < 0.01

    def test_namespace_count(self):
        c = ResponseCache()
        c.set("a", "1", namespace="x")
        c.set("b", "2", namespace="y")
        assert c.metrics().namespaces == 2


class TestStructure:
    def test_cache_response(self):
        c = ResponseCache()
        c.set("q", "answer")
        result = c.get("q")
        assert isinstance(result, CacheResponse)
        assert result.age_seconds >= 0
        assert result.hits >= 1

    def test_metrics_structure(self):
        c = ResponseCache()
        m = c.metrics()
        assert isinstance(m, CacheMetrics)
        assert m.total_entries == 0
        assert m.hit_rate == 0.0
