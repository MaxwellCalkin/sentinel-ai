"""Tests for semantic cache."""

import time
import pytest
from sentinel.semantic_cache import SemanticCache, CacheHit, CacheStats


# ---------------------------------------------------------------------------
# Basic put/get
# ---------------------------------------------------------------------------

class TestBasicPutGet:
    def test_exact_match(self):
        c = SemanticCache(threshold=0.8)
        c.put("What is Python?", "Python is a programming language.")
        hit = c.get("What is Python?")
        assert hit is not None
        assert hit.response == "Python is a programming language."
        assert hit.similarity == 1.0

    def test_similar_match(self):
        c = SemanticCache(threshold=0.5)
        c.put("How does Python programming work?", "Python uses an interpreter.")
        hit = c.get("Tell me about Python programming")
        assert hit is not None
        assert hit.similarity >= 0.5

    def test_no_match(self):
        c = SemanticCache(threshold=0.9)
        c.put("What is Python?", "A language.")
        hit = c.get("What is the weather today?")
        assert hit is None

    def test_empty_cache(self):
        c = SemanticCache()
        assert c.get("anything") is None

    def test_cache_size(self):
        c = SemanticCache()
        c.put("q1", "r1")
        c.put("q2", "r2")
        assert c.size == 2


# ---------------------------------------------------------------------------
# Cache hit details
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_hit_structure(self):
        c = SemanticCache(threshold=0.5)
        c.put("Python programming language", "Python is great", metadata={"source": "docs"})
        hit = c.get("Python programming language")
        assert isinstance(hit, CacheHit)
        assert hit.original_query == "Python programming language"
        assert hit.metadata["source"] == "docs"


# ---------------------------------------------------------------------------
# TTL / expiration
# ---------------------------------------------------------------------------

class TestTTL:
    def test_entry_not_expired(self):
        c = SemanticCache(threshold=0.5)
        c.put("Python language", "Answer", ttl=3600)
        assert c.get("Python language") is not None

    def test_entry_expired(self):
        c = SemanticCache(threshold=0.5)
        c.put("Python language", "Answer", ttl=0.001)
        time.sleep(0.01)
        assert c.get("Python language") is None

    def test_default_ttl(self):
        c = SemanticCache(threshold=0.5, default_ttl=0.001)
        c.put("Python language", "Answer")
        time.sleep(0.01)
        assert c.get("Python language") is None


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_max_entries(self):
        c = SemanticCache(threshold=0.5, max_entries=3)
        c.put("query one", "r1")
        c.put("query two", "r2")
        c.put("query three", "r3")
        c.put("query four", "r4")
        assert c.size == 3
        stats = c.stats()
        assert stats.evictions == 1


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

class TestInvalidation:
    def test_invalidate(self):
        c = SemanticCache(threshold=0.5)
        c.put("Python language", "Answer")
        assert c.invalidate("Python language")
        assert c.size == 0

    def test_invalidate_nonexistent(self):
        c = SemanticCache()
        assert not c.invalidate("nope")

    def test_clear(self):
        c = SemanticCache()
        c.put("q1", "r1")
        c.put("q2", "r2")
        count = c.clear()
        assert count == 2
        assert c.size == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_hit_rate(self):
        c = SemanticCache(threshold=0.5)
        c.put("Python programming", "Answer")
        c.get("Python programming")  # hit
        c.get("totally unrelated query about cooking")  # miss
        stats = c.stats()
        assert isinstance(stats, CacheStats)
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_empty_stats(self):
        c = SemanticCache()
        stats = c.stats()
        assert stats.hit_rate == 0.0
        assert stats.entries == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_stop_words_only(self):
        c = SemanticCache(threshold=0.5)
        c.put("the is a an", "response")
        # All stop words removed, should miss
        assert c.get("the is a an") is None

    def test_metadata_preserved(self):
        c = SemanticCache(threshold=0.5)
        c.put("Python basics", "response", metadata={"model": "claude"})
        hit = c.get("Python basics")
        assert hit.metadata["model"] == "claude"

    def test_multiple_similar(self):
        c = SemanticCache(threshold=0.3)
        c.put("Python programming language basics", "Answer 1")
        c.put("Python programming language advanced", "Answer 2")
        hit = c.get("Python programming language advanced topics")
        assert hit is not None
        # Should match the more similar one
        assert hit.response == "Answer 2"
