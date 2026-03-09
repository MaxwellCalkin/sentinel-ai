"""Tests for sentinel.safety_cache — SafetyCache module."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from sentinel.safety_cache import (
    CacheConfig,
    CacheEntry,
    CacheReport,
    CacheStats,
    SafetyCache,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

class TestCacheEntryDataclass:
    def test_create_with_defaults(self):
        entry = CacheEntry(key="k", value="v", created_at=1.0, expires_at=2.0)
        assert entry.key == "k"
        assert entry.value == "v"
        assert entry.hits == 0
        assert entry.tags == []

    def test_expired_property_returns_true_when_past(self):
        entry = CacheEntry(key="k", value="v", created_at=0.0, expires_at=0.0)
        assert entry.expired is True

    def test_expired_property_returns_false_when_future(self):
        entry = CacheEntry(
            key="k", value="v", created_at=0.0, expires_at=time.time() + 9999
        )
        assert entry.expired is False


class TestCacheConfigDataclass:
    def test_defaults(self):
        config = CacheConfig()
        assert config.ttl_seconds == 300.0
        assert config.max_entries == 1000
        assert config.eviction == "lru"

    def test_invalid_eviction_raises(self):
        with pytest.raises(ValueError, match="Invalid eviction strategy"):
            CacheConfig(eviction="random")

    def test_invalid_max_entries_raises(self):
        with pytest.raises(ValueError, match="max_entries must be at least 1"):
            CacheConfig(max_entries=0)

    def test_invalid_ttl_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            CacheConfig(ttl_seconds=-1)


# ---------------------------------------------------------------------------
# Basic get / set
# ---------------------------------------------------------------------------

class TestSetAndGet:
    def test_set_then_get_returns_value(self):
        cache = SafetyCache()
        cache.set("key1", {"safe": True})
        assert cache.get("key1") == {"safe": True}

    def test_get_missing_key_returns_none(self):
        cache = SafetyCache()
        assert cache.get("nonexistent") is None

    def test_set_overwrites_existing_key(self):
        cache = SafetyCache()
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------

class TestTTLExpiration:
    def test_expired_entry_returns_none_on_get(self):
        cache = SafetyCache(CacheConfig(ttl_seconds=0.01))
        cache.set("k", "v")
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_non_expired_entry_returns_value(self):
        cache = SafetyCache(CacheConfig(ttl_seconds=60))
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_per_entry_ttl_override(self):
        cache = SafetyCache(CacheConfig(ttl_seconds=60))
        cache.set("short", "v", ttl=0.01)
        cache.set("long", "v", ttl=9999)
        time.sleep(0.02)
        assert cache.get("short") is None
        assert cache.get("long") == "v"

    def test_expired_entry_cleaned_on_get(self):
        cache = SafetyCache(CacheConfig(ttl_seconds=0.01))
        cache.set("k", "v")
        time.sleep(0.02)
        cache.get("k")
        assert cache.has("k") is False


# ---------------------------------------------------------------------------
# has / delete
# ---------------------------------------------------------------------------

class TestHasAndDelete:
    def test_has_returns_true_for_existing(self):
        cache = SafetyCache()
        cache.set("k", "v")
        assert cache.has("k") is True

    def test_has_returns_false_for_missing(self):
        cache = SafetyCache()
        assert cache.has("nope") is False

    def test_has_returns_false_for_expired(self):
        cache = SafetyCache(CacheConfig(ttl_seconds=0.01))
        cache.set("k", "v")
        time.sleep(0.02)
        assert cache.has("k") is False

    def test_delete_existing_returns_true(self):
        cache = SafetyCache()
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_delete_missing_returns_false(self):
        cache = SafetyCache()
        assert cache.delete("nope") is False


# ---------------------------------------------------------------------------
# Tag-based invalidation
# ---------------------------------------------------------------------------

class TestTagInvalidation:
    def test_invalidate_by_tag_removes_matching(self):
        cache = SafetyCache()
        cache.set("a", 1, tags=["user-1"])
        cache.set("b", 2, tags=["user-1"])
        cache.set("c", 3, tags=["user-2"])
        count = cache.invalidate_by_tag("user-1")
        assert count == 2
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_invalidate_by_tag_returns_zero_when_none_match(self):
        cache = SafetyCache()
        cache.set("a", 1, tags=["x"])
        assert cache.invalidate_by_tag("y") == 0

    def test_entry_with_multiple_tags(self):
        cache = SafetyCache()
        cache.set("a", 1, tags=["alpha", "beta"])
        assert cache.invalidate_by_tag("beta") == 1
        assert cache.has("a") is False


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all_entries(self):
        cache = SafetyCache()
        for i in range(5):
            cache.set(f"k{i}", i)
        cache.clear()
        for i in range(5):
            assert cache.get(f"k{i}") is None

    def test_clear_on_empty_cache(self):
        cache = SafetyCache()
        cache.clear()
        assert cache.report().total_entries == 0


# ---------------------------------------------------------------------------
# Eviction strategies
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_evicts_least_recently_used(self):
        cache = SafetyCache(CacheConfig(max_entries=3, eviction="lru"))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it recently used
        cache.get("a")

        # Adding "d" should evict "b" (least recently used)
        cache.set("d", 4)
        assert cache.has("a") is True
        assert cache.has("b") is False
        assert cache.has("c") is True
        assert cache.has("d") is True


class TestFIFOEviction:
    def test_evicts_oldest_created(self):
        cache = SafetyCache(CacheConfig(max_entries=3, eviction="fifo"))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" — should not matter for FIFO
        cache.get("a")

        # Adding "d" should evict "a" (created first)
        cache.set("d", 4)
        assert cache.has("a") is False
        assert cache.has("b") is True
        assert cache.has("c") is True
        assert cache.has("d") is True


class TestLFUEviction:
    def test_evicts_least_frequently_used(self):
        cache = SafetyCache(CacheConfig(max_entries=3, eviction="lfu"))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" three times and "c" once; "b" stays at 0 hits
        cache.get("a")
        cache.get("a")
        cache.get("a")
        cache.get("c")

        # Adding "d" should evict "b" (least frequently used)
        cache.set("d", 4)
        assert cache.has("a") is True
        assert cache.has("b") is False
        assert cache.has("c") is True
        assert cache.has("d") is True


class TestMaxEntriesEnforcement:
    def test_never_exceeds_max_entries(self):
        cache = SafetyCache(CacheConfig(max_entries=5))
        for i in range(20):
            cache.set(f"k{i}", i)
        assert cache.report().total_entries <= 5

    def test_eviction_count_tracked(self):
        cache = SafetyCache(CacheConfig(max_entries=2))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts one
        cache.set("d", 4)  # evicts one
        assert cache.stats().evictions == 2


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_empty_cache(self):
        cache = SafetyCache()
        report = cache.report()
        assert report.total_entries == 0
        assert report.hit_rate == 0.0
        assert report.miss_rate == 0.0
        assert report.evictions == 0
        assert report.avg_age == 0.0

    def test_report_reflects_state(self):
        cache = SafetyCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")       # hit
        cache.get("missing")  # miss

        report = cache.report()
        assert report.total_entries == 2
        assert report.hit_rate == 0.5
        assert report.miss_rate == 0.5
        assert report.evictions == 0
        assert report.avg_age >= 0.0

    def test_report_type(self):
        cache = SafetyCache()
        assert isinstance(cache.report(), CacheReport)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats_all_zero(self):
        cache = SafetyCache()
        stats = cache.stats()
        assert stats.total_gets == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.evictions == 0

    def test_stats_track_operations(self):
        cache = SafetyCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")       # hit
        cache.get("a")       # hit
        cache.get("missing")  # miss

        stats = cache.stats()
        assert stats.sets == 2
        assert stats.total_gets == 3
        assert stats.hits == 2
        assert stats.misses == 1

    def test_stats_type(self):
        cache = SafetyCache()
        assert isinstance(cache.stats(), CacheStats)

    def test_hit_rate_calculation(self):
        cache = SafetyCache()
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("b")  # miss

        report = cache.report()
        expected_hit_rate = 2 / 3
        assert abs(report.hit_rate - round(expected_hit_rate, 4)) < 0.001

    def test_stats_returns_copy(self):
        cache = SafetyCache()
        stats1 = cache.stats()
        cache.set("k", "v")
        stats2 = cache.stats()
        assert stats1.sets == 0
        assert stats2.sets == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_value_can_be_cached(self):
        cache = SafetyCache()
        cache.set("k", None)
        # Distinguish between "key not found" and "value is None"
        assert cache.has("k") is True
        assert cache.get("k") is None  # value is None, not a miss — but API returns None

    def test_set_with_empty_tags(self):
        cache = SafetyCache()
        cache.set("k", "v", tags=[])
        assert cache.get("k") == "v"

    def test_default_config_when_none(self):
        cache = SafetyCache(config=None)
        assert cache._config.ttl_seconds == 300.0
        assert cache._config.max_entries == 1000
        assert cache._config.eviction == "lru"

    def test_overwrite_preserves_capacity(self):
        cache = SafetyCache(CacheConfig(max_entries=2))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 99)  # overwrite, should not trigger eviction
        assert cache.report().total_entries == 2
        assert cache.stats().evictions == 0
        assert cache.get("a") == 99
        assert cache.get("b") == 2

    def test_miss_increments_miss_stat_on_expired(self):
        cache = SafetyCache(CacheConfig(ttl_seconds=0.01))
        cache.set("k", "v")
        time.sleep(0.02)
        cache.get("k")
        assert cache.stats().misses == 1
        assert cache.stats().hits == 0
