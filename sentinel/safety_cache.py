"""Safety scan result caching with configurable TTL and eviction.

Cache safety scan results to avoid re-scanning identical inputs.
Supports LRU, FIFO, and LFU eviction strategies with tag-based
invalidation.

Usage:
    from sentinel.safety_cache import SafetyCache, CacheConfig

    cache = SafetyCache(CacheConfig(ttl_seconds=600, max_entries=500))
    cache.set("prompt-hash", {"safe": True}, tags=["user-123"])
    result = cache.get("prompt-hash")
    count = cache.invalidate_by_tag("user-123")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


VALID_EVICTION_STRATEGIES = ("lru", "fifo", "lfu")


@dataclass
class CacheEntry:
    """A single cached safety scan result."""
    key: str
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    tags: list[str] = field(default_factory=list)
    last_accessed_at: float = 0.0

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class CacheConfig:
    """Configuration for SafetyCache."""
    ttl_seconds: float = 300.0
    max_entries: int = 1000
    eviction: str = "lru"

    def __post_init__(self) -> None:
        if self.eviction not in VALID_EVICTION_STRATEGIES:
            raise ValueError(
                f"Invalid eviction strategy '{self.eviction}'. "
                f"Must be one of: {', '.join(VALID_EVICTION_STRATEGIES)}"
            )
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")


@dataclass
class CacheReport:
    """Snapshot report of cache state and performance."""
    total_entries: int
    hit_rate: float
    miss_rate: float
    evictions: int
    avg_age: float


@dataclass
class CacheStats:
    """Cumulative cache operation statistics."""
    total_gets: int = 0
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0


class SafetyCache:
    """Cache safety scan results with TTL and configurable eviction.

    Supports three eviction strategies:
    - LRU: evict least recently accessed entry
    - FIFO: evict oldest entry by creation time
    - LFU: evict least frequently accessed entry
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self._config = config or CacheConfig()
        self._entries: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key.

        Returns None if the key does not exist or has expired.
        Updates access tracking for LRU eviction on hit.
        """
        self._stats.total_gets += 1
        entry = self._entries.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.expired:
            del self._entries[key]
            self._stats.misses += 1
            return None

        entry.hits += 1
        entry.last_accessed_at = time.time()
        self._stats.hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            tags: Optional tags for group invalidation.
            ttl: Per-entry TTL override in seconds.
        """
        self._stats.sets += 1
        effective_ttl = ttl if ttl is not None else self._config.ttl_seconds
        now = time.time()

        if key in self._entries:
            self._entries[key] = self._build_entry(key, value, tags, now, effective_ttl)
            return

        self._evict_if_at_capacity()

        self._entries[key] = self._build_entry(key, value, tags, now, effective_ttl)

    def has(self, key: str) -> bool:
        """Check whether a non-expired entry exists for the key."""
        entry = self._entries.get(key)
        if entry is None:
            return False
        if entry.expired:
            del self._entries[key]
            return False
        return True

    def delete(self, key: str) -> bool:
        """Delete an entry by key. Returns True if the entry existed."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def invalidate_by_tag(self, tag: str) -> int:
        """Delete all entries matching the given tag. Returns count deleted."""
        keys_to_delete = [
            key for key, entry in self._entries.items()
            if tag in entry.tags
        ]
        for key in keys_to_delete:
            del self._entries[key]
        return len(keys_to_delete)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._entries.clear()

    def report(self) -> CacheReport:
        """Generate a snapshot report of cache state and performance."""
        now = time.time()
        total_entries = len(self._entries)
        total_lookups = self._stats.hits + self._stats.misses
        hit_rate = self._stats.hits / total_lookups if total_lookups > 0 else 0.0
        miss_rate = self._stats.misses / total_lookups if total_lookups > 0 else 0.0
        avg_age = self._compute_average_age(now)

        return CacheReport(
            total_entries=total_entries,
            hit_rate=round(hit_rate, 4),
            miss_rate=round(miss_rate, 4),
            evictions=self._stats.evictions,
            avg_age=round(avg_age, 2),
        )

    def stats(self) -> CacheStats:
        """Return cumulative operation statistics."""
        return CacheStats(
            total_gets=self._stats.total_gets,
            hits=self._stats.hits,
            misses=self._stats.misses,
            sets=self._stats.sets,
            evictions=self._stats.evictions,
        )

    def _build_entry(
        self,
        key: str,
        value: Any,
        tags: list[str] | None,
        now: float,
        ttl: float,
    ) -> CacheEntry:
        return CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
            tags=tags or [],
            last_accessed_at=now,
        )

    def _evict_if_at_capacity(self) -> None:
        while len(self._entries) >= self._config.max_entries:
            self._evict_one()

    def _evict_one(self) -> None:
        if not self._entries:
            return
        victim_key = self._select_eviction_victim()
        del self._entries[victim_key]
        self._stats.evictions += 1

    def _select_eviction_victim(self) -> str:
        strategy = self._config.eviction
        if strategy == "lru":
            return self._find_lru_victim()
        if strategy == "fifo":
            return self._find_fifo_victim()
        return self._find_lfu_victim()

    def _find_lru_victim(self) -> str:
        return min(self._entries, key=lambda k: self._entries[k].last_accessed_at)

    def _find_fifo_victim(self) -> str:
        return min(self._entries, key=lambda k: self._entries[k].created_at)

    def _find_lfu_victim(self) -> str:
        return min(self._entries, key=lambda k: self._entries[k].hits)

    def _compute_average_age(self, now: float) -> float:
        if not self._entries:
            return 0.0
        total_age = sum(now - entry.created_at for entry in self._entries.values())
        return total_age / len(self._entries)
