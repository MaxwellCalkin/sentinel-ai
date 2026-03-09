"""Deterministic response cache for LLM outputs.

Cache LLM responses by exact input hash for faster repeat
queries. Supports TTL expiration, LRU eviction, and
namespace isolation.

Usage:
    from sentinel.response_cache import ResponseCache

    cache = ResponseCache(max_entries=1000, default_ttl=3600)
    cache.set("What is Python?", "Python is a programming language...")
    hit = cache.get("What is Python?")
    print(hit.response)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A cached response."""
    key_hash: str
    query: str
    response: str
    timestamp: float
    ttl: float | None
    hits: int = 0
    namespace: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl


@dataclass
class CacheResponse:
    """Response from cache lookup."""
    hit: bool
    response: str = ""
    age_seconds: float = 0.0
    hits: int = 0


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    total_entries: int
    total_hits: int
    total_misses: int
    hit_rate: float
    namespaces: int
    evictions: int


class ResponseCache:
    """Deterministic LLM response cache."""

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: float | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._total_hits = 0
        self._total_misses = 0
        self._evictions = 0

    def _make_key(self, query: str, namespace: str = "default") -> str:
        """Create hash key from query and namespace."""
        raw = f"{namespace}:{query}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def set(
        self,
        query: str,
        response: str,
        ttl: float | None = None,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Cache a response."""
        key = self._make_key(query, namespace)
        entry = CacheEntry(
            key_hash=key, query=query, response=response,
            timestamp=time.time(), ttl=ttl if ttl is not None else self._default_ttl,
            namespace=namespace, metadata=metadata or {},
        )
        self._cache[key] = entry

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        # Evict if over limit
        while len(self._cache) > self._max_entries:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
            self._evictions += 1

    def get(self, query: str, namespace: str = "default") -> CacheResponse:
        """Look up a cached response."""
        key = self._make_key(query, namespace)
        entry = self._cache.get(key)

        if entry is None or entry.expired:
            if entry and entry.expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            self._total_misses += 1
            return CacheResponse(hit=False)

        entry.hits += 1
        self._total_hits += 1

        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        age = time.time() - entry.timestamp
        return CacheResponse(
            hit=True, response=entry.response,
            age_seconds=round(age, 2), hits=entry.hits,
        )

    def invalidate(self, query: str, namespace: str = "default") -> bool:
        """Remove a cached entry."""
        key = self._make_key(query, namespace)
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    def clear(self, namespace: str | None = None) -> int:
        """Clear cache entries. If namespace given, clear only that namespace."""
        if namespace:
            keys = [k for k, v in self._cache.items() if v.namespace == namespace]
            for k in keys:
                del self._cache[k]
                if k in self._access_order:
                    self._access_order.remove(k)
            return len(keys)
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        return count

    def metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        namespaces = len(set(e.namespace for e in self._cache.values()))
        total = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total if total > 0 else 0.0
        return CacheMetrics(
            total_entries=len(self._cache),
            total_hits=self._total_hits,
            total_misses=self._total_misses,
            hit_rate=round(hit_rate, 4),
            namespaces=namespaces,
            evictions=self._evictions,
        )

    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)
