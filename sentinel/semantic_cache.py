"""Semantic caching for LLM responses.

Cache LLM responses and match new queries by text similarity,
reducing API calls and latency for similar questions.

Usage:
    from sentinel.semantic_cache import SemanticCache

    cache = SemanticCache(threshold=0.85)
    cache.put("What is Python?", "Python is a programming language...")
    hit = cache.get("Tell me about Python")
    if hit:
        print(hit.response)  # Cached response
"""

from __future__ import annotations

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could must need dare "
    "i you he she it we they me him her us them my your his its our their "
    "this that these those what which who whom whose where when how why "
    "in on at to for from by with of and or but not no nor so yet".split()
)

_WORD_RE = re.compile(r'[a-z0-9]+')


@dataclass
class CacheEntry:
    """A cached query-response pair."""
    query: str
    response: str
    tokens: dict[str, float]  # TF vector
    timestamp: float
    ttl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)


@dataclass
class CacheHit:
    """Result of a cache lookup."""
    query: str
    response: str
    similarity: float
    original_query: str
    metadata: dict[str, Any]


@dataclass
class CacheStats:
    """Cache statistics."""
    entries: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int


class SemanticCache:
    """Cache LLM responses with semantic matching.

    Uses TF-IDF cosine similarity to match queries against
    cached entries. Zero external dependencies.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        max_entries: int = 1000,
        default_ttl: float | None = None,
    ) -> None:
        """
        Args:
            threshold: Minimum similarity to consider a cache hit.
            max_entries: Maximum cache size (LRU eviction).
            default_ttl: Default TTL in seconds. None = no expiry.
        """
        self._threshold = threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._entries: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _tokenize(self, text: str) -> dict[str, float]:
        """Tokenize text into TF vector."""
        words = _WORD_RE.findall(text.lower())
        words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
        if not words:
            return {}
        tf: dict[str, float] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        # Normalize
        total = len(words)
        return {w: c / total for w, c in tf.items()}

    def _cosine_similarity(self, a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        mag_a = sum(v * v for v in a.values()) ** 0.5
        mag_b = sum(v * v for v in b.values()) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def put(
        self,
        query: str,
        response: str,
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a query-response pair to cache.

        Args:
            query: The user query.
            response: The LLM response.
            ttl: TTL in seconds. None uses default_ttl.
            metadata: Additional metadata.
        """
        # Evict if at capacity
        while len(self._entries) >= self._max_entries:
            self._entries.pop(0)  # LRU: remove oldest
            self._evictions += 1

        self._entries.append(CacheEntry(
            query=query,
            response=response,
            tokens=self._tokenize(query),
            timestamp=time.time(),
            ttl=ttl if ttl is not None else self._default_ttl,
            metadata=metadata or {},
        ))

    def get(self, query: str) -> CacheHit | None:
        """Look up a query in the cache.

        Args:
            query: The query to look up.

        Returns:
            CacheHit if similar query found, None otherwise.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            self._misses += 1
            return None

        best_sim = 0.0
        best_entry: CacheEntry | None = None

        # Remove expired entries
        self._entries = [e for e in self._entries if not e.expired]

        for entry in self._entries:
            sim = self._cosine_similarity(query_tokens, entry.tokens)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self._threshold:
            best_entry.hit_count += 1
            self._hits += 1
            return CacheHit(
                query=query,
                response=best_entry.response,
                similarity=round(best_sim, 4),
                original_query=best_entry.query,
                metadata=best_entry.metadata,
            )

        self._misses += 1
        return None

    def invalidate(self, query: str) -> bool:
        """Remove a specific cached entry by exact query match."""
        for i, entry in enumerate(self._entries):
            if entry.query == query:
                self._entries.pop(i)
                return True
        return False

    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        count = len(self._entries)
        self._entries.clear()
        return count

    @property
    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> CacheStats:
        total = self._hits + self._misses
        return CacheStats(
            entries=len(self._entries),
            hits=self._hits,
            misses=self._misses,
            hit_rate=self._hits / total if total > 0 else 0.0,
            evictions=self._evictions,
        )
