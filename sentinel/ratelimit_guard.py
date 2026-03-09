"""Token bucket rate limiter for LLM API calls.

Protect against API abuse and cost overruns with configurable
per-user, per-model, and global rate limits.

Usage:
    from sentinel.ratelimit_guard import RateLimitGuard

    limiter = RateLimitGuard(requests_per_minute=60)
    if limiter.allow("user_123"):
        response = llm.complete(prompt)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    retry_after: float | None = None  # Seconds until next token available
    bucket: str = "global"


@dataclass
class BucketStats:
    """Statistics for a rate limit bucket."""
    bucket: str
    tokens: float
    capacity: int
    requests_total: int
    requests_denied: int
    denial_rate: float


class TokenBucket:
    """Thread-safe token bucket implementation."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        """
        Args:
            capacity: Maximum tokens in bucket.
            refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.requests_total = 0
        self.requests_denied = 0
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_consume(self, tokens: int = 1) -> RateLimitResult:
        with self._lock:
            self._refill()
            self.requests_total += 1

            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.tokens),
                    limit=self.capacity,
                )
            else:
                self.requests_denied += 1
                deficit = tokens - self.tokens
                retry_after = deficit / self.refill_rate if self.refill_rate > 0 else None
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.capacity,
                    retry_after=round(retry_after, 3) if retry_after else None,
                )

    def stats(self, bucket_name: str) -> BucketStats:
        with self._lock:
            self._refill()
            return BucketStats(
                bucket=bucket_name,
                tokens=round(self.tokens, 2),
                capacity=self.capacity,
                requests_total=self.requests_total,
                requests_denied=self.requests_denied,
                denial_rate=(
                    self.requests_denied / self.requests_total
                    if self.requests_total > 0 else 0.0
                ),
            )


class RateLimitGuard:
    """Rate limiter with per-key token buckets.

    Supports per-user, per-model, or any key-based rate limiting
    with configurable capacity and refill rates.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int | None = None,
    ) -> None:
        """
        Args:
            requests_per_minute: Sustained request rate.
            burst: Max burst size. Defaults to requests_per_minute.
        """
        self._rpm = requests_per_minute
        self._burst = burst or requests_per_minute
        self._refill_rate = requests_per_minute / 60.0
        self._buckets: dict[str, TokenBucket] = {}
        self._custom_limits: dict[str, tuple[int, int]] = {}  # key -> (rpm, burst)
        self._lock = threading.Lock()

    def set_limit(self, key: str, requests_per_minute: int, burst: int | None = None) -> None:
        """Set custom rate limit for a specific key."""
        self._custom_limits[key] = (requests_per_minute, burst or requests_per_minute)
        # Reset bucket if it exists
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    def _get_bucket(self, key: str) -> TokenBucket:
        with self._lock:
            if key not in self._buckets:
                if key in self._custom_limits:
                    rpm, burst = self._custom_limits[key]
                else:
                    rpm, burst = self._rpm, self._burst
                self._buckets[key] = TokenBucket(
                    capacity=burst,
                    refill_rate=rpm / 60.0,
                )
            return self._buckets[key]

    def allow(self, key: str = "global", cost: int = 1) -> RateLimitResult:
        """Check if request is allowed.

        Args:
            key: Rate limit bucket key (user_id, model, etc.).
            cost: Token cost of this request.

        Returns:
            RateLimitResult with allowed status and remaining tokens.
        """
        bucket = self._get_bucket(key)
        result = bucket.try_consume(cost)
        result.bucket = key
        return result

    def wait(self, key: str = "global", cost: int = 1, timeout: float = 30.0) -> RateLimitResult:
        """Wait until request is allowed or timeout.

        Args:
            key: Rate limit bucket key.
            cost: Token cost.
            timeout: Max seconds to wait.

        Returns:
            RateLimitResult (allowed=False if timed out).
        """
        deadline = time.monotonic() + timeout
        while True:
            result = self.allow(key, cost)
            if result.allowed:
                return result
            if time.monotonic() >= deadline:
                return result
            sleep_time = min(result.retry_after or 0.1, deadline - time.monotonic())
            if sleep_time <= 0:
                return result
            time.sleep(sleep_time)

    def stats(self, key: str = "global") -> BucketStats | None:
        """Get stats for a bucket."""
        with self._lock:
            bucket = self._buckets.get(key)
        if bucket is None:
            return None
        return bucket.stats(key)

    def all_stats(self) -> dict[str, BucketStats]:
        """Get stats for all buckets."""
        with self._lock:
            keys = list(self._buckets.keys())
        return {k: self._buckets[k].stats(k) for k in keys}

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    def reset(self, key: str | None = None) -> None:
        """Reset one or all buckets."""
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()
