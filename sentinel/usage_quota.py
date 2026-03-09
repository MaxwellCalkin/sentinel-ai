"""Usage quota enforcement for LLM APIs.

Track and enforce per-user and per-team quotas for API
calls, tokens, and costs with configurable time windows.

Usage:
    from sentinel.usage_quota import UsageQuota

    quota = UsageQuota()
    quota.set_limit("user-1", max_requests=100, window_hours=24)
    allowed = quota.check("user-1")
    print(allowed.within_quota)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuotaLimit:
    """Quota limit configuration."""
    entity_id: str
    max_requests: int
    max_tokens: int
    window_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaStatus:
    """Current quota status for an entity."""
    entity_id: str
    within_quota: bool
    requests_used: int
    requests_remaining: int
    tokens_used: int
    tokens_remaining: int
    reset_in_seconds: float
    utilization: float  # 0.0 to 1.0


@dataclass
class QuotaUsageRecord:
    """A single usage record."""
    entity_id: str
    requests: int
    tokens: int
    timestamp: float


class UsageQuota:
    """Enforce usage quotas for LLM APIs."""

    def __init__(self, default_max_requests: int = 1000, default_max_tokens: int = 100000, default_window_hours: float = 24) -> None:
        self._default_max_requests = default_max_requests
        self._default_max_tokens = default_max_tokens
        self._default_window = default_window_hours * 3600
        self._limits: dict[str, QuotaLimit] = {}
        self._usage: dict[str, list[QuotaUsageRecord]] = {}

    def set_limit(self, entity_id: str, max_requests: int | None = None, max_tokens: int | None = None, window_hours: float | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Set quota limit for an entity."""
        self._limits[entity_id] = QuotaLimit(
            entity_id=entity_id,
            max_requests=max_requests if max_requests is not None else self._default_max_requests,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
            window_seconds=window_hours * 3600 if window_hours is not None else self._default_window,
            metadata=metadata or {},
        )

    def record(self, entity_id: str, requests: int = 1, tokens: int = 0) -> None:
        """Record usage for an entity."""
        if entity_id not in self._usage:
            self._usage[entity_id] = []
        self._usage[entity_id].append(QuotaUsageRecord(
            entity_id=entity_id, requests=requests, tokens=tokens, timestamp=time.time(),
        ))

    def check(self, entity_id: str) -> QuotaStatus:
        """Check quota status for an entity."""
        limit = self._limits.get(entity_id)
        if not limit:
            limit = QuotaLimit(
                entity_id=entity_id,
                max_requests=self._default_max_requests,
                max_tokens=self._default_max_tokens,
                window_seconds=self._default_window,
            )

        now = time.time()
        window_start = now - limit.window_seconds
        records = self._usage.get(entity_id, [])
        active = [r for r in records if r.timestamp >= window_start]

        requests_used = sum(r.requests for r in active)
        tokens_used = sum(r.tokens for r in active)
        requests_remaining = max(0, limit.max_requests - requests_used)
        tokens_remaining = max(0, limit.max_tokens - tokens_used)

        within_quota = requests_used < limit.max_requests and tokens_used < limit.max_tokens

        # Find reset time
        if active:
            oldest = min(r.timestamp for r in active)
            reset_in = (oldest + limit.window_seconds) - now
        else:
            reset_in = 0.0

        req_util = requests_used / limit.max_requests if limit.max_requests > 0 else 0
        tok_util = tokens_used / limit.max_tokens if limit.max_tokens > 0 else 0
        utilization = max(req_util, tok_util)

        return QuotaStatus(
            entity_id=entity_id,
            within_quota=within_quota,
            requests_used=requests_used,
            requests_remaining=requests_remaining,
            tokens_used=tokens_used,
            tokens_remaining=tokens_remaining,
            reset_in_seconds=round(max(0, reset_in), 2),
            utilization=round(min(1.0, utilization), 4),
        )

    def consume(self, entity_id: str, requests: int = 1, tokens: int = 0) -> QuotaStatus:
        """Record usage and return updated status."""
        self.record(entity_id, requests, tokens)
        return self.check(entity_id)

    def reset(self, entity_id: str) -> bool:
        """Reset usage for an entity."""
        if entity_id in self._usage:
            self._usage[entity_id] = []
            return True
        return False

    def list_entities(self) -> list[str]:
        """List all entities with limits or usage."""
        entities = set(self._limits.keys()) | set(self._usage.keys())
        return sorted(entities)
