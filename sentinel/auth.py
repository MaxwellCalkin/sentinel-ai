"""API key authentication and rate limiting for the hosted API.

Usage with FastAPI:
    from sentinel.auth import require_api_key, RateLimiter

    app = create_app()
    app.dependency_overrides[...] = ...

Or standalone:
    limiter = RateLimiter(requests_per_minute=60)
    if not limiter.allow("client-key"):
        raise RateLimited()
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimitState:
    tokens: float
    last_refill: float


class RateLimiter:
    """Token bucket rate limiter. Thread-safe for single-process use."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int | None = None,
    ):
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = burst or requests_per_minute
        self._buckets: dict[str, RateLimitState] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()

        if key not in self._buckets:
            self._buckets[key] = RateLimitState(
                tokens=self._burst - 1, last_refill=now
            )
            return True

        state = self._buckets[key]
        elapsed = now - state.last_refill
        state.tokens = min(self._burst, state.tokens + elapsed * self._rate)
        state.last_refill = now

        if state.tokens >= 1:
            state.tokens -= 1
            return True
        return False

    def remaining(self, key: str) -> int:
        if key not in self._buckets:
            return self._burst
        return max(0, int(self._buckets[key].tokens))

    def reset(self, key: str | None = None) -> None:
        if key:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()


@dataclass
class APIKey:
    key_id: str
    key_hash: str
    name: str
    tier: str = "free"  # free, pro, enterprise
    rate_limit: int = 60  # requests per minute
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class APIKeyStore:
    """In-memory API key store. Override for database-backed stores."""

    def __init__(self):
        self._keys: dict[str, APIKey] = {}

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def register(self, raw_key: str, name: str, tier: str = "free", rate_limit: int = 60) -> APIKey:
        key_id = raw_key[:8] if len(raw_key) >= 8 else raw_key
        api_key = APIKey(
            key_id=key_id,
            key_hash=self.hash_key(raw_key),
            name=name,
            tier=tier,
            rate_limit=rate_limit,
        )
        self._keys[api_key.key_hash] = api_key
        return api_key

    def validate(self, raw_key: str) -> APIKey | None:
        h = self.hash_key(raw_key)
        key = self._keys.get(h)
        if key and key.enabled:
            return key
        return None

    def revoke(self, raw_key: str) -> bool:
        h = self.hash_key(raw_key)
        if h in self._keys:
            self._keys[h].enabled = False
            return True
        return False


def create_authenticated_app(
    key_store: APIKeyStore | None = None,
    rate_limiter: RateLimiter | None = None,
):
    """Create a FastAPI app with API key auth and rate limiting."""
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from sentinel.api import create_app

    if key_store is None:
        key_store = APIKeyStore()
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    app = create_app()

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Skip auth for health endpoint
        if request.url.path == "/health":
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid Authorization header. Use 'Bearer <api_key>'"},
            )

        raw_key = auth[7:]
        api_key = key_store.validate(raw_key)
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or revoked API key"},
            )

        if not rate_limiter.allow(api_key.key_id):
            remaining = rate_limiter.remaining(api_key.key_id)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "remaining": remaining},
                headers={"Retry-After": "60"},
            )

        request.state.api_key = api_key
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(
            rate_limiter.remaining(api_key.key_id)
        )
        return response

    return app, key_store, rate_limiter
