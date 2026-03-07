"""Tests for API key authentication and rate limiting."""

import time
from sentinel.auth import RateLimiter, APIKeyStore, APIKey


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.allow("client1") is True
        assert limiter.allow("client1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(requests_per_minute=60, burst=3)
        assert limiter.allow("c") is True
        assert limiter.allow("c") is True
        assert limiter.allow("c") is True
        assert limiter.allow("c") is False  # burst exhausted

    def test_separate_keys_independent(self):
        limiter = RateLimiter(requests_per_minute=60, burst=2)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False
        assert limiter.allow("b") is True  # different key, still has tokens

    def test_remaining_count(self):
        limiter = RateLimiter(requests_per_minute=60, burst=5)
        limiter.allow("x")
        assert limiter.remaining("x") == 4

    def test_remaining_unknown_key(self):
        limiter = RateLimiter(requests_per_minute=60, burst=10)
        assert limiter.remaining("unknown") == 10

    def test_reset_specific_key(self):
        limiter = RateLimiter(requests_per_minute=60, burst=2)
        limiter.allow("a")
        limiter.allow("a")
        limiter.reset("a")
        assert limiter.remaining("a") == 2

    def test_reset_all(self):
        limiter = RateLimiter(requests_per_minute=60, burst=2)
        limiter.allow("a")
        limiter.allow("b")
        limiter.reset()
        assert limiter.remaining("a") == 2
        assert limiter.remaining("b") == 2


class TestAPIKeyStore:
    def test_register_and_validate(self):
        store = APIKeyStore()
        store.register("sk-test-key-12345678", name="test", tier="pro")
        key = store.validate("sk-test-key-12345678")
        assert key is not None
        assert key.name == "test"
        assert key.tier == "pro"

    def test_invalid_key_rejected(self):
        store = APIKeyStore()
        store.register("sk-valid-key-abcdefgh", name="valid")
        assert store.validate("sk-wrong-key-12345678") is None

    def test_revoked_key_rejected(self):
        store = APIKeyStore()
        store.register("sk-revoke-me-12345678", name="revokable")
        assert store.validate("sk-revoke-me-12345678") is not None
        store.revoke("sk-revoke-me-12345678")
        assert store.validate("sk-revoke-me-12345678") is None

    def test_key_hash_is_deterministic(self):
        h1 = APIKeyStore.hash_key("test-key")
        h2 = APIKeyStore.hash_key("test-key")
        assert h1 == h2

    def test_different_keys_different_hashes(self):
        h1 = APIKeyStore.hash_key("key-a")
        h2 = APIKeyStore.hash_key("key-b")
        assert h1 != h2
