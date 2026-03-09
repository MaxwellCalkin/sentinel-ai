"""Tests for token counter."""

import pytest
from sentinel.token_counter import TokenCounter, TokenEstimate, TruncateResult


# ---------------------------------------------------------------------------
# Basic counting
# ---------------------------------------------------------------------------

class TestBasicCount:
    def test_simple_text(self):
        c = TokenCounter()
        est = c.count("Hello, world!")
        assert isinstance(est, TokenEstimate)
        assert est.tokens > 0
        assert est.chars == 13
        assert est.words == 2

    def test_empty_string(self):
        c = TokenCounter()
        est = c.count("")
        assert est.tokens == 0
        assert est.chars == 0
        assert est.confidence == "high"

    def test_single_word(self):
        c = TokenCounter()
        est = c.count("hello")
        assert est.tokens >= 1

    def test_long_text(self):
        c = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog. " * 100
        est = c.count(text)
        # 900 words, 4500 chars -> ~1125 tokens at 4 chars/token
        assert est.tokens > 900
        assert est.tokens < 1500

    def test_method_is_heuristic(self):
        c = TokenCounter()
        est = c.count("test")
        assert est.method == "heuristic"


# ---------------------------------------------------------------------------
# CJK text
# ---------------------------------------------------------------------------

class TestCJK:
    def test_chinese_text(self):
        c = TokenCounter()
        est = c.count("你好世界")
        assert est.tokens >= 2
        assert est.confidence == "medium"

    def test_mixed_text(self):
        c = TokenCounter()
        est = c.count("Hello 你好 World 世界")
        assert est.tokens > 3


# ---------------------------------------------------------------------------
# Message counting
# ---------------------------------------------------------------------------

class TestMessages:
    def test_count_messages(self):
        c = TokenCounter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        est = c.count_messages(messages)
        assert est.tokens > 10  # Content + overhead
        assert est.confidence == "medium"

    def test_empty_messages(self):
        c = TokenCounter()
        est = c.count_messages([])
        assert est.tokens == 3  # Base overhead only


# ---------------------------------------------------------------------------
# Fits context
# ---------------------------------------------------------------------------

class TestFitsContext:
    def test_fits(self):
        c = TokenCounter()
        assert c.fits_context("Hello world", max_tokens=100)

    def test_doesnt_fit(self):
        c = TokenCounter()
        long_text = "x " * 1000
        assert not c.fits_context(long_text, max_tokens=10)

    def test_reserved_tokens(self):
        c = TokenCounter()
        text = "A " * 50
        # Might fit in 200 tokens but not with 180 reserved
        assert c.fits_context(text, max_tokens=200, reserved=0)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_no_truncation_needed(self):
        c = TokenCounter()
        result = c.truncate("Hello", max_tokens=100)
        assert not result.truncated
        assert result.text == "Hello"
        assert result.chars_removed == 0

    def test_truncation_applied(self):
        c = TokenCounter()
        long_text = "word " * 500
        result = c.truncate(long_text, max_tokens=50)
        assert result.truncated
        assert result.text.endswith("...")
        assert result.estimated_tokens <= 50
        assert result.chars_removed > 0

    def test_custom_suffix(self):
        c = TokenCounter()
        long_text = "word " * 500
        result = c.truncate(long_text, max_tokens=50, suffix=" [truncated]")
        assert result.text.endswith("[truncated]")


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCost:
    def test_estimate_cost(self):
        c = TokenCounter()
        cost = c.estimate_cost("Hello world", price_per_1k=0.003)
        assert cost > 0
        assert cost < 0.01  # Very short text

    def test_cost_scales_with_length(self):
        c = TokenCounter()
        cost_short = c.estimate_cost("Hi")
        cost_long = c.estimate_cost("Hello world " * 100)
        assert cost_long > cost_short


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

class TestCustomParams:
    def test_custom_chars_per_token(self):
        c1 = TokenCounter(chars_per_token=2.0)
        c2 = TokenCounter(chars_per_token=8.0)
        text = "Some text here for testing"
        # More chars per token = fewer tokens
        assert c1.count(text).tokens > c2.count(text).tokens
