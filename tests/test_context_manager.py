"""Tests for the context window manager."""

import pytest
from sentinel.context_manager import ContextWindowManager, Message, TokenUsage


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestBasicOperations:
    def test_add_message(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "Hello")
        assert mgr.message_count == 1

    def test_add_multiple(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "Hello")
        mgr.add_message("assistant", "Hi there")
        assert mgr.message_count == 2

    def test_set_system_prompt(self):
        mgr = ContextWindowManager()
        mgr.set_system_prompt("You are a helpful assistant")
        assert mgr.usage.system_tokens > 0

    def test_clear(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "Hello")
        mgr.set_system_prompt("System prompt")
        mgr.clear()
        assert mgr.message_count == 0
        # System prompt should persist
        assert mgr.usage.system_tokens > 0

    def test_get_messages(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "Hello")
        msgs = mgr.get_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "Hello"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_token_estimate(self):
        mgr = ContextWindowManager()
        msg = mgr.add_message("user", "Hello world")  # 11 chars ~ 2-3 tokens
        assert msg.tokens > 0

    def test_long_message_tokens(self):
        mgr = ContextWindowManager()
        msg = mgr.add_message("user", "x" * 1000)
        assert msg.tokens == 250  # 1000 / 4

    def test_empty_message_min_token(self):
        mgr = ContextWindowManager()
        msg = mgr.add_message("user", "")
        assert msg.tokens >= 1  # Minimum 1 token


# ---------------------------------------------------------------------------
# FIFO strategy
# ---------------------------------------------------------------------------

class TestFifoStrategy:
    def test_fifo_removes_oldest(self):
        mgr = ContextWindowManager(max_tokens=100, reserve_output=0, strategy="fifo")
        # Each message ~25 tokens (100 chars / 4)
        mgr.add_message("user", "A" * 100)
        mgr.add_message("assistant", "B" * 100)
        mgr.add_message("user", "C" * 100)
        mgr.add_message("assistant", "D" * 100)
        mgr.add_message("user", "E" * 100)

        msgs = mgr.get_messages()
        # Should keep messages that fit within 100 tokens
        assert len(msgs) < 5
        # Most recent should be preserved
        assert msgs[-1].content == "E" * 100

    def test_fifo_keeps_pinned(self):
        mgr = ContextWindowManager(max_tokens=60, reserve_output=0, strategy="fifo")
        mgr.add_message("user", "A" * 100, pinned=True)
        mgr.add_message("assistant", "B" * 100)
        mgr.add_message("user", "C" * 100)

        msgs = mgr.get_messages()
        # Pinned message should always be present
        assert any(m.content == "A" * 100 for m in msgs)


# ---------------------------------------------------------------------------
# Sliding window strategy
# ---------------------------------------------------------------------------

class TestSlidingStrategy:
    def test_sliding_keeps_recent(self):
        mgr = ContextWindowManager(max_tokens=60, reserve_output=0, strategy="sliding")
        mgr.add_message("user", "A" * 100)      # 25 tokens
        mgr.add_message("assistant", "B" * 100)  # 25 tokens
        mgr.add_message("user", "C" * 100)       # 25 tokens

        msgs = mgr.get_messages()
        # Should keep 2 most recent (50 tokens fits in 60)
        assert len(msgs) == 2
        assert msgs[0].content == "B" * 100
        assert msgs[1].content == "C" * 100

    def test_sliding_includes_pinned(self):
        mgr = ContextWindowManager(max_tokens=60, reserve_output=0, strategy="sliding")
        mgr.add_message("user", "A" * 100, pinned=True)
        mgr.add_message("assistant", "B" * 100)
        mgr.add_message("user", "C" * 100)

        msgs = mgr.get_messages()
        # Pinned should be included even if over budget
        assert any(m.content == "A" * 100 for m in msgs)


# ---------------------------------------------------------------------------
# Smart strategy
# ---------------------------------------------------------------------------

class TestSmartStrategy:
    def test_smart_keeps_pinned_and_recent(self):
        mgr = ContextWindowManager(max_tokens=80, reserve_output=0, strategy="smart")
        mgr.add_message("user", "A" * 100, pinned=True)  # 25 tokens
        mgr.add_message("assistant", "B" * 100)           # 25 tokens
        mgr.add_message("user", "C" * 100)                # 25 tokens
        mgr.add_message("assistant", "D" * 100)           # 25 tokens

        msgs = mgr.get_messages()
        # Should include pinned A + as many recent as fit (80-25=55 tokens left, so 2 more)
        assert any(m.content == "A" * 100 for m in msgs)
        assert msgs[-1].content == "D" * 100

    def test_smart_maintains_order(self):
        mgr = ContextWindowManager(max_tokens=200, reserve_output=0, strategy="smart")
        mgr.add_message("user", "First")
        mgr.add_message("assistant", "Second")
        mgr.add_message("user", "Third")

        msgs = mgr.get_messages()
        assert msgs[0].content == "First"
        assert msgs[1].content == "Second"
        assert msgs[2].content == "Third"

    def test_smart_default(self):
        mgr = ContextWindowManager()  # default is smart
        mgr.add_message("user", "Hello")
        msgs = mgr.get_messages()
        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# System prompt handling
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_reduces_budget(self):
        mgr = ContextWindowManager(max_tokens=100, reserve_output=0)
        mgr.set_system_prompt("x" * 200)  # 50 tokens
        # Budget is 100 - 50 = 50 tokens for messages
        mgr.add_message("user", "A" * 100)  # 25 tokens — fits
        mgr.add_message("user", "B" * 100)  # 25 tokens — fits
        mgr.add_message("user", "C" * 100)  # 25 tokens — one too many

        msgs = mgr.get_messages()
        assert len(msgs) == 2  # Only 2 fit after system prompt

    def test_formatted_includes_system(self):
        mgr = ContextWindowManager()
        mgr.set_system_prompt("You are helpful")
        mgr.add_message("user", "Hello")

        formatted = mgr.get_formatted_messages()
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == "You are helpful"
        assert formatted[1]["role"] == "user"

    def test_update_system_prompt(self):
        mgr = ContextWindowManager()
        mgr.set_system_prompt("Version 1")
        mgr.set_system_prompt("Version 2")
        formatted = mgr.get_formatted_messages()
        assert formatted[0]["content"] == "Version 2"


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_usage_basic(self):
        mgr = ContextWindowManager(max_tokens=1000, reserve_output=100)
        mgr.add_message("user", "Hello world test")  # ~4 tokens

        usage = mgr.usage
        assert usage.total_tokens > 0
        assert usage.available_tokens > 0
        assert usage.output_reserve == 100
        assert usage.messages_count == 1
        assert 0.0 < usage.utilization < 1.0

    def test_usage_over_budget(self):
        mgr = ContextWindowManager(max_tokens=10, reserve_output=0)
        for _ in range(100):
            mgr.add_message("user", "x" * 100)

        usage = mgr.usage
        # Smart strategy should keep messages within budget
        assert usage.total_tokens <= 10

    def test_no_messages(self):
        mgr = ContextWindowManager()
        usage = mgr.usage
        assert usage.total_tokens == 0
        assert usage.messages_count == 0
        assert usage.utilization == 0.0


# ---------------------------------------------------------------------------
# Output reserve
# ---------------------------------------------------------------------------

class TestOutputReserve:
    def test_reserve_reduces_budget(self):
        mgr = ContextWindowManager(max_tokens=100, reserve_output=50)
        # Only 50 tokens available for messages
        mgr.add_message("user", "A" * 100)  # 25 tokens
        mgr.add_message("user", "B" * 100)  # 25 tokens
        mgr.add_message("user", "C" * 100)  # 25 tokens — over budget

        msgs = mgr.get_messages()
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# Formatted messages
# ---------------------------------------------------------------------------

class TestFormattedMessages:
    def test_format(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "Hello")
        mgr.add_message("assistant", "World")

        formatted = mgr.get_formatted_messages()
        assert formatted[0] == {"role": "user", "content": "Hello"}
        assert formatted[1] == {"role": "assistant", "content": "World"}

    def test_no_system(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "Hello")
        formatted = mgr.get_formatted_messages()
        assert len(formatted) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_conversation(self):
        mgr = ContextWindowManager()
        msgs = mgr.get_messages()
        assert len(msgs) == 0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ContextWindowManager(strategy="invalid")

    def test_pinned_message(self):
        mgr = ContextWindowManager()
        msg = mgr.add_message("user", "Important", pinned=True)
        assert msg.pinned

    def test_many_messages(self):
        mgr = ContextWindowManager(max_tokens=1000, reserve_output=100)
        for i in range(1000):
            mgr.add_message("user", f"Message {i}")
        msgs = mgr.get_messages()
        total = sum(m.tokens for m in msgs)
        assert total <= 900  # max - reserve

    def test_total_messages(self):
        mgr = ContextWindowManager()
        mgr.add_message("user", "A")
        mgr.add_message("user", "B")
        assert mgr.total_messages == 2
