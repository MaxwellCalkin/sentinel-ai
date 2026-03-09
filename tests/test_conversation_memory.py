"""Tests for conversation memory."""

import pytest
from sentinel.conversation_memory import ConversationMemory, MemoryMessage, MemoryStats


class TestAddMessage:
    def test_add_message(self):
        m = ConversationMemory()
        msg = m.add("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_limit(self):
        m = ConversationMemory(max_messages=3)
        for i in range(5):
            m.add("user", f"Message {i}")
        context = m.get_context()
        assert len(context) == 3

    def test_token_estimate(self):
        m = ConversationMemory()
        msg = m.add("user", "a" * 100)
        assert msg.token_estimate == 25  # 100 / 4.0

    def test_metadata(self):
        m = ConversationMemory()
        msg = m.add("user", "Hello", metadata={"source": "api"})
        assert msg.metadata["source"] == "api"


class TestSensitiveFilter:
    def test_filter_ssn(self):
        m = ConversationMemory()
        msg = m.add("user", "My SSN is 123-45-6789")
        assert "123-45-6789" not in msg.content
        assert "[SSN_REDACTED]" in msg.content
        assert msg.filtered

    def test_filter_email(self):
        m = ConversationMemory()
        msg = m.add("user", "Contact me at user@example.com")
        assert "user@example.com" not in msg.content

    def test_filter_disabled(self):
        m = ConversationMemory(filter_sensitive=False)
        msg = m.add("user", "My SSN is 123-45-6789")
        assert "123-45-6789" in msg.content

    def test_no_false_positive(self):
        m = ConversationMemory()
        msg = m.add("user", "The temperature is 72 degrees")
        assert not msg.filtered


class TestGetContext:
    def test_basic_context(self):
        m = ConversationMemory()
        m.add("user", "Hello")
        m.add("assistant", "Hi there")
        ctx = m.get_context()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"

    def test_max_messages(self):
        m = ConversationMemory()
        for i in range(10):
            m.add("user", f"Message {i}")
        ctx = m.get_context(max_messages=3)
        assert len(ctx) == 3

    def test_max_tokens(self):
        m = ConversationMemory()
        m.add("user", "short")
        m.add("user", "a" * 1000)  # ~250 tokens
        ctx = m.get_context(max_tokens=50)
        assert len(ctx) <= 2


class TestSearch:
    def test_search_found(self):
        m = ConversationMemory()
        m.add("user", "Tell me about Python programming")
        m.add("user", "What about JavaScript?")
        m.add("user", "Back to Python please")
        results = m.search("python")
        assert len(results) == 2

    def test_search_not_found(self):
        m = ConversationMemory()
        m.add("user", "Hello world")
        results = m.search("nonexistent")
        assert len(results) == 0

    def test_search_limit(self):
        m = ConversationMemory()
        for i in range(10):
            m.add("user", f"Python message {i}")
        results = m.search("python", limit=3)
        assert len(results) == 3


class TestSummarize:
    def test_summarize(self):
        m = ConversationMemory()
        m.add("user", "Tell me about machine learning algorithms")
        m.add("assistant", "Machine learning encompasses various algorithms")
        summary = m.summarize()
        assert "2 messages" in summary

    def test_summarize_empty(self):
        m = ConversationMemory()
        assert m.summarize() == ""

    def test_summarize_n_messages(self):
        m = ConversationMemory()
        for i in range(10):
            m.add("user", f"Message about programming {i}")
        summary = m.summarize(n_messages=3)
        assert "3 messages" in summary


class TestStats:
    def test_stats(self):
        m = ConversationMemory()
        m.add("user", "Hello")
        m.add("assistant", "Hi")
        stats = m.stats()
        assert isinstance(stats, MemoryStats)
        assert stats.total_messages == 2
        assert stats.total_tokens > 0

    def test_filtered_count(self):
        m = ConversationMemory()
        m.add("user", "My SSN is 123-45-6789")
        m.add("user", "Normal message")
        stats = m.stats()
        assert stats.filtered_count == 1


class TestClear:
    def test_clear(self):
        m = ConversationMemory()
        m.add("user", "Hello")
        m.add("user", "World")
        cleared = m.clear()
        assert cleared == 2
        assert len(m.get_context()) == 0
