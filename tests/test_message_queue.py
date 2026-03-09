"""Tests for message queue."""

import pytest
from sentinel.message_queue import MessageQueue, QueueMessage, QueueStats


class TestEnqueue:
    def test_basic_enqueue(self):
        q = MessageQueue()
        msg = q.enqueue("Hello world")
        assert msg is not None
        assert msg.content == "Hello world"
        assert q.size() == 1

    def test_with_priority(self):
        q = MessageQueue()
        q.enqueue("Low priority", priority=10)
        q.enqueue("High priority", priority=1)
        msg = q.dequeue()
        assert msg.content == "High priority"

    def test_with_metadata(self):
        q = MessageQueue()
        msg = q.enqueue("Test", metadata={"user": "admin"})
        assert msg.metadata["user"] == "admin"

    def test_queue_full(self):
        q = MessageQueue(max_size=2)
        q.enqueue("A")
        q.enqueue("B")
        result = q.enqueue("C")
        assert result is None
        assert q.size() == 2


class TestDequeue:
    def test_basic_dequeue(self):
        q = MessageQueue()
        q.enqueue("Test")
        msg = q.dequeue()
        assert msg.content == "Test"
        assert q.size() == 0

    def test_empty_dequeue(self):
        q = MessageQueue()
        assert q.dequeue() is None

    def test_priority_ordering(self):
        q = MessageQueue()
        q.enqueue("C", priority=3)
        q.enqueue("A", priority=1)
        q.enqueue("B", priority=2)
        assert q.dequeue().content == "A"
        assert q.dequeue().content == "B"
        assert q.dequeue().content == "C"


class TestFilter:
    def test_filter_rejects(self):
        q = MessageQueue(filter_fn=lambda x: "safe" in x)
        assert q.enqueue("safe content") is not None
        assert q.enqueue("blocked content") is None
        assert q.size() == 1

    def test_filter_stats(self):
        q = MessageQueue(filter_fn=lambda x: len(x) < 20)
        q.enqueue("short")
        q.enqueue("this is a very long message that should be filtered out")
        stats = q.stats()
        assert stats.total_filtered == 1


class TestRetry:
    def test_retry_success(self):
        q = MessageQueue(max_retries=3)
        msg = q.enqueue("Test")
        q.dequeue()
        assert q.retry(msg)
        assert q.size() == 1

    def test_retry_to_dead_letter(self):
        q = MessageQueue(max_retries=2)
        msg = q.enqueue("Test")
        q.dequeue()
        msg.retries = 2
        assert not q.retry(msg)
        assert len(q.get_dead_letter()) == 1


class TestPeek:
    def test_peek(self):
        q = MessageQueue()
        q.enqueue("Test")
        msg = q.peek()
        assert msg.content == "Test"
        assert q.size() == 1  # not removed

    def test_peek_empty(self):
        q = MessageQueue()
        assert q.peek() is None


class TestStats:
    def test_stats_tracking(self):
        q = MessageQueue()
        q.enqueue("A")
        q.enqueue("B")
        q.dequeue()
        stats = q.stats()
        assert stats.total_enqueued == 2
        assert stats.total_dequeued == 1
        assert stats.size == 1


class TestClear:
    def test_clear(self):
        q = MessageQueue()
        q.enqueue("A")
        q.enqueue("B")
        cleared = q.clear()
        assert cleared == 2
        assert q.size() == 0


class TestStructure:
    def test_message_structure(self):
        q = MessageQueue()
        msg = q.enqueue("Test", priority=3)
        assert isinstance(msg, QueueMessage)
        assert msg.message_id.startswith("msg-")
        assert msg.retries == 0
        assert msg.timestamp > 0

    def test_stats_structure(self):
        q = MessageQueue()
        stats = q.stats()
        assert isinstance(stats, QueueStats)
        assert stats.avg_wait_ms >= 0
