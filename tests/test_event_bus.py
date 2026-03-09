"""Tests for event bus."""

import pytest
from sentinel.event_bus import EventBus, Event, EventStats


class TestPublishSubscribe:
    def test_basic_subscribe_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish("test", {"key": "value"})
        assert len(received) == 1
        assert received[0].data["key"] == "value"

    def test_multiple_subscribers(self):
        bus = EventBus()
        count = [0]
        bus.subscribe("test", lambda e: count.__setitem__(0, count[0] + 1))
        bus.subscribe("test", lambda e: count.__setitem__(0, count[0] + 1))
        bus.publish("test")
        assert count[0] == 2

    def test_no_subscribers(self):
        bus = EventBus()
        event = bus.publish("unsubscribed", {"data": "test"})
        assert event.event_type == "unsubscribed"

    def test_different_event_types(self):
        bus = EventBus()
        a_count = [0]
        b_count = [0]
        bus.subscribe("type_a", lambda e: a_count.__setitem__(0, a_count[0] + 1))
        bus.subscribe("type_b", lambda e: b_count.__setitem__(0, b_count[0] + 1))
        bus.publish("type_a")
        assert a_count[0] == 1
        assert b_count[0] == 0


class TestWildcard:
    def test_wildcard_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("scan.*", lambda e: received.append(e))
        bus.publish("scan.blocked")
        bus.publish("scan.allowed")
        bus.publish("other.event")
        assert len(received) == 2


class TestUnsubscribe:
    def test_unsubscribe(self):
        bus = EventBus()
        handler = lambda e: None
        bus.subscribe("test", handler)
        assert bus.unsubscribe("test", handler)

    def test_unsubscribe_missing(self):
        bus = EventBus()
        assert not bus.unsubscribe("test", lambda e: None)


class TestHistory:
    def test_history_recorded(self):
        bus = EventBus()
        bus.publish("a")
        bus.publish("b")
        history = bus.get_history()
        assert len(history) == 2

    def test_history_filtered(self):
        bus = EventBus()
        bus.publish("a")
        bus.publish("b")
        bus.publish("a")
        history = bus.get_history(event_type="a")
        assert len(history) == 2

    def test_history_limit(self):
        bus = EventBus()
        for i in range(20):
            bus.publish("test")
        history = bus.get_history(limit=5)
        assert len(history) == 5

    def test_clear_history(self):
        bus = EventBus()
        bus.publish("test")
        cleared = bus.clear_history()
        assert cleared == 1
        assert len(bus.get_history()) == 0


class TestErrorHandling:
    def test_handler_error_counted(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: 1/0)
        bus.publish("test")
        assert bus.stats().errors == 1

    def test_error_doesnt_stop_others(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: 1/0)
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish("test")
        assert len(received) == 1


class TestStats:
    def test_stats(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.publish("test")
        bus.publish("other")
        stats = bus.stats()
        assert isinstance(stats, EventStats)
        assert stats.total_published == 2
        assert stats.subscribers >= 1
        assert stats.event_types == 2


class TestStructure:
    def test_event_structure(self):
        bus = EventBus()
        event = bus.publish("test", {"key": "val"}, source="scanner")
        assert isinstance(event, Event)
        assert event.event_type == "test"
        assert event.source == "scanner"
        assert event.timestamp > 0
