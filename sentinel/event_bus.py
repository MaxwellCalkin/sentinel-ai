"""Publish-subscribe event bus for safety events.

Decouple safety event producers from consumers. Subscribe
handlers to event types for monitoring, alerting, and logging.

Usage:
    from sentinel.event_bus import EventBus

    bus = EventBus()
    bus.subscribe("scan.blocked", lambda e: print(f"Blocked: {e.data}"))
    bus.publish("scan.blocked", {"text": "malicious input"})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    """A published event."""
    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = ""


@dataclass
class EventStats:
    """Event bus statistics."""
    total_published: int
    total_delivered: int
    subscribers: int
    event_types: int
    errors: int


class EventBus:
    """Publish-subscribe event bus for safety events."""

    def __init__(self, max_history: int = 1000) -> None:
        self._subscribers: dict[str, list[Callable[[Event], Any]]] = {}
        self._history: list[Event] = []
        self._max_history = max_history
        self._total_published = 0
        self._total_delivered = 0
        self._errors = 0

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Event], Any]) -> bool:
        """Unsubscribe a handler from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                return True
            except ValueError:
                return False
        return False

    def publish(self, event_type: str, data: dict[str, Any] | None = None, source: str = "") -> Event:
        """Publish an event to all subscribers."""
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
        )

        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        self._total_published += 1

        # Notify exact match subscribers
        handlers = list(self._subscribers.get(event_type, []))

        # Notify wildcard subscribers
        for pattern, subs in self._subscribers.items():
            if pattern.endswith("*") and event_type.startswith(pattern[:-1]):
                handlers.extend(subs)

        for handler in handlers:
            try:
                handler(event)
                self._total_delivered += 1
            except Exception:
                self._errors += 1

        return event

    def get_history(self, event_type: str | None = None, limit: int = 50) -> list[Event]:
        """Get recent event history."""
        if event_type:
            filtered = [e for e in self._history if e.event_type == event_type]
        else:
            filtered = list(self._history)
        return filtered[-limit:]

    def stats(self) -> EventStats:
        """Get event bus statistics."""
        event_types = set(e.event_type for e in self._history)
        subscriber_count = sum(len(subs) for subs in self._subscribers.values())
        return EventStats(
            total_published=self._total_published,
            total_delivered=self._total_delivered,
            subscribers=subscriber_count,
            event_types=len(event_types),
            errors=self._errors,
        )

    def clear_history(self) -> int:
        """Clear event history."""
        count = len(self._history)
        self._history.clear()
        return count

    def clear_subscribers(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
