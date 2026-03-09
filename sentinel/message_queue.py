"""Safety-aware message queue for LLM requests.

Queue and prioritize LLM requests with built-in safety
filtering, priority ordering, and dead letter handling.

Usage:
    from sentinel.message_queue import MessageQueue

    queue = MessageQueue(max_size=100)
    queue.enqueue("Summarize this document", priority=1)
    msg = queue.dequeue()
"""

from __future__ import annotations

import time
import heapq
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class QueueMessage:
    """A message in the queue."""
    priority: int
    timestamp: float = field(compare=False)
    content: str = field(compare=False)
    message_id: str = field(compare=False, default="")
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)
    retries: int = field(compare=False, default=0)


@dataclass
class QueueStats:
    """Queue statistics."""
    size: int
    total_enqueued: int
    total_dequeued: int
    total_filtered: int
    total_dead_letter: int
    avg_wait_ms: float


class MessageQueue:
    """Safety-aware message queue with priority ordering."""

    def __init__(
        self,
        max_size: int = 1000,
        max_retries: int = 3,
        filter_fn: Callable[[str], bool] | None = None,
    ) -> None:
        """
        Args:
            max_size: Maximum queue size.
            max_retries: Max retries before dead letter.
            filter_fn: Optional function that returns True to accept message.
        """
        self._heap: list[QueueMessage] = []
        self._max_size = max_size
        self._max_retries = max_retries
        self._filter_fn = filter_fn
        self._dead_letter: list[QueueMessage] = []
        self._lock = threading.Lock()
        self._msg_counter = 0
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_filtered = 0
        self._wait_times: list[float] = []

    def enqueue(
        self,
        content: str,
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> QueueMessage | None:
        """Add message to queue.

        Args:
            content: Message content.
            priority: Priority (lower = higher priority).
            metadata: Optional metadata.

        Returns:
            QueueMessage if accepted, None if filtered or queue full.
        """
        # Apply filter
        if self._filter_fn and not self._filter_fn(content):
            self._total_filtered += 1
            return None

        with self._lock:
            if len(self._heap) >= self._max_size:
                return None

            self._msg_counter += 1
            msg = QueueMessage(
                priority=priority,
                timestamp=time.time(),
                content=content,
                message_id=f"msg-{self._msg_counter}",
                metadata=metadata or {},
            )
            heapq.heappush(self._heap, msg)
            self._total_enqueued += 1
            return msg

    def dequeue(self) -> QueueMessage | None:
        """Get highest priority message from queue."""
        with self._lock:
            if not self._heap:
                return None
            msg = heapq.heappop(self._heap)
            self._total_dequeued += 1
            wait_ms = (time.time() - msg.timestamp) * 1000
            self._wait_times.append(wait_ms)
            return msg

    def peek(self) -> QueueMessage | None:
        """Peek at highest priority message without removing."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def retry(self, msg: QueueMessage) -> bool:
        """Retry a failed message.

        Returns:
            True if re-queued, False if sent to dead letter.
        """
        msg.retries += 1
        if msg.retries > self._max_retries:
            self._dead_letter.append(msg)
            self._total_filtered += 1
            return False

        with self._lock:
            heapq.heappush(self._heap, msg)
            return True

    def size(self) -> int:
        """Current queue size."""
        with self._lock:
            return len(self._heap)

    def stats(self) -> QueueStats:
        """Get queue statistics."""
        avg_wait = sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0
        return QueueStats(
            size=self.size(),
            total_enqueued=self._total_enqueued,
            total_dequeued=self._total_dequeued,
            total_filtered=self._total_filtered,
            total_dead_letter=len(self._dead_letter),
            avg_wait_ms=round(avg_wait, 2),
        )

    def get_dead_letter(self) -> list[QueueMessage]:
        """Get dead letter messages."""
        return list(self._dead_letter)

    def clear(self) -> int:
        """Clear the queue. Returns count cleared."""
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            return count
