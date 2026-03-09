"""Safety-aware conversation memory management.

Manage conversation history with automatic summarization,
context window enforcement, and sensitive data filtering
before storage.

Usage:
    from sentinel.conversation_memory import ConversationMemory

    memory = ConversationMemory(max_messages=50)
    memory.add("user", "What is my account balance?")
    memory.add("assistant", "Your balance is $5,000.")
    context = memory.get_context(max_tokens=1000)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryMessage:
    """A message in conversation memory."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0
    filtered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_messages: int
    total_tokens: int
    oldest_message_age: float
    filtered_count: int
    summaries_created: int


# Patterns for sensitive data that should be filtered from memory
_SENSITIVE_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', "[SSN_REDACTED]"),
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', "[CARD_REDACTED]"),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL_REDACTED]"),
    (r'(?i)\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36})\b', "[KEY_REDACTED]"),
]


class ConversationMemory:
    """Safety-aware conversation memory."""

    def __init__(
        self,
        max_messages: int = 100,
        max_tokens: int = 50000,
        filter_sensitive: bool = True,
        chars_per_token: float = 4.0,
    ) -> None:
        self._messages: list[MemoryMessage] = []
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._filter_sensitive = filter_sensitive
        self._chars_per_token = chars_per_token
        self._summaries_created = 0
        self._filtered_count = 0

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> MemoryMessage:
        """Add a message to memory."""
        filtered = False
        stored_content = content

        if self._filter_sensitive:
            stored_content, was_filtered = self._filter(content)
            if was_filtered:
                filtered = True
                self._filtered_count += 1

        token_est = int(len(stored_content) / self._chars_per_token)

        msg = MemoryMessage(
            role=role, content=stored_content,
            token_estimate=token_est, filtered=filtered,
            metadata=metadata or {},
        )
        self._messages.append(msg)

        # Enforce message limit
        while len(self._messages) > self._max_messages:
            self._messages.pop(0)

        # Enforce token limit
        while self._total_tokens() > self._max_tokens and len(self._messages) > 1:
            self._messages.pop(0)

        return msg

    def get_context(self, max_tokens: int | None = None, max_messages: int | None = None) -> list[dict[str, str]]:
        """Get conversation context within limits."""
        messages = list(self._messages)

        if max_messages and len(messages) > max_messages:
            messages = messages[-max_messages:]

        if max_tokens:
            result = []
            total = 0
            for msg in reversed(messages):
                if total + msg.token_estimate > max_tokens:
                    break
                result.insert(0, msg)
                total += msg.token_estimate
            messages = result

        return [{"role": m.role, "content": m.content} for m in messages]

    def search(self, query: str, limit: int = 5) -> list[MemoryMessage]:
        """Search memory for messages containing query."""
        query_lower = query.lower()
        matches = []
        for msg in reversed(self._messages):
            if query_lower in msg.content.lower():
                matches.append(msg)
                if len(matches) >= limit:
                    break
        return matches

    def summarize(self, n_messages: int | None = None) -> str:
        """Create a simple summary of recent conversation."""
        messages = self._messages[-n_messages:] if n_messages else self._messages
        if not messages:
            return ""

        topics = set()
        for msg in messages:
            words = msg.content.lower().split()
            # Extract potential topic words (longer, non-common words)
            for w in words:
                clean = re.sub(r'[^a-z]', '', w)
                if len(clean) > 5:
                    topics.add(clean)

        self._summaries_created += 1
        topic_str = ", ".join(sorted(list(topics)[:10]))
        return f"Conversation ({len(messages)} messages) covering: {topic_str}" if topic_str else f"Conversation ({len(messages)} messages)"

    def stats(self) -> MemoryStats:
        """Get memory statistics."""
        now = time.time()
        oldest_age = (now - self._messages[0].timestamp) if self._messages else 0.0
        return MemoryStats(
            total_messages=len(self._messages),
            total_tokens=self._total_tokens(),
            oldest_message_age=round(oldest_age, 2),
            filtered_count=self._filtered_count,
            summaries_created=self._summaries_created,
        )

    def clear(self) -> int:
        count = len(self._messages)
        self._messages.clear()
        return count

    def _total_tokens(self) -> int:
        return sum(m.token_estimate for m in self._messages)

    def _filter(self, text: str) -> tuple[str, bool]:
        """Filter sensitive data from text."""
        result = text
        filtered = False
        for pattern, replacement in _SENSITIVE_PATTERNS:
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result)
                filtered = True
        return result, filtered
