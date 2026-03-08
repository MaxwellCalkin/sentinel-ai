"""Context window management for LLM conversations.

Tracks token usage across multi-turn conversations, automatically
summarizes or truncates older messages to stay within model context
limits, and provides visibility into token allocation.

Usage:
    from sentinel.context_manager import ContextWindowManager

    mgr = ContextWindowManager(max_tokens=100000, reserve_output=4096)
    mgr.add_message("user", "Hello, how are you?")
    mgr.add_message("assistant", "I'm doing great!")

    messages = mgr.get_messages()  # Returns messages within budget
    print(mgr.usage)               # Token usage breakdown
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Message:
    """A conversation message."""
    role: str            # user, assistant, system
    content: str
    tokens: int = 0      # Estimated token count
    pinned: bool = False  # Pinned messages are never removed
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage breakdown."""
    total_tokens: int
    system_tokens: int
    message_tokens: int
    available_tokens: int
    output_reserve: int
    messages_count: int
    utilization: float  # 0.0-1.0

    @property
    def over_budget(self) -> bool:
        return self.available_tokens < 0


class ContextWindowManager:
    """Manage conversation context within token limits.

    Provides automatic context management strategies:
    - FIFO: Remove oldest messages first
    - Sliding window: Keep N most recent messages
    - Smart: Keep system prompt + pinned + recent messages
    """

    # Rough token estimation: ~4 chars per token for English
    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        max_tokens: int = 100000,
        reserve_output: int = 4096,
        strategy: str = "smart",
    ):
        """
        Args:
            max_tokens: Maximum context window size in tokens.
            reserve_output: Tokens to reserve for model output.
            strategy: Context management strategy (fifo, sliding, smart).
        """
        if strategy not in ("fifo", "sliding", "smart"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self._max_tokens = max_tokens
        self._reserve_output = reserve_output
        self._strategy = strategy
        self._messages: list[Message] = []
        self._system_prompt: Message | None = None

    def set_system_prompt(self, content: str) -> None:
        """Set or update the system prompt."""
        self._system_prompt = Message(
            role="system",
            content=content,
            tokens=self._estimate_tokens(content),
            pinned=True,
        )

    def add_message(self, role: str, content: str, pinned: bool = False) -> Message:
        """Add a message to the conversation."""
        msg = Message(
            role=role,
            content=content,
            tokens=self._estimate_tokens(content),
            pinned=pinned,
        )
        self._messages.append(msg)
        return msg

    def get_messages(self) -> list[Message]:
        """Get messages that fit within the token budget."""
        budget = self._max_tokens - self._reserve_output

        if self._system_prompt:
            budget -= self._system_prompt.tokens

        if self._strategy == "fifo":
            return self._apply_fifo(budget)
        elif self._strategy == "sliding":
            return self._apply_sliding(budget)
        else:  # smart
            return self._apply_smart(budget)

    def get_formatted_messages(self) -> list[dict[str, str]]:
        """Get messages as dicts suitable for API calls."""
        result = []
        if self._system_prompt:
            result.append({
                "role": self._system_prompt.role,
                "content": self._system_prompt.content,
            })
        for msg in self.get_messages():
            result.append({"role": msg.role, "content": msg.content})
        return result

    @property
    def usage(self) -> TokenUsage:
        """Get current token usage breakdown."""
        system_tokens = self._system_prompt.tokens if self._system_prompt else 0
        messages = self.get_messages()
        message_tokens = sum(m.tokens for m in messages)
        total = system_tokens + message_tokens
        available = self._max_tokens - self._reserve_output - total

        return TokenUsage(
            total_tokens=total,
            system_tokens=system_tokens,
            message_tokens=message_tokens,
            available_tokens=available,
            output_reserve=self._reserve_output,
            messages_count=len(messages),
            utilization=total / (self._max_tokens - self._reserve_output)
            if self._max_tokens > self._reserve_output
            else 1.0,
        )

    @property
    def message_count(self) -> int:
        """Total messages stored (before truncation)."""
        return len(self._messages)

    @property
    def total_messages(self) -> int:
        """Alias for message_count."""
        return len(self._messages)

    def clear(self) -> None:
        """Clear all messages (keeps system prompt)."""
        self._messages.clear()

    def _apply_fifo(self, budget: int) -> list[Message]:
        """Remove oldest non-pinned messages first."""
        # Start with all messages
        result = list(self._messages)
        total = sum(m.tokens for m in result)

        while total > budget and result:
            # Find first non-pinned message
            for i, msg in enumerate(result):
                if not msg.pinned:
                    total -= msg.tokens
                    result.pop(i)
                    break
            else:
                break  # All remaining are pinned

        return result

    def _apply_sliding(self, budget: int) -> list[Message]:
        """Keep most recent messages that fit."""
        result: list[Message] = []
        total = 0

        # Add messages from newest to oldest
        for msg in reversed(self._messages):
            if total + msg.tokens <= budget:
                result.insert(0, msg)
                total += msg.tokens
            elif msg.pinned:
                # Always include pinned messages
                result.insert(0, msg)
                total += msg.tokens

        return result

    def _apply_smart(self, budget: int) -> list[Message]:
        """Keep pinned messages + as many recent messages as fit."""
        pinned = [m for m in self._messages if m.pinned]
        unpinned = [m for m in self._messages if not m.pinned]

        pinned_tokens = sum(m.tokens for m in pinned)
        remaining_budget = budget - pinned_tokens

        # Add recent unpinned messages that fit
        recent: list[Message] = []
        for msg in reversed(unpinned):
            if remaining_budget >= msg.tokens:
                recent.insert(0, msg)
                remaining_budget -= msg.tokens
            else:
                break  # Budget exhausted

        # Merge pinned and recent, maintaining original order
        all_kept = set(id(m) for m in pinned) | set(id(m) for m in recent)
        return [m for m in self._messages if id(m) in all_kept]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        # Rough heuristic: ~4 chars per token for English
        return max(1, len(text) // self._CHARS_PER_TOKEN)
