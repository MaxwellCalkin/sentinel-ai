"""Drop-in client wrappers for Anthropic and OpenAI SDKs.

Wraps the client object so all API calls automatically get input/output
safety scanning. Unlike the function-based wrappers, these require zero
changes to your existing code beyond wrapping the client at initialization.

Usage with Anthropic:
    from anthropic import Anthropic
    from sentinel.middleware.guard import guard_anthropic

    client = guard_anthropic(Anthropic())
    # All messages.create() calls now have safety scanning
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_input}],
    )

Usage with OpenAI:
    from openai import OpenAI
    from sentinel.middleware.guard import guard_openai

    client = guard_openai(OpenAI())
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sentinel.core import SentinelGuard, ScanResult, RiskLevel


class BlockedInputError(Exception):
    """Raised when input is blocked by safety scanning."""

    def __init__(self, result: ScanResult):
        self.result = result
        categories = [f.category for f in result.findings]
        super().__init__(
            f"Input blocked (risk={result.risk.value}): {', '.join(categories)}"
        )


class BlockedOutputError(Exception):
    """Raised when output is blocked by safety scanning."""

    def __init__(self, result: ScanResult):
        self.result = result
        categories = [f.category for f in result.findings]
        super().__init__(
            f"Output blocked (risk={result.risk.value}): {', '.join(categories)}"
        )


@dataclass
class ScanEvent:
    """Record of a scan performed by the guard wrapper."""

    direction: str  # "input" or "output"
    text: str
    result: ScanResult
    blocked: bool


@dataclass
class GuardConfig:
    """Configuration for guard wrappers."""

    guard: SentinelGuard = field(default_factory=SentinelGuard.default)
    block_on_input: bool = True
    block_on_output: bool = True
    on_scan: Callable[[ScanEvent], None] | None = None
    _scan_log: list[ScanEvent] = field(default_factory=list)

    def _do_scan(self, text: str, direction: str) -> ScanResult:
        result = self.guard.scan(text)
        event = ScanEvent(direction, text, result, result.blocked)
        self._scan_log.append(event)
        if self.on_scan:
            self.on_scan(event)
        return result

    @property
    def scan_log(self) -> list[ScanEvent]:
        return list(self._scan_log)


def _extract_message_texts(messages: list[dict]) -> list[str]:
    """Extract text content from message dicts."""
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
    return texts


class _GuardedAnthropicMessages:
    def __init__(self, original: Any, config: GuardConfig):
        self._original = original
        self._config = config

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        for text in _extract_message_texts(messages):
            if text.strip():
                result = self._config._do_scan(text, "input")
                if result.blocked and self._config.block_on_input:
                    raise BlockedInputError(result)

        response = self._original.create(**kwargs)

        texts = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", ""))
        output = "\n".join(texts)
        if output.strip():
            result = self._config._do_scan(output, "output")
            if result.blocked and self._config.block_on_output:
                raise BlockedOutputError(result)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _GuardedAnthropicClient:
    def __init__(self, client: Any, config: GuardConfig):
        self._client = client
        self._messages = _GuardedAnthropicMessages(client.messages, config)
        self._config = config

    @property
    def messages(self) -> _GuardedAnthropicMessages:
        return self._messages

    @property
    def scan_log(self) -> list[ScanEvent]:
        return self._config.scan_log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _GuardedOpenAICompletions:
    def __init__(self, original: Any, config: GuardConfig):
        self._original = original
        self._config = config

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        for text in _extract_message_texts(messages):
            if text.strip():
                result = self._config._do_scan(text, "input")
                if result.blocked and self._config.block_on_input:
                    raise BlockedInputError(result)

        response = self._original.create(**kwargs)

        texts = []
        for choice in getattr(response, "choices", []):
            msg = getattr(choice, "message", None)
            if msg:
                content = getattr(msg, "content", None)
                if content:
                    texts.append(content)
        output = "\n".join(texts)
        if output.strip():
            result = self._config._do_scan(output, "output")
            if result.blocked and self._config.block_on_output:
                raise BlockedOutputError(result)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _GuardedOpenAIChat:
    def __init__(self, chat: Any, config: GuardConfig):
        self._chat = chat
        self._completions = _GuardedOpenAICompletions(chat.completions, config)

    @property
    def completions(self) -> _GuardedOpenAICompletions:
        return self._completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _GuardedOpenAIClient:
    def __init__(self, client: Any, config: GuardConfig):
        self._client = client
        self._chat = _GuardedOpenAIChat(client.chat, config)
        self._config = config

    @property
    def chat(self) -> _GuardedOpenAIChat:
        return self._chat

    @property
    def scan_log(self) -> list[ScanEvent]:
        return self._config.scan_log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def guard_anthropic(
    client: Any,
    *,
    guard: SentinelGuard | None = None,
    block_on_input: bool = True,
    block_on_output: bool = True,
    on_scan: Callable[[ScanEvent], None] | None = None,
) -> _GuardedAnthropicClient:
    """Wrap an Anthropic client with automatic safety scanning.

    Args:
        client: An Anthropic() client instance.
        guard: Custom SentinelGuard (default: all scanners).
        block_on_input: Raise BlockedInputError on unsafe input (default: True).
        block_on_output: Raise BlockedOutputError on unsafe output (default: True).
        on_scan: Callback invoked for each scan event.

    Returns:
        Wrapped client — use exactly like the original Anthropic client.
    """
    config = GuardConfig(
        guard=guard or SentinelGuard.default(),
        block_on_input=block_on_input,
        block_on_output=block_on_output,
        on_scan=on_scan,
    )
    return _GuardedAnthropicClient(client, config)


def guard_openai(
    client: Any,
    *,
    guard: SentinelGuard | None = None,
    block_on_input: bool = True,
    block_on_output: bool = True,
    on_scan: Callable[[ScanEvent], None] | None = None,
) -> _GuardedOpenAIClient:
    """Wrap an OpenAI client with automatic safety scanning.

    Args:
        client: An OpenAI() client instance.
        guard: Custom SentinelGuard (default: all scanners).
        block_on_input: Raise BlockedInputError on unsafe input (default: True).
        block_on_output: Raise BlockedOutputError on unsafe output (default: True).
        on_scan: Callback invoked for each scan event.

    Returns:
        Wrapped client — use exactly like the original OpenAI client.
    """
    config = GuardConfig(
        guard=guard or SentinelGuard.default(),
        block_on_input=block_on_input,
        block_on_output=block_on_output,
        on_scan=on_scan,
    )
    return _GuardedOpenAIClient(client, config)
