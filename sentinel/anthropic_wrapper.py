"""Drop-in safety wrapper for the Anthropic Python SDK.

Wraps ``anthropic.Anthropic`` (and ``AsyncAnthropic``) so that every
``messages.create()`` call is automatically scanned by Sentinel before
sending and after receiving.  Blocked inputs raise ``InputBlockedError``;
blocked outputs raise ``OutputBlockedError``.

Usage:
    from sentinel.anthropic_wrapper import SafeAnthropic

    client = SafeAnthropic(api_key="sk-...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # response.safety is a ScanResult with findings, risk, etc.

Streaming is also supported — each chunk is scanned in real time:
    with client.messages.stream(...) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    # stream.safety has the final ScanResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from sentinel.core import Finding, RiskLevel, ScanResult, SentinelGuard


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SafetyError(Exception):
    """Base class for Sentinel safety exceptions."""

    def __init__(self, message: str, scan_result: ScanResult):
        super().__init__(message)
        self.scan_result = scan_result


class InputBlockedError(SafetyError):
    """Raised when user input is blocked by Sentinel."""
    pass


class OutputBlockedError(SafetyError):
    """Raised when model output is blocked by Sentinel."""
    pass


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class SafeResponse:
    """Wraps an Anthropic API response with safety metadata."""
    response: Any  # anthropic.types.Message
    input_scan: ScanResult
    output_scan: ScanResult

    @property
    def safety(self) -> ScanResult:
        """Combined safety result (worst of input + output)."""
        worst_risk = max(self.input_scan.risk, self.output_scan.risk)
        all_findings = self.input_scan.findings + self.output_scan.findings
        return ScanResult(
            text=self.output_scan.text,
            findings=all_findings,
            risk=worst_risk,
            blocked=self.input_scan.blocked or self.output_scan.blocked,
            latency_ms=self.input_scan.latency_ms + self.output_scan.latency_ms,
        )

    # Proxy attribute access to the underlying response
    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)


# ---------------------------------------------------------------------------
# Messages wrapper
# ---------------------------------------------------------------------------

def _extract_text(messages: list[dict]) -> str:
    """Extract all text content from a messages list for scanning."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


def _extract_response_text(response: Any) -> str:
    """Extract text from an Anthropic API response."""
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts)


class SafeMessages:
    """Wraps ``client.messages`` with safety scanning."""

    def __init__(self, messages: Any, guard: SentinelGuard, config: dict):
        self._messages = messages
        self._guard = guard
        self._config = config

    def create(self, **kwargs: Any) -> SafeResponse:
        """Create a message with safety scanning on input and output."""
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")

        # Scan input
        input_text = _extract_text(messages)
        if system:
            input_text = system + "\n" + input_text

        input_scan = self._guard.scan(
            input_text,
            context={"direction": "input", "model": kwargs.get("model", "unknown")},
        )

        if input_scan.blocked and self._config.get("block_on_input", True):
            raise InputBlockedError(
                f"Input blocked by Sentinel (risk={input_scan.risk.value}, "
                f"findings={len(input_scan.findings)})",
                scan_result=input_scan,
            )

        # Call the real API
        response = self._messages.create(**kwargs)

        # Scan output
        output_text = _extract_response_text(response)
        output_scan = self._guard.scan(
            output_text,
            context={"direction": "output", "model": kwargs.get("model", "unknown")},
        )

        if output_scan.blocked and self._config.get("block_on_output", True):
            raise OutputBlockedError(
                f"Output blocked by Sentinel (risk={output_scan.risk.value}, "
                f"findings={len(output_scan.findings)})",
                scan_result=output_scan,
            )

        return SafeResponse(
            response=response,
            input_scan=input_scan,
            output_scan=output_scan,
        )

    def stream(self, **kwargs: Any) -> SafeStreamManager:
        """Stream a message with real-time safety scanning."""
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")

        # Scan input before streaming
        input_text = _extract_text(messages)
        if system:
            input_text = system + "\n" + input_text

        input_scan = self._guard.scan(
            input_text,
            context={"direction": "input", "model": kwargs.get("model", "unknown")},
        )

        if input_scan.blocked and self._config.get("block_on_input", True):
            raise InputBlockedError(
                f"Input blocked by Sentinel (risk={input_scan.risk.value})",
                scan_result=input_scan,
            )

        stream = self._messages.stream(**kwargs)
        return SafeStreamManager(stream, self._guard, input_scan, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class SafeAsyncMessages:
    """Wraps ``async_client.messages`` with safety scanning."""

    def __init__(self, messages: Any, guard: SentinelGuard, config: dict):
        self._messages = messages
        self._guard = guard
        self._config = config

    async def create(self, **kwargs: Any) -> SafeResponse:
        """Create a message with safety scanning on input and output."""
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")

        input_text = _extract_text(messages)
        if system:
            input_text = system + "\n" + input_text

        input_scan = await self._guard.scan_async(
            input_text,
            context={"direction": "input", "model": kwargs.get("model", "unknown")},
        )

        if input_scan.blocked and self._config.get("block_on_input", True):
            raise InputBlockedError(
                f"Input blocked by Sentinel (risk={input_scan.risk.value})",
                scan_result=input_scan,
            )

        response = await self._messages.create(**kwargs)

        output_text = _extract_response_text(response)
        output_scan = await self._guard.scan_async(
            output_text,
            context={"direction": "output", "model": kwargs.get("model", "unknown")},
        )

        if output_scan.blocked and self._config.get("block_on_output", True):
            raise OutputBlockedError(
                f"Output blocked by Sentinel (risk={output_scan.risk.value})",
                scan_result=output_scan,
            )

        return SafeResponse(
            response=response,
            input_scan=input_scan,
            output_scan=output_scan,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


# ---------------------------------------------------------------------------
# Streaming wrapper
# ---------------------------------------------------------------------------

class SafeStreamManager:
    """Context manager for streaming with real-time safety scanning."""

    def __init__(
        self,
        stream: Any,
        guard: SentinelGuard,
        input_scan: ScanResult,
        config: dict,
    ):
        self._stream = stream
        self._guard = guard
        self._input_scan = input_scan
        self._config = config
        self._accumulated_text = ""
        self._output_scan: ScanResult | None = None
        self._blocked = False
        self._scan_interval = config.get("stream_scan_interval", 200)

    def __enter__(self) -> SafeStreamManager:
        self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stream.__exit__(*args)

    @property
    def text_stream(self) -> Iterator[str]:
        """Yield text chunks with periodic safety scanning."""
        chars_since_scan = 0
        for text in self._stream.text_stream:
            self._accumulated_text += text
            chars_since_scan += len(text)

            # Periodic scan during streaming
            if chars_since_scan >= self._scan_interval:
                scan = self._guard.scan(
                    self._accumulated_text,
                    context={"direction": "output", "streaming": True},
                )
                chars_since_scan = 0

                if scan.blocked and self._config.get("block_on_output", True):
                    self._output_scan = scan
                    self._blocked = True
                    return

            yield text

        # Final scan after stream completes
        self._output_scan = self._guard.scan(
            self._accumulated_text,
            context={"direction": "output"},
        )

    @property
    def safety(self) -> ScanResult:
        """Final safety result after streaming completes."""
        if self._output_scan is None:
            self._output_scan = self._guard.scan(self._accumulated_text)
        worst_risk = max(self._input_scan.risk, self._output_scan.risk)
        all_findings = self._input_scan.findings + self._output_scan.findings
        return ScanResult(
            text=self._accumulated_text,
            findings=all_findings,
            risk=worst_risk,
            blocked=self._blocked or self._input_scan.blocked,
            latency_ms=self._input_scan.latency_ms + self._output_scan.latency_ms,
        )

    @property
    def blocked(self) -> bool:
        return self._blocked

    def get_final_message(self) -> Any:
        """Proxy to underlying stream's get_final_message."""
        return self._stream.get_final_message()


# ---------------------------------------------------------------------------
# Main client wrappers
# ---------------------------------------------------------------------------

class SafeAnthropic:
    """Drop-in replacement for ``anthropic.Anthropic`` with safety scanning.

    All ``messages.create()`` and ``messages.stream()`` calls are scanned.
    Other methods (completions, etc.) pass through unchanged.

    Args:
        guard: SentinelGuard instance (defaults to SentinelGuard.default())
        block_on_input: Raise InputBlockedError on unsafe input (default True)
        block_on_output: Raise OutputBlockedError on unsafe output (default True)
        stream_scan_interval: Characters between scans during streaming (default 200)
        **kwargs: Forwarded to ``anthropic.Anthropic()``.
    """

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        block_on_input: bool = True,
        block_on_output: bool = True,
        stream_scan_interval: int = 200,
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required. Install it with: "
                "pip install anthropic"
            )

        self._client = anthropic.Anthropic(**kwargs)
        self._guard = guard or SentinelGuard.default()
        self._config = {
            "block_on_input": block_on_input,
            "block_on_output": block_on_output,
            "stream_scan_interval": stream_scan_interval,
        }

    @property
    def messages(self) -> SafeMessages:
        return SafeMessages(self._client.messages, self._guard, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class SafeAsyncAnthropic:
    """Drop-in replacement for ``anthropic.AsyncAnthropic`` with safety scanning.

    Args:
        guard: SentinelGuard instance (defaults to SentinelGuard.default())
        block_on_input: Raise InputBlockedError on unsafe input (default True)
        block_on_output: Raise OutputBlockedError on unsafe output (default True)
        **kwargs: Forwarded to ``anthropic.AsyncAnthropic()``.
    """

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        block_on_input: bool = True,
        block_on_output: bool = True,
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required. Install it with: "
                "pip install anthropic"
            )

        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._guard = guard or SentinelGuard.default()
        self._config = {
            "block_on_input": block_on_input,
            "block_on_output": block_on_output,
        }

    @property
    def messages(self) -> SafeAsyncMessages:
        return SafeAsyncMessages(self._client.messages, self._guard, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
