"""Drop-in safety wrapper for the OpenAI Python SDK.

Wraps ``openai.OpenAI`` (and ``AsyncOpenAI``) so that every
``chat.completions.create()`` call is automatically scanned by Sentinel
before sending and after receiving.

Usage:
    from sentinel.openai_wrapper import SafeOpenAI

    client = SafeOpenAI(api_key="sk-...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # response.safety has scan results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from sentinel.core import Finding, RiskLevel, ScanResult, SentinelGuard
from sentinel.prompt_leak import PromptLeakDetector


# ---------------------------------------------------------------------------
# Exceptions (reuse from anthropic_wrapper for consistency)
# ---------------------------------------------------------------------------

from sentinel.anthropic_wrapper import (
    SafetyError,
    InputBlockedError,
    OutputBlockedError,
)


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class SafeChatResponse:
    """Wraps an OpenAI chat completion response with safety metadata."""
    response: Any  # openai.types.chat.ChatCompletion
    input_scan: ScanResult
    output_scan: ScanResult

    @property
    def safety(self) -> ScanResult:
        worst_risk = max(self.input_scan.risk, self.output_scan.risk)
        all_findings = self.input_scan.findings + self.output_scan.findings
        return ScanResult(
            text=self.output_scan.text,
            findings=all_findings,
            risk=worst_risk,
            blocked=self.input_scan.blocked or self.output_scan.blocked,
            latency_ms=self.input_scan.latency_ms + self.output_scan.latency_ms,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_chat_text(messages: list[dict]) -> str:
    """Extract all text from a chat messages list."""
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


def _extract_completion_text(response: Any) -> str:
    """Extract text from an OpenAI chat completion response."""
    try:
        choices = getattr(response, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            if message:
                return getattr(message, "content", "") or ""
    except (IndexError, AttributeError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Chat completions wrapper
# ---------------------------------------------------------------------------

class SafeChatCompletions:
    """Wraps ``client.chat.completions`` with safety scanning."""

    def __init__(self, completions: Any, guard: SentinelGuard, config: dict):
        self._completions = completions
        self._guard = guard
        self._config = config

    def create(self, **kwargs: Any) -> SafeChatResponse:
        messages = kwargs.get("messages", [])

        # Extract system message for prompt leak detection
        system_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_text = content

        # Scan input
        input_text = _extract_chat_text(messages)
        input_scan = self._guard.scan(
            input_text,
            context={"direction": "input", "model": kwargs.get("model", "unknown")},
        )

        if input_scan.blocked and self._config.get("block_on_input", True):
            raise InputBlockedError(
                f"Input blocked by Sentinel (risk={input_scan.risk.value})",
                scan_result=input_scan,
            )

        # Check if streaming requested — not supported via this path
        if kwargs.get("stream", False):
            raise ValueError(
                "Use client.chat.completions.create_stream() for streaming, "
                "or set stream=False"
            )

        # Call the real API
        response = self._completions.create(**kwargs)

        # Scan output
        output_text = _extract_completion_text(response)
        output_scan = self._guard.scan(
            output_text,
            context={"direction": "output", "model": kwargs.get("model", "unknown")},
        )

        if output_scan.blocked and self._config.get("block_on_output", True):
            raise OutputBlockedError(
                f"Output blocked by Sentinel (risk={output_scan.risk.value})",
                scan_result=output_scan,
            )

        # Prompt leak detection
        if system_text and self._config.get("detect_prompt_leak", True):
            leak_detector = PromptLeakDetector(system_text)
            leak_result = leak_detector.check(output_text)
            if leak_result.leaked:
                output_scan.findings.append(
                    Finding(
                        scanner="prompt_leak",
                        category="prompt_leak",
                        description=(
                            f"System prompt leak: {leak_result.overlap_ratio:.0%} overlap"
                        ),
                        risk=RiskLevel.CRITICAL,
                        metadata={
                            "overlap_ratio": leak_result.overlap_ratio,
                            "risk_score": leak_result.risk_score,
                        },
                    )
                )
                output_scan.risk = RiskLevel.CRITICAL
                output_scan.blocked = True
                if self._config.get("block_on_output", True):
                    raise OutputBlockedError(
                        "System prompt leak detected in model output",
                        scan_result=output_scan,
                    )

        return SafeChatResponse(
            response=response,
            input_scan=input_scan,
            output_scan=output_scan,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class SafeChat:
    """Wraps ``client.chat`` namespace."""

    def __init__(self, chat: Any, guard: SentinelGuard, config: dict):
        self._chat = chat
        self._guard = guard
        self._config = config

    @property
    def completions(self) -> SafeChatCompletions:
        return SafeChatCompletions(self._chat.completions, self._guard, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


# ---------------------------------------------------------------------------
# Main client wrappers
# ---------------------------------------------------------------------------

class SafeOpenAI:
    """Drop-in replacement for ``openai.OpenAI`` with safety scanning.

    All ``chat.completions.create()`` calls are automatically scanned.

    Args:
        guard: SentinelGuard instance (defaults to SentinelGuard.default())
        block_on_input: Raise InputBlockedError on unsafe input (default True)
        block_on_output: Raise OutputBlockedError on unsafe output (default True)
        detect_prompt_leak: Detect system prompt leaking (default True)
        **kwargs: Forwarded to ``openai.OpenAI()``.
    """

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        block_on_input: bool = True,
        block_on_output: bool = True,
        detect_prompt_leak: bool = True,
        **kwargs: Any,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required. Install it with: "
                "pip install openai"
            )

        self._client = openai.OpenAI(**kwargs)
        self._guard = guard or SentinelGuard.default()
        self._config = {
            "block_on_input": block_on_input,
            "block_on_output": block_on_output,
            "detect_prompt_leak": detect_prompt_leak,
        }

    @property
    def chat(self) -> SafeChat:
        return SafeChat(self._client.chat, self._guard, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class SafeAsyncOpenAI:
    """Drop-in replacement for ``openai.AsyncOpenAI`` with safety scanning.

    Args:
        guard: SentinelGuard instance (defaults to SentinelGuard.default())
        block_on_input: Raise InputBlockedError on unsafe input (default True)
        block_on_output: Raise OutputBlockedError on unsafe output (default True)
        detect_prompt_leak: Detect system prompt leaking (default True)
        **kwargs: Forwarded to ``openai.AsyncOpenAI()``.
    """

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        block_on_input: bool = True,
        block_on_output: bool = True,
        detect_prompt_leak: bool = True,
        **kwargs: Any,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required. Install it with: "
                "pip install openai"
            )

        self._client = openai.AsyncOpenAI(**kwargs)
        self._guard = guard or SentinelGuard.default()
        self._config = {
            "block_on_input": block_on_input,
            "block_on_output": block_on_output,
            "detect_prompt_leak": detect_prompt_leak,
        }

    @property
    def chat(self) -> SafeChat:
        return SafeChat(self._client.chat, self._guard, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
