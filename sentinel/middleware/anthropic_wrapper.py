"""Native Anthropic/Claude SDK integration.

Usage:
    from anthropic import Anthropic
    from sentinel.middleware.anthropic_wrapper import guarded_message

    client = Anthropic()
    result = guarded_message(
        client,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    if not result["blocked"]:
        print(result["response"].content[0].text)
"""

from __future__ import annotations

from typing import Any
from sentinel.core import SentinelGuard, ScanResult


def guarded_message(
    client: Any,
    guard: SentinelGuard | None = None,
    scan_input: bool = True,
    scan_output: bool = True,
    **kwargs: Any,
) -> dict:
    """Wrap an Anthropic messages.create() call with Sentinel scanning."""
    if guard is None:
        guard = SentinelGuard.default()

    result: dict[str, Any] = {"input_scan": None, "output_scan": None, "blocked": False}

    # Scan input messages
    if scan_input:
        messages = kwargs.get("messages", [])
        user_texts = []
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    user_texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_texts.append(block["text"])

        if user_texts:
            input_scan = guard.scan(" ".join(user_texts))
            result["input_scan"] = input_scan
            if input_scan.blocked:
                result["response"] = None
                result["blocked"] = True
                result["block_reason"] = "Input blocked by Sentinel safety scan"
                return result

    # Also scan system prompt if present
    if scan_input and "system" in kwargs:
        system = kwargs["system"]
        if isinstance(system, str):
            sys_scan = guard.scan(system)
            if sys_scan.blocked:
                result["response"] = None
                result["blocked"] = True
                result["block_reason"] = "System prompt blocked by Sentinel safety scan"
                return result

    # Call the Anthropic API
    response = client.messages.create(**kwargs)
    result["response"] = response

    # Scan output
    if scan_output:
        output_texts = []
        for block in response.content:
            if hasattr(block, "text"):
                output_texts.append(block.text)

        if output_texts:
            output_scan = guard.scan(" ".join(output_texts))
            result["output_scan"] = output_scan
            if output_scan.blocked:
                result["blocked"] = True
                result["block_reason"] = "Output blocked by Sentinel safety scan"
            elif output_scan.redacted_text:
                result["redacted_output"] = output_scan.redacted_text

    return result


async def guarded_message_async(
    client: Any,
    guard: SentinelGuard | None = None,
    scan_input: bool = True,
    scan_output: bool = True,
    **kwargs: Any,
) -> dict:
    """Async version for use with anthropic.AsyncAnthropic."""
    if guard is None:
        guard = SentinelGuard.default()

    result: dict[str, Any] = {"input_scan": None, "output_scan": None, "blocked": False}

    if scan_input:
        messages = kwargs.get("messages", [])
        user_texts = []
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    user_texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_texts.append(block["text"])

        if user_texts:
            input_scan = guard.scan(" ".join(user_texts))
            result["input_scan"] = input_scan
            if input_scan.blocked:
                result["response"] = None
                result["blocked"] = True
                result["block_reason"] = "Input blocked by Sentinel safety scan"
                return result

    response = await client.messages.create(**kwargs)
    result["response"] = response

    if scan_output:
        output_texts = []
        for block in response.content:
            if hasattr(block, "text"):
                output_texts.append(block.text)

        if output_texts:
            output_scan = guard.scan(" ".join(output_texts))
            result["output_scan"] = output_scan
            if output_scan.blocked:
                result["blocked"] = True
                result["block_reason"] = "Output blocked by Sentinel safety scan"
            elif output_scan.redacted_text:
                result["redacted_output"] = output_scan.redacted_text

    return result
