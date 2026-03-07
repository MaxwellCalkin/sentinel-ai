"""Drop-in middleware for OpenAI and Anthropic SDK calls.

Usage:
    from sentinel.middleware.openai_wrapper import guarded_chat
    response = guarded_chat(client, model="gpt-4", messages=[...])
"""

from __future__ import annotations

from typing import Any
from sentinel.core import SentinelGuard, ScanResult


def guarded_chat(
    client: Any,
    guard: SentinelGuard | None = None,
    scan_input: bool = True,
    scan_output: bool = True,
    **kwargs: Any,
) -> dict:
    """Wrap an OpenAI-compatible chat completion with Sentinel scanning.

    Returns a dict with 'response' (the raw API response), 'input_scan',
    and 'output_scan' results.
    """
    if guard is None:
        guard = SentinelGuard.default()

    result: dict[str, Any] = {"input_scan": None, "output_scan": None}

    # Scan input messages
    if scan_input:
        messages = kwargs.get("messages", [])
        user_text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        if user_text:
            input_result = guard.scan(user_text)
            result["input_scan"] = input_result
            if input_result.blocked:
                result["response"] = None
                result["blocked"] = True
                result["block_reason"] = "Input blocked by Sentinel safety scan"
                return result

    # Call the LLM
    response = client.chat.completions.create(**kwargs)
    result["response"] = response

    # Scan output
    if scan_output:
        output_text = response.choices[0].message.content or ""
        if output_text:
            output_result = guard.scan(output_text)
            result["output_scan"] = output_result
            if output_result.blocked:
                result["blocked"] = True
                result["block_reason"] = "Output blocked by Sentinel safety scan"
            elif output_result.redacted_text:
                result["redacted_output"] = output_result.redacted_text

    result.setdefault("blocked", False)
    return result
