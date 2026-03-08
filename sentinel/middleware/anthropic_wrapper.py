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

Streaming usage:
    from sentinel.middleware.anthropic_wrapper import guarded_stream

    for event in guarded_stream(
        client,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    ):
        if event["blocked"]:
            print(f"BLOCKED: {event['block_reason']}")
            break
        if event["text"]:
            print(event["text"], end="", flush=True)
    # After loop, event has final scan results
"""

from __future__ import annotations

from typing import Any, Generator, AsyncGenerator
from sentinel.core import SentinelGuard, ScanResult, RiskLevel
from sentinel.streaming import StreamingGuard


def _scan_inputs(
    guard: SentinelGuard,
    result: dict[str, Any],
    kwargs: dict[str, Any],
) -> bool:
    """Scan input messages and system prompt. Returns True if blocked."""
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
            return True

    if "system" in kwargs:
        system = kwargs["system"]
        if isinstance(system, str):
            sys_scan = guard.scan(system)
            if sys_scan.blocked:
                result["response"] = None
                result["blocked"] = True
                result["block_reason"] = "System prompt blocked by Sentinel safety scan"
                return True

    return False


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

    if scan_input and _scan_inputs(guard, result, kwargs):
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

    # Scan tool_use blocks in the response
    if scan_output:
        from sentinel.scanners.tool_use import ToolUseScanner
        tool_scanner = ToolUseScanner()
        tool_findings = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                findings = tool_scanner.scan_tool_call(
                    block.name, block.input if hasattr(block, "input") else {}
                )
                tool_findings.extend(findings)
        if tool_findings:
            result["tool_findings"] = tool_findings
            from sentinel.core import RiskLevel
            max_risk = max(f.risk for f in tool_findings)
            if max_risk >= RiskLevel.HIGH:
                result["blocked"] = True
                result["block_reason"] = "Tool call blocked by Sentinel safety scan"

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

    if scan_input and _scan_inputs(guard, result, kwargs):
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

    if scan_output:
        from sentinel.scanners.tool_use import ToolUseScanner
        tool_scanner = ToolUseScanner()
        tool_findings = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                findings = tool_scanner.scan_tool_call(
                    block.name, block.input if hasattr(block, "input") else {}
                )
                tool_findings.extend(findings)
        if tool_findings:
            result["tool_findings"] = tool_findings
            max_risk = max(f.risk for f in tool_findings)
            if max_risk >= RiskLevel.HIGH:
                result["blocked"] = True
                result["block_reason"] = "Tool call blocked by Sentinel safety scan"

    return result


def guarded_stream(
    client: Any,
    guard: SentinelGuard | None = None,
    scan_input: bool = True,
    **kwargs: Any,
) -> Generator[dict[str, Any], None, None]:
    """Stream Anthropic messages with real-time safety scanning.

    Yields events with:
        - text: safe text chunk to display (empty string if buffered/blocked)
        - blocked: True if stream was blocked by safety scan
        - block_reason: reason for blocking (if blocked)
        - findings: list of findings detected so far
        - done: True on the final event

    Usage:
        for event in guarded_stream(client, model="claude-sonnet-4-6", ...):
            if event["blocked"]:
                print(f"BLOCKED: {event['block_reason']}")
                break
            print(event["text"], end="", flush=True)
    """
    if guard is None:
        guard = SentinelGuard.default()

    # Scan input first
    result: dict[str, Any] = {"input_scan": None, "output_scan": None, "blocked": False}
    if scan_input and _scan_inputs(guard, result, kwargs):
        yield {
            "text": "",
            "blocked": True,
            "block_reason": result["block_reason"],
            "findings": [],
            "done": True,
        }
        return

    streaming_guard = StreamingGuard(guard=guard)

    # Force stream=True
    kwargs["stream"] = True
    stream = client.messages.create(**kwargs)

    try:
        for event in stream:
            # Extract text from streaming events
            event_type = getattr(event, "type", "")

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", "") == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        chunk_result = streaming_guard.feed(text)
                        if chunk_result.blocked:
                            yield {
                                "text": "",
                                "blocked": True,
                                "block_reason": "Output blocked by Sentinel safety scan",
                                "findings": chunk_result.findings,
                                "done": True,
                            }
                            return
                        yield {
                            "text": chunk_result.safe_text,
                            "blocked": False,
                            "block_reason": None,
                            "findings": chunk_result.findings,
                            "done": False,
                        }

            elif event_type == "message_stop":
                final = streaming_guard.finalize()
                yield {
                    "text": final.safe_text,
                    "blocked": final.blocked,
                    "block_reason": "Output blocked by Sentinel safety scan" if final.blocked else None,
                    "findings": streaming_guard.all_findings,
                    "done": True,
                    "full_text": streaming_guard.full_text,
                    "redacted_text": final.redacted_text,
                }
                return
    finally:
        if hasattr(stream, "close"):
            stream.close()

    # If we exit the loop without message_stop, finalize anyway
    final = streaming_guard.finalize()
    yield {
        "text": final.safe_text,
        "blocked": final.blocked,
        "block_reason": "Output blocked by Sentinel safety scan" if final.blocked else None,
        "findings": streaming_guard.all_findings,
        "done": True,
        "full_text": streaming_guard.full_text,
        "redacted_text": final.redacted_text,
    }


async def guarded_stream_async(
    client: Any,
    guard: SentinelGuard | None = None,
    scan_input: bool = True,
    **kwargs: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Async streaming version with real-time safety scanning.

    Usage:
        async for event in guarded_stream_async(client, model="...", ...):
            if event["blocked"]:
                break
            print(event["text"], end="", flush=True)
    """
    if guard is None:
        guard = SentinelGuard.default()

    result: dict[str, Any] = {"input_scan": None, "output_scan": None, "blocked": False}
    if scan_input and _scan_inputs(guard, result, kwargs):
        yield {
            "text": "",
            "blocked": True,
            "block_reason": result["block_reason"],
            "findings": [],
            "done": True,
        }
        return

    streaming_guard = StreamingGuard(guard=guard)

    kwargs["stream"] = True
    stream = await client.messages.create(**kwargs)

    try:
        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", "") == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        chunk_result = streaming_guard.feed(text)
                        if chunk_result.blocked:
                            yield {
                                "text": "",
                                "blocked": True,
                                "block_reason": "Output blocked by Sentinel safety scan",
                                "findings": chunk_result.findings,
                                "done": True,
                            }
                            return
                        yield {
                            "text": chunk_result.safe_text,
                            "blocked": False,
                            "block_reason": None,
                            "findings": chunk_result.findings,
                            "done": False,
                        }

            elif event_type == "message_stop":
                final = streaming_guard.finalize()
                yield {
                    "text": final.safe_text,
                    "blocked": final.blocked,
                    "block_reason": "Output blocked by Sentinel safety scan" if final.blocked else None,
                    "findings": streaming_guard.all_findings,
                    "done": True,
                    "full_text": streaming_guard.full_text,
                    "redacted_text": final.redacted_text,
                }
                return
    finally:
        if hasattr(stream, "close"):
            await stream.close()

    final = streaming_guard.finalize()
    yield {
        "text": final.safe_text,
        "blocked": final.blocked,
        "block_reason": "Output blocked by Sentinel safety scan" if final.blocked else None,
        "findings": streaming_guard.all_findings,
        "done": True,
        "full_text": streaming_guard.full_text,
        "redacted_text": final.redacted_text,
    }
