"""LLM API Firewall — transparent safety proxy for any LLM API.

Sits between your application and any LLM API (Anthropic, OpenAI, etc.),
scanning all requests and responses for safety issues in real-time.

Usage:
    # Start the proxy (proxies to Anthropic API by default):
    sentinel proxy --target https://api.anthropic.com --port 8330

    # Point your app at the proxy instead of the real API:
    export ANTHROPIC_BASE_URL=http://localhost:8330

    # Or for OpenAI:
    sentinel proxy --target https://api.openai.com --port 8330
    export OPENAI_BASE_URL=http://localhost:8330

    # Python usage:
    from anthropic import Anthropic
    client = Anthropic(base_url="http://localhost:8330")
    # All requests are now safety-scanned transparently

How it works:
    1. Intercepts outgoing messages.create requests
    2. Scans user messages for prompt injection, harmful content
    3. Forwards safe requests to the real API
    4. Scans API responses for PII, harmful content, prompt leakage
    5. Returns scan metadata in X-Sentinel-* response headers
    6. Blocks dangerous requests/responses with 422 status
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional
from dataclasses import dataclass, field

from sentinel.core import SentinelGuard, RiskLevel


@dataclass
class ProxyConfig:
    """Configuration for the LLM API firewall."""

    target_url: str = "https://api.anthropic.com"
    port: int = 8330
    scan_input: bool = True
    scan_output: bool = True
    block_threshold: RiskLevel = RiskLevel.HIGH
    redact_pii: bool = True
    log_findings: bool = True
    pass_auth: bool = True
    allowed_models: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)


def _extract_user_text(body: dict) -> str:
    """Extract scannable text from an LLM API request body."""
    texts = []

    # System prompt
    system = body.get("system", "")
    if isinstance(system, str) and system:
        texts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block["text"])

    # Messages
    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))

    return " ".join(texts)


def _extract_response_text(body: dict) -> str:
    """Extract scannable text from an LLM API response body."""
    texts = []
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return " ".join(texts)


def _extract_tool_calls(body: dict) -> list[dict]:
    """Extract tool_use blocks from a response."""
    tools = []
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tools.append({
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            })
    return tools


def create_proxy_app(
    config: ProxyConfig | None = None,
    guard: SentinelGuard | None = None,
) -> Any:
    """Create a FastAPI proxy application.

    Returns a FastAPI app that acts as a transparent reverse proxy
    with safety scanning for LLM API requests and responses.
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import httpx

    if config is None:
        config = ProxyConfig()
    if guard is None:
        guard = SentinelGuard.default()

    # Persistent HTTP client for proxying
    http_client = httpx.AsyncClient(
        base_url=config.target_url,
        timeout=120.0,
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        await http_client.aclose()

    app = FastAPI(
        title="Sentinel AI Firewall",
        description="Transparent safety proxy for LLM APIs",
        lifespan=lifespan,
    )

    @app.get("/_sentinel/health")
    async def health():
        return {
            "status": "ok",
            "mode": "proxy",
            "target": config.target_url,
            "scan_input": config.scan_input,
            "scan_output": config.scan_output,
        }

    @app.get("/_sentinel/stats")
    async def stats():
        return {
            "requests_scanned": _stats["requests"],
            "requests_blocked": _stats["blocked"],
            "findings_total": _stats["findings"],
            "pii_redacted": _stats["pii_redacted"],
        }

    _stats = {"requests": 0, "blocked": 0, "findings": 0, "pii_redacted": 0}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, path: str):
        _stats["requests"] += 1

        # Read request body
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8") if body_bytes else ""

        # For non-message endpoints, pass through directly
        is_messages = path.rstrip("/").endswith("/messages")
        if not is_messages or request.method != "POST":
            return await _forward_request(request, path, body_bytes, http_client)

        # Parse the request body
        try:
            body = json.loads(body_text)
        except (json.JSONDecodeError, ValueError):
            return await _forward_request(request, path, body_bytes, http_client)

        # Check model allowlist
        if config.allowed_models:
            model = body.get("model", "")
            if model and model not in config.allowed_models:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "Model not allowed by Sentinel AI firewall",
                        "model": model,
                        "allowed_models": config.allowed_models,
                    },
                    headers={"X-Sentinel-Blocked": "true"},
                )

        # Scan input
        sentinel_headers = {}
        if config.scan_input:
            user_text = _extract_user_text(body)
            if user_text:
                start = time.perf_counter()
                input_scan = guard.scan(user_text)
                scan_ms = (time.perf_counter() - start) * 1000

                sentinel_headers["X-Sentinel-Input-Risk"] = input_scan.risk.value
                sentinel_headers["X-Sentinel-Input-Scan-Ms"] = f"{scan_ms:.2f}"

                if input_scan.findings:
                    _stats["findings"] += len(input_scan.findings)
                    sentinel_headers["X-Sentinel-Input-Findings"] = str(
                        len(input_scan.findings)
                    )

                if input_scan.risk >= config.block_threshold:
                    _stats["blocked"] += 1
                    findings_summary = [
                        {
                            "category": f.category,
                            "description": f.description,
                            "risk": f.risk.value,
                        }
                        for f in input_scan.findings[:5]
                    ]
                    return JSONResponse(
                        status_code=422,
                        content={
                            "error": "Request blocked by Sentinel AI firewall",
                            "risk": input_scan.risk.value,
                            "findings": findings_summary,
                        },
                        headers={
                            "X-Sentinel-Blocked": "true",
                            **sentinel_headers,
                        },
                    )

        # Check if streaming
        is_streaming = body.get("stream", False)

        if is_streaming:
            # For streaming, forward and scan chunks
            return await _forward_streaming(
                request, path, body_bytes, http_client,
                guard, config, sentinel_headers, _stats,
            )

        # Forward the request to the target API
        headers = _proxy_headers(request, config)
        response = await http_client.post(
            f"/{path}",
            content=body_bytes,
            headers=headers,
        )

        # Parse and scan the response
        response_headers = dict(response.headers)
        response_headers.update(sentinel_headers)

        if config.scan_output and response.status_code == 200:
            try:
                resp_body = response.json()
                resp_text = _extract_response_text(resp_body)

                if resp_text:
                    start = time.perf_counter()
                    output_scan = guard.scan(resp_text)
                    scan_ms = (time.perf_counter() - start) * 1000

                    response_headers["X-Sentinel-Output-Risk"] = output_scan.risk.value
                    response_headers["X-Sentinel-Output-Scan-Ms"] = f"{scan_ms:.2f}"

                    if output_scan.findings:
                        _stats["findings"] += len(output_scan.findings)
                        response_headers["X-Sentinel-Output-Findings"] = str(
                            len(output_scan.findings)
                        )

                    # Redact PII in response
                    if config.redact_pii and output_scan.redacted_text:
                        _stats["pii_redacted"] += 1
                        for block in resp_body.get("content", []):
                            if block.get("type") == "text":
                                block["text"] = output_scan.redacted_text
                        return JSONResponse(
                            status_code=200,
                            content=resp_body,
                            headers=response_headers,
                        )

                    # Block dangerous output
                    if output_scan.risk >= config.block_threshold:
                        _stats["blocked"] += 1
                        return JSONResponse(
                            status_code=422,
                            content={
                                "error": "Response blocked by Sentinel AI firewall",
                                "risk": output_scan.risk.value,
                            },
                            headers={
                                "X-Sentinel-Blocked": "true",
                                **response_headers,
                            },
                        )

                # Scan tool calls
                tool_calls = _extract_tool_calls(resp_body)
                if tool_calls and config.scan_output:
                    from sentinel.scanners.tool_use import ToolUseScanner
                    tool_scanner = ToolUseScanner()
                    for tc in tool_calls:
                        if tc["name"] in config.blocked_tools:
                            _stats["blocked"] += 1
                            return JSONResponse(
                                status_code=422,
                                content={
                                    "error": f"Tool '{tc['name']}' blocked by firewall policy",
                                },
                                headers={"X-Sentinel-Blocked": "true"},
                            )
                        findings = tool_scanner.scan_tool_call(tc["name"], tc["input"])
                        if findings:
                            max_risk = max(f.risk for f in findings)
                            if max_risk >= config.block_threshold:
                                _stats["blocked"] += 1
                                return JSONResponse(
                                    status_code=422,
                                    content={
                                        "error": "Tool call blocked by Sentinel AI firewall",
                                        "tool": tc["name"],
                                        "risk": max_risk.value,
                                    },
                                    headers={"X-Sentinel-Blocked": "true"},
                                )

                return JSONResponse(
                    status_code=response.status_code,
                    content=resp_body,
                    headers=response_headers,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Pass through unmodified for non-200 or unparseable responses
        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text},
            headers=response_headers,
        )

    return app


def _proxy_headers(request: Any, config: ProxyConfig) -> dict:
    """Build headers for the upstream request."""
    headers = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in ("host", "content-length", "transfer-encoding"):
            continue
        if lower == "authorization" and config.pass_auth:
            headers[key] = value
        elif lower.startswith("x-") or lower in (
            "anthropic-version", "content-type", "accept",
        ):
            headers[key] = value
    return headers


async def _forward_request(request: Any, path: str, body: bytes, client: Any) -> Any:
    """Forward a request to the target API without scanning."""
    from fastapi.responses import Response

    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ("host", "content-length", "transfer-encoding"):
            headers[key] = value

    response = await client.request(
        method=request.method,
        url=f"/{path}",
        content=body,
        headers=headers,
    )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )


async def _forward_streaming(
    request: Any,
    path: str,
    body: bytes,
    client: Any,
    guard: SentinelGuard,
    config: ProxyConfig,
    sentinel_headers: dict,
    stats: dict,
) -> Any:
    """Forward a streaming request with real-time safety scanning."""
    from fastapi.responses import StreamingResponse
    from sentinel.streaming import StreamingGuard

    headers = _proxy_headers(request, config)

    streaming_guard = StreamingGuard(guard=guard)

    async def stream_with_scanning():
        async with client.stream(
            "POST", f"/{path}", content=body, headers=headers
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    yield line + "\n"
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    yield line + "\n"
                    continue

                try:
                    event = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    yield line + "\n"
                    continue

                # Extract text from streaming event
                event_type = event.get("type", "")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text and config.scan_output:
                            chunk_result = streaming_guard.feed(text)
                            if chunk_result.blocked:
                                stats["blocked"] += 1
                                # Send an error event and stop
                                error_event = {
                                    "type": "error",
                                    "error": {
                                        "type": "sentinel_blocked",
                                        "message": "Output blocked by Sentinel AI firewall",
                                    },
                                }
                                yield f"data: {json.dumps(error_event)}\n\n"
                                return

                yield line + "\n"

            # Finalize streaming guard
            if config.scan_output:
                final = streaming_guard.finalize()
                if final.findings:
                    stats["findings"] += len(final.findings)

    return StreamingResponse(
        stream_with_scanning(),
        media_type="text/event-stream",
        headers=sentinel_headers,
    )
