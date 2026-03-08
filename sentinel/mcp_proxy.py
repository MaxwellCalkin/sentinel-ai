"""MCP Safety Proxy — transparent safety layer between Claude and any MCP server.

Routes all tool calls through Sentinel's safety scanners before forwarding
to the upstream MCP server. Blocks dangerous tool arguments and scans
responses for PII leaks before returning to Claude.

Usage:
    # Proxy any MCP server through Sentinel:
    sentinel mcp-proxy -- npx @modelcontextprotocol/server-filesystem /tmp

    # In Claude Desktop config:
    {
      "mcpServers": {
        "safe-filesystem": {
          "command": "sentinel",
          "args": ["mcp-proxy", "--", "npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
      }
    }
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

from sentinel.core import SentinelGuard, RiskLevel


def _read_mcp_message(stream) -> dict | None:
    """Read a single MCP JSON-RPC message from a stream."""
    header = ""
    while True:
        line = stream.readline()
        if not line:
            return None
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        header += line
        if header.endswith("\r\n\r\n") or header.endswith("\n\n"):
            break

    content_length = 0
    for h in header.strip().split("\n"):
        h = h.strip()
        if h.lower().startswith("content-length:"):
            content_length = int(h.split(":", 1)[1].strip())

    if content_length == 0:
        return None

    body = stream.read(content_length)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return json.loads(body)


def _write_mcp_message(stream, message: dict) -> None:
    """Write a single MCP JSON-RPC message to a stream."""
    data = json.dumps(message)
    out = f"Content-Length: {len(data)}\r\n\r\n{data}"
    if hasattr(stream, "buffer"):
        stream.buffer.write(out.encode("utf-8"))
        stream.buffer.flush()
    else:
        stream.write(out)
        stream.flush()


def _scan_tool_arguments(guard: SentinelGuard, tool_name: str, arguments: dict) -> list:
    """Scan tool call arguments for safety issues."""
    from sentinel.scanners.tool_use import ToolUseScanner

    findings = []

    # Scan with the tool-use scanner
    tool_scanner = ToolUseScanner()
    findings.extend(tool_scanner.scan_tool_call(tool_name, arguments))

    # Also scan string values through the full guard
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 3:
            result = guard.scan(value)
            findings.extend(result.findings)

    return findings


def _scan_response(guard: SentinelGuard, result: dict) -> tuple[dict, list]:
    """Scan tool response content for PII and safety issues.

    Returns (possibly-redacted result, findings).
    """
    findings = []
    content = result.get("content", [])
    modified = False

    for i, item in enumerate(content):
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            scan = guard.scan(item["text"])
            findings.extend(scan.findings)

            # Auto-redact PII in responses
            pii_found = any(f.category == "pii" for f in scan.findings)
            if pii_found and scan.redacted_text:
                content[i] = {**item, "text": scan.redacted_text}
                modified = True

    if modified:
        result = {**result, "content": content}

    return result, findings


def _make_block_response(msg_id: Any, tool_name: str, findings: list) -> dict:
    """Create a JSON-RPC error response when a tool call is blocked."""
    descriptions = [f.description for f in findings[:3]]
    max_risk = max((f.risk for f in findings), default=RiskLevel.NONE)

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"BLOCKED by Sentinel AI safety proxy.\n"
                        f"Tool: {tool_name}\n"
                        f"Risk: {max_risk.value.upper()}\n"
                        f"Findings:\n"
                        + "\n".join(f"  - {d}" for d in descriptions)
                    ),
                }
            ],
            "isError": True,
        },
    }


def run_proxy(upstream_cmd: list[str], *, block_threshold: RiskLevel = RiskLevel.HIGH) -> int:
    """Run the MCP safety proxy.

    Args:
        upstream_cmd: Command to start the upstream MCP server.
        block_threshold: Minimum risk level to block a tool call.

    Returns:
        Exit code.
    """
    guard = SentinelGuard.default()

    # Start upstream MCP server
    try:
        upstream = subprocess.Popen(
            upstream_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
    except FileNotFoundError:
        print(
            f"Error: could not start upstream MCP server: {upstream_cmd[0]}",
            file=sys.stderr,
        )
        return 1

    stats = {"scanned": 0, "blocked": 0, "pii_redacted": 0}

    def _log(msg: str) -> None:
        print(f"[sentinel-proxy] {msg}", file=sys.stderr)

    _log(f"Proxying MCP server: {' '.join(upstream_cmd)}")

    try:
        while True:
            # Read message from Claude (stdin)
            message = _read_mcp_message(sys.stdin)
            if message is None:
                break

            msg_id = message.get("id")
            method = message.get("method", "")
            params = message.get("params", {})

            # For tool calls, scan arguments before forwarding
            if method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                stats["scanned"] += 1

                findings = _scan_tool_arguments(guard, tool_name, arguments)
                max_risk = max(
                    (f.risk for f in findings), default=RiskLevel.NONE
                )

                if max_risk >= block_threshold:
                    stats["blocked"] += 1
                    _log(
                        f"BLOCKED {tool_name}({json.dumps(arguments)[:100]}) "
                        f"— {max_risk.value}"
                    )
                    # Send block response directly to Claude, don't forward
                    response = _make_block_response(msg_id, tool_name, findings)
                    _write_mcp_message(sys.stdout, response)
                    continue
                elif findings:
                    _log(
                        f"WARNING {tool_name} — {len(findings)} finding(s), "
                        f"risk={max_risk.value} (below block threshold)"
                    )

            # Forward message to upstream
            _write_mcp_message(upstream.stdin, message)

            # Read response from upstream
            response = _read_mcp_message(upstream.stdout)
            if response is None:
                break

            # For tool call responses, scan the result
            if method == "tools/call" and "result" in response:
                result, resp_findings = _scan_response(
                    guard, response["result"]
                )
                if resp_findings:
                    pii_count = sum(
                        1 for f in resp_findings if f.category == "pii"
                    )
                    if pii_count:
                        stats["pii_redacted"] += 1
                        _log(
                            f"REDACTED {pii_count} PII item(s) in response "
                            f"from {params.get('name', '?')}"
                        )
                    response = {**response, "result": result}

            # Forward response to Claude
            _write_mcp_message(sys.stdout, response)

    except KeyboardInterrupt:
        pass
    finally:
        upstream.terminate()
        upstream.wait(timeout=5)
        _log(
            f"Proxy stats: {stats['scanned']} scanned, "
            f"{stats['blocked']} blocked, "
            f"{stats['pii_redacted']} PII-redacted"
        )

    return 0
