"""Sentinel AI MCP (Model Context Protocol) server.

Exposes Sentinel's safety scanning as MCP tools that any MCP-compatible
client can use — including Claude Desktop, Claude Code, and custom agents.

Usage:
    # Run the server (stdio transport for MCP):
    python -m sentinel.mcp_server

    # Add to Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "sentinel-ai": {
          "command": "python",
          "args": ["-m", "sentinel.mcp_server"]
        }
      }
    }

    # Or with uvx (if published to PyPI):
    {
      "mcpServers": {
        "sentinel-ai": {
          "command": "uvx",
          "args": ["sentinel-ai", "--mcp"]
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any

from sentinel.core import SentinelGuard, RiskLevel


def _create_guard() -> SentinelGuard:
    return SentinelGuard.default()


_guard = _create_guard()


# --- MCP Protocol Implementation (stdio JSON-RPC) ---

def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {},
        },
        "serverInfo": {
            "name": "sentinel-ai",
            "version": "0.6.0",
        },
    }


def handle_tools_list(params: dict) -> dict:
    return {
        "tools": [
            {
                "name": "scan_text",
                "description": (
                    "Scan text for safety issues: prompt injection, PII, "
                    "harmful content, hallucination, toxicity, and dangerous "
                    "tool-use patterns. Returns risk level, findings, and "
                    "optionally redacted text."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to scan for safety issues",
                        },
                        "scanners": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of specific scanners to use. "
                                "Available: prompt_injection, pii, harmful_content, "
                                "hallucination, toxicity, tool_use"
                            ),
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "scan_tool_call",
                "description": (
                    "Scan a tool call for safety issues. Checks tool name "
                    "and arguments for dangerous shell commands, data "
                    "exfiltration, credential access, and privilege escalation."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool being called",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "The arguments being passed to the tool",
                        },
                    },
                    "required": ["tool_name", "arguments"],
                },
            },
            {
                "name": "check_pii",
                "description": (
                    "Check text for PII (personally identifiable information) "
                    "and return redacted version. Detects emails, SSNs, credit "
                    "cards, phone numbers, API keys, and more."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to check for PII",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "get_risk_report",
                "description": (
                    "Generate a detailed safety risk report for a block of "
                    "text. Includes all findings across all scanners with "
                    "risk levels, categories, and remediation suggestions."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to generate risk report for",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "scan_conversation",
                "description": (
                    "Scan a multi-turn conversation for safety issues. "
                    "Detects gradual jailbreak escalation, topic persistence "
                    "attacks, sandwich attacks, and re-attempts after blocks. "
                    "Returns per-turn and conversation-level risk assessment."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["user", "assistant"],
                                    },
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                            "description": "List of conversation messages",
                        },
                    },
                    "required": ["messages"],
                },
            },
            {
                "name": "test_robustness",
                "description": (
                    "Test scanner robustness against adversarial evasion "
                    "techniques. Generates variants of an attack payload "
                    "(homoglyphs, leetspeak, zero-width chars, payload "
                    "splitting, synonym substitution) and reports which "
                    "ones evade detection."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Attack payload to test variants of",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "generate_rsp_report",
                "description": (
                    "Generate an RSP-aligned risk report following Anthropic's "
                    "Responsible Scaling Policy v3.0 framework. Covers threat "
                    "domain assessments, risk distribution, mitigations, and "
                    "actionable recommendations."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to assess",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "json"],
                            "description": "Output format (default: markdown)",
                        },
                    },
                    "required": ["texts"],
                },
            },
        ],
    }


def handle_tool_call(params: dict) -> dict:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "scan_text":
        return _tool_scan_text(arguments)
    elif tool_name == "scan_tool_call":
        return _tool_scan_tool_call(arguments)
    elif tool_name == "check_pii":
        return _tool_check_pii(arguments)
    elif tool_name == "get_risk_report":
        return _tool_get_risk_report(arguments)
    elif tool_name == "scan_conversation":
        return _tool_scan_conversation(arguments)
    elif tool_name == "test_robustness":
        return _tool_test_robustness(arguments)
    elif tool_name == "generate_rsp_report":
        return _tool_generate_rsp_report(arguments)
    else:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True,
        }


def _tool_scan_text(args: dict) -> dict:
    text = args.get("text", "")
    scanner_filter = args.get("scanners")

    result = _guard.scan(text)

    findings = result.findings
    if scanner_filter:
        findings = [f for f in findings if f.scanner in scanner_filter]

    output = {
        "safe": result.safe,
        "blocked": result.blocked,
        "risk": result.risk.value,
        "latency_ms": result.latency_ms,
        "findings_count": len(findings),
        "findings": [
            {
                "scanner": f.scanner,
                "category": f.category,
                "description": f.description,
                "risk": f.risk.value,
            }
            for f in findings
        ],
    }

    if result.redacted_text:
        output["redacted_text"] = result.redacted_text

    return {
        "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
    }


def _tool_scan_tool_call(args: dict) -> dict:
    from sentinel.scanners.tool_use import ToolUseScanner

    scanner = ToolUseScanner()
    tool_name = args.get("tool_name", "")
    arguments = args.get("arguments", {})

    findings = scanner.scan_tool_call(tool_name, arguments)

    output = {
        "safe": len(findings) == 0,
        "tool_name": tool_name,
        "findings_count": len(findings),
        "findings": [
            {
                "description": f.description,
                "risk": f.risk.value,
                "threat_type": f.metadata.get("threat_type", "unknown"),
            }
            for f in findings
        ],
    }

    if findings:
        max_risk = max(f.risk for f in findings)
        output["max_risk"] = max_risk.value
        output["recommendation"] = (
            "BLOCK" if max_risk >= RiskLevel.HIGH else "REVIEW"
        )

    return {
        "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
    }


def _tool_check_pii(args: dict) -> dict:
    text = args.get("text", "")

    from sentinel.scanners.pii import PIIScanner

    scanner = PIIScanner()
    findings = scanner.scan(text)

    result = _guard.scan(text)

    output = {
        "has_pii": len(findings) > 0,
        "pii_count": len(findings),
        "pii_types": list({f.metadata.get("pii_type", "UNKNOWN") for f in findings}),
    }

    if result.redacted_text:
        output["redacted_text"] = result.redacted_text
    elif findings:
        # Manually redact
        from sentinel.core import SentinelGuard
        guard = SentinelGuard(scanners=[scanner])
        r = guard.scan(text)
        if r.redacted_text:
            output["redacted_text"] = r.redacted_text

    output["findings"] = [
        {
            "type": f.metadata.get("pii_type", "UNKNOWN"),
            "risk": f.risk.value,
        }
        for f in findings
    ]

    return {
        "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
    }


def _tool_get_risk_report(args: dict) -> dict:
    text = args.get("text", "")
    result = _guard.scan(text)

    risk_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    for f in result.findings:
        risk_counts[f.risk.value] = risk_counts.get(f.risk.value, 0) + 1
        category_counts[f.category] = category_counts.get(f.category, 0) + 1

    report_lines = [
        f"# Safety Risk Report",
        f"",
        f"**Overall Risk**: {result.risk.value.upper()}",
        f"**Blocked**: {'Yes' if result.blocked else 'No'}",
        f"**Safe**: {'Yes' if result.safe else 'No'}",
        f"**Scan Latency**: {result.latency_ms}ms",
        f"**Total Findings**: {len(result.findings)}",
        f"",
    ]

    if result.findings:
        report_lines.append("## Findings by Category")
        for cat, count in sorted(category_counts.items()):
            report_lines.append(f"- **{cat}**: {count}")

        report_lines.append("")
        report_lines.append("## Findings by Risk")
        for risk, count in sorted(risk_counts.items()):
            report_lines.append(f"- **{risk}**: {count}")

        report_lines.append("")
        report_lines.append("## Detailed Findings")
        for i, f in enumerate(result.findings, 1):
            report_lines.append(
                f"{i}. [{f.risk.value.upper()}] **{f.category}**: {f.description}"
            )
    else:
        report_lines.append("No safety issues detected.")

    if result.redacted_text:
        report_lines.extend(["", "## Redacted Text", result.redacted_text])

    return {
        "content": [{"type": "text", "text": "\n".join(report_lines)}],
    }


def _tool_generate_rsp_report(args: dict) -> dict:
    from sentinel.rsp_report import RiskReportGenerator

    texts = args.get("texts", [])
    fmt = args.get("format", "markdown")

    generator = RiskReportGenerator(_guard)
    report = generator.generate(texts=texts)

    if fmt == "json":
        import json as json_mod
        output = json_mod.dumps(report.to_dict(), indent=2)
    else:
        output = report.to_markdown()

    return {
        "content": [{"type": "text", "text": output}],
    }


def _tool_test_robustness(args: dict) -> dict:
    from sentinel.adversarial import AdversarialTester

    text = args.get("text", "")
    tester = AdversarialTester(_guard)
    report = tester.test_robustness(text)

    output = {
        "original_text": report.original_text,
        "original_detected": report.original_detected,
        "total_variants": report.total_variants,
        "detected_count": report.detected_count,
        "evaded_count": report.evaded_count,
        "detection_rate": f"{report.detection_rate:.0%}",
        "evaded_techniques": [
            {"technique": v.technique, "description": v.description}
            for v in report.evaded
        ],
    }

    return {
        "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
    }


def _tool_scan_conversation(args: dict) -> dict:
    from sentinel.conversation import ConversationGuard

    messages = args.get("messages", [])
    conv = ConversationGuard(_guard)

    turns = []
    for msg in messages:
        result = conv.add_message(msg["role"], msg["content"])
        turns.append({
            "turn": result.turn_number,
            "role": result.role,
            "risk": result.scan.risk.value,
            "blocked": result.scan.blocked,
            "escalation": result.escalation_detected,
            "escalation_reason": result.escalation_reason,
            "findings_count": len(result.scan.findings),
        })

    summary = conv.summarize()
    output = {
        "conversation_risk": summary.conversation_risk.value,
        "total_turns": summary.total_turns,
        "blocked_turns": summary.blocked_turns,
        "escalations": summary.escalations,
        "category_counts": summary.category_counts,
        "flags": summary.flags,
        "turns": turns,
    }

    return {
        "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
    }


def _send_response(response: dict) -> None:
    """Send a JSON-RPC response via stdout."""
    data = json.dumps(response)
    message = f"Content-Length: {len(data)}\r\n\r\n{data}"
    sys.stdout.write(message)
    sys.stdout.flush()


def _read_message() -> dict | None:
    """Read a JSON-RPC message from stdin."""
    # Read Content-Length header
    header = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        header += line
        if header.endswith("\r\n\r\n") or header.endswith("\n\n"):
            break

    # Parse content length
    content_length = 0
    for line in header.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())

    if content_length == 0:
        return None

    # Read body
    body = sys.stdin.read(content_length)
    return json.loads(body)


def main() -> None:
    """Run the MCP server on stdio."""
    while True:
        message = _read_message()
        if message is None:
            break

        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})

        result = None
        error = None

        if method == "initialize":
            result = handle_initialize(params)
        elif method == "notifications/initialized":
            continue  # No response needed for notifications
        elif method == "tools/list":
            result = handle_tools_list(params)
        elif method == "tools/call":
            result = handle_tool_call(params)
        else:
            error = {
                "code": -32601,
                "message": f"Method not found: {method}",
            }

        if msg_id is not None:
            response: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
            if error:
                response["error"] = error
            else:
                response["result"] = result
            _send_response(response)


if __name__ == "__main__":
    main()
