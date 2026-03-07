"""Sentinel AI command-line interface.

Usage:
    sentinel scan "text to scan"
    sentinel scan --file input.txt
    sentinel scan --stdin < file.txt
    echo "text" | sentinel scan --stdin
    sentinel serve --port 8329
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentinel.core import SentinelGuard, ScanResult


def _format_result(result: ScanResult, fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps(
            {
                "safe": result.safe,
                "blocked": result.blocked,
                "risk": result.risk.value,
                "findings": [
                    {
                        "scanner": f.scanner,
                        "category": f.category,
                        "description": f.description,
                        "risk": f.risk.value,
                        "span": list(f.span) if f.span else None,
                        "metadata": f.metadata,
                    }
                    for f in result.findings
                ],
                "redacted_text": result.redacted_text,
                "latency_ms": result.latency_ms,
            },
            indent=2,
        )

    lines = []
    status = "SAFE" if result.safe else ("BLOCKED" if result.blocked else "RISKY")
    lines.append(f"Status: {status} (risk: {result.risk.value})")
    lines.append(f"Latency: {result.latency_ms}ms")

    if result.findings:
        lines.append(f"\nFindings ({len(result.findings)}):")
        for i, f in enumerate(result.findings, 1):
            lines.append(f"  {i}. [{f.risk.value.upper()}] {f.description}")
            lines.append(f"     Scanner: {f.scanner} | Category: {f.category}")
            if f.span:
                lines.append(f"     Position: {f.span[0]}-{f.span[1]}")
    else:
        lines.append("\nNo findings.")

    if result.redacted_text:
        lines.append(f"\nRedacted output:\n  {result.redacted_text}")

    return "\n".join(lines)


def cmd_scan(args: argparse.Namespace) -> int:
    guard = SentinelGuard.default()

    if args.stdin:
        text = sys.stdin.read()
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = " ".join(args.text)
    else:
        print("Error: provide text, --file, or --stdin", file=sys.stderr)
        return 1

    if not text.strip():
        print("Error: empty input", file=sys.stderr)
        return 1

    result = guard.scan(text)
    print(_format_result(result, args.format))
    return 1 if result.blocked else 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Error: install sentinel-ai[api] for server support", file=sys.stderr)
        return 1

    print(f"Starting Sentinel AI server on port {args.port}...")
    uvicorn.run("sentinel.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel AI - Real-time safety guardrails for LLMs",
    )
    parser.add_argument("--version", action="version", version="sentinel-ai 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan text for safety issues")
    scan_parser.add_argument("text", nargs="*", help="Text to scan")
    scan_parser.add_argument("--file", "-f", help="Read text from file")
    scan_parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    scan_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", "-p", type=int, default=8329, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args(argv)

    if args.command == "scan":
        return cmd_scan(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
