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


def cmd_red_team(args: argparse.Namespace) -> int:
    from sentinel.adversarial import AdversarialTester
    from sentinel.core import RiskLevel

    if args.text:
        texts = [" ".join(args.text)]
    elif args.file:
        texts = [
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        print("Error: provide text or --file", file=sys.stderr)
        return 1

    tester = AdversarialTester()

    if len(texts) == 1:
        report = tester.test_robustness(texts[0])
        if args.format == "json":
            print(json.dumps({
                "original": report.original_text,
                "original_detected": report.original_detected,
                "total_variants": report.total_variants,
                "detected": report.detected_count,
                "evaded": report.evaded_count,
                "detection_rate": f"{report.detection_rate:.0%}",
                "evasion_techniques": [
                    {"technique": v.technique, "text": v.text}
                    for v in report.evaded
                ],
            }, indent=2))
        else:
            print(report.summary())
    else:
        batch = tester.test_batch(texts)
        if args.format == "json":
            print(json.dumps({
                "payloads_tested": len(batch.reports),
                "total_variants": batch.total_variants,
                "overall_detection_rate": f"{batch.overall_detection_rate:.0%}",
                "total_evasions": batch.total_evaded,
                "weak_techniques": batch.weak_techniques,
            }, indent=2))
        else:
            print(batch.summary())

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from sentinel.benchmarks import run_benchmark

    results = run_benchmark()
    if args.format == "json":
        print(json.dumps({
            "total_cases": results.total,
            "accuracy": f"{results.accuracy:.1%}",
            "precision": f"{results.precision:.1%}",
            "recall": f"{results.recall:.1%}",
            "f1_score": f"{results.f1:.1%}",
            "tp": results.true_positives,
            "fp": results.false_positives,
            "tn": results.true_negatives,
            "fn": results.false_negatives,
        }, indent=2))
    else:
        print(results.summary())

    return 0 if results.accuracy == 1.0 else 1


def cmd_init(args: argparse.Namespace) -> int:
    from sentinel.init_config import run_init

    actions = run_init(
        hooks=not args.no_hooks,
        mcp=not args.no_mcp,
        policy=not args.no_policy,
    )

    if actions:
        print("Sentinel AI initialized:")
        for action in actions:
            print(f"  - {action}")
    else:
        print("Nothing to configure.")

    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    from sentinel.hooks import run_hook

    return run_hook()


def cmd_mcp_proxy(args: argparse.Namespace) -> int:
    from sentinel.mcp_proxy import run_proxy
    from sentinel.core import RiskLevel

    if not args.upstream_cmd:
        print("Error: provide upstream MCP server command after --", file=sys.stderr)
        print("Example: sentinel mcp-proxy -- npx @mcp/server-filesystem /tmp", file=sys.stderr)
        return 1

    threshold = RiskLevel[args.block_on.upper()] if args.block_on else RiskLevel.HIGH
    return run_proxy(args.upstream_cmd, block_threshold=threshold)


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
    parser.add_argument("--version", action="version", version="sentinel-ai 0.5.2")
    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan text for safety issues")
    scan_parser.add_argument("text", nargs="*", help="Text to scan")
    scan_parser.add_argument("--file", "-f", help="Read text from file")
    scan_parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    scan_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # red-team command
    rt_parser = subparsers.add_parser(
        "red-team", help="Test scanner robustness with adversarial variants"
    )
    rt_parser.add_argument("text", nargs="*", help="Attack payload to test")
    rt_parser.add_argument(
        "--file", "-f", help="File with one payload per line"
    )
    rt_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", help="Run the accuracy benchmark suite"
    )
    bench_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # init command
    init_parser = subparsers.add_parser(
        "init", help="Auto-configure Sentinel AI for Claude Code and MCP"
    )
    init_parser.add_argument(
        "--no-hooks", action="store_true", help="Skip Claude Code hooks setup"
    )
    init_parser.add_argument(
        "--no-mcp", action="store_true", help="Skip MCP server config"
    )
    init_parser.add_argument(
        "--no-policy", action="store_true", help="Skip policy.yaml creation"
    )

    # hook command (for Claude Code hooks integration)
    subparsers.add_parser(
        "hook", help="Run as a Claude Code PreToolUse hook (reads event from stdin)"
    )

    # mcp-proxy command
    proxy_parser = subparsers.add_parser(
        "mcp-proxy", help="Run as a safety proxy for any MCP server"
    )
    proxy_parser.add_argument(
        "--block-on",
        choices=["low", "medium", "high", "critical"],
        default="high",
        help="Minimum risk level to block (default: high)",
    )
    proxy_parser.add_argument(
        "upstream_cmd", nargs=argparse.REMAINDER,
        help="Upstream MCP server command (after --)",
    )

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", "-p", type=int, default=8329, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args(argv)

    if args.command == "scan":
        return cmd_scan(args)
    elif args.command == "red-team":
        return cmd_red_team(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "init":
        return cmd_init(args)
    elif args.command == "hook":
        return cmd_hook(args)
    elif args.command == "mcp-proxy":
        return cmd_mcp_proxy(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
