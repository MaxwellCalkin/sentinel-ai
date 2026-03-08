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


def _format_result(result: ScanResult, fmt: str = "text", artifact_uri: str = "input") -> str:
    if fmt == "sarif":
        from sentinel.sarif import scan_result_to_sarif, sarif_to_json
        return sarif_to_json(scan_result_to_sarif(result, artifact_uri=artifact_uri))

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


def cmd_code_scan(args: argparse.Namespace) -> int:
    from sentinel.scanners.code_scanner import CodeScanner

    scanner = CodeScanner()

    if args.stdin:
        code = sys.stdin.read()
        filename = args.filename or "stdin"
    elif args.file:
        code = Path(args.file).read_text(encoding="utf-8")
        filename = args.file
    else:
        print("Error: provide --file or --stdin", file=sys.stderr)
        return 1

    findings = scanner.scan(code, filename=filename)

    if args.format == "sarif":
        from sentinel.sarif import findings_to_sarif, sarif_to_json
        print(sarif_to_json(findings_to_sarif(findings, artifact_uri=filename)))
    elif args.format == "json":
        print(json.dumps([
            {
                "category": f.category,
                "description": f.description,
                "risk": f.risk.value,
                "line": f.metadata.get("line"),
                "match": f.metadata.get("match", ""),
            }
            for f in findings
        ], indent=2))
    else:
        if not findings:
            print(f"No vulnerabilities found in {filename}")
        else:
            print(f"Found {len(findings)} vulnerability(ies) in {filename}:\n")
            for f in findings:
                print(f"  [{f.risk.value.upper()}] {f.description}")
                print(f"    Category: {f.category}")
                if f.metadata.get("match"):
                    print(f"    Match: {f.metadata['match'][:80]}")
                print()

    return 1 if any(f.risk >= RiskLevel.HIGH for f in findings) else 0


def cmd_pre_commit(args: argparse.Namespace) -> int:
    """Scan git staged files for code vulnerabilities."""
    import subprocess

    from sentinel.scanners.code_scanner import CodeScanner
    from sentinel.core import RiskLevel

    scannable_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".php", ".java", ".go", ".rs"}

    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: not a git repository or git not found", file=sys.stderr)
        return 1

    staged_files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    if not staged_files:
        return 0

    code_files = [f for f in staged_files if Path(f).suffix in scannable_exts]
    if not code_files:
        return 0

    scanner = CodeScanner()
    threshold = RiskLevel[args.block_on.upper()]
    all_findings = []
    blocked = False

    for filepath in code_files:
        try:
            code = Path(filepath).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            continue

        findings = scanner.scan(code, filename=filepath)
        if not findings:
            continue

        all_findings.extend(findings)
        high_findings = [f for f in findings if f.risk >= threshold]
        if high_findings:
            blocked = True
            print(f"\n{filepath}:")
            for f in high_findings:
                line = f.metadata.get("line", "?")
                print(f"  L{line} [{f.risk.value.upper()}] {f.description}")

    if blocked:
        print(f"\nSentinel AI: blocked commit — {len(all_findings)} vulnerability(ies) found")
        print("Fix the issues above or use --no-verify to bypass (not recommended)")
        return 1

    if all_findings and not args.quiet:
        print(f"Sentinel AI: {len(all_findings)} low-risk finding(s) in staged files (allowed)")

    return 0


def cmd_mcp_proxy(args: argparse.Namespace) -> int:
    from sentinel.mcp_proxy import run_proxy
    from sentinel.core import RiskLevel

    if not args.upstream_cmd:
        print("Error: provide upstream MCP server command after --", file=sys.stderr)
        print("Example: sentinel mcp-proxy -- npx @mcp/server-filesystem /tmp", file=sys.stderr)
        return 1

    threshold = RiskLevel[args.block_on.upper()] if args.block_on else RiskLevel.HIGH
    return run_proxy(args.upstream_cmd, block_threshold=threshold)


def cmd_proxy(args: argparse.Namespace) -> int:
    """Run the LLM API firewall proxy."""
    try:
        import uvicorn
    except ImportError:
        print("Error: install sentinel-ai[api] for proxy support (pip install sentinel-guardrails[api])", file=sys.stderr)
        return 1

    from sentinel.proxy import create_proxy_app, ProxyConfig
    from sentinel.core import RiskLevel

    threshold = RiskLevel[args.block_on.upper()]
    config = ProxyConfig(
        target_url=args.target.rstrip("/"),
        port=args.port,
        scan_input=not args.no_input_scan,
        scan_output=not args.no_output_scan,
        redact_pii=not args.no_redact,
        block_threshold=threshold,
    )

    print(f"Sentinel AI Firewall starting on port {config.port}")
    print(f"  Target: {config.target_url}")
    print(f"  Input scanning: {'ON' if config.scan_input else 'OFF'}")
    print(f"  Output scanning: {'ON' if config.scan_output else 'OFF'}")
    print(f"  PII redaction: {'ON' if config.redact_pii else 'OFF'}")
    print(f"  Block threshold: {threshold.value}")
    print()
    print(f"Point your app at http://localhost:{config.port}")
    print(f"  export ANTHROPIC_BASE_URL=http://localhost:{config.port}")
    print()

    app = create_proxy_app(config)
    uvicorn.run(app, host="0.0.0.0", port=config.port)
    return 0


def cmd_claudemd_scan(args: argparse.Namespace) -> int:
    """Scan CLAUDE.md and project instruction files for injection vectors."""
    from sentinel.claudemd_scanner import scan_claudemd, scan_project_instructions

    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        reports = [scan_claudemd(content, file_path=args.file)]
    else:
        project_dir = Path(args.dir) if args.dir else None
        reports = scan_project_instructions(project_dir)

    if not reports:
        print("No instruction files found (CLAUDE.md, .cursorrules, etc.)")
        return 0

    if args.format == "json":
        print(json.dumps([{
            "file": r.file_path,
            "safe": r.safe,
            "risk": r.risk.value,
            "lines_scanned": r.lines_scanned,
            "findings": [{
                "category": f.category,
                "description": f.description,
                "risk": f.risk.value,
                "line": f.line,
                "match": f.match,
            } for f in r.findings],
        } for r in reports], indent=2))
    else:
        for r in reports:
            print(r.summary())
            print()

    has_critical = any(not r.safe for r in reports)
    return 1 if has_critical else 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit project security configuration."""
    from sentinel.audit import run_audit

    project_dir = Path(args.dir) if args.dir else None
    report = run_audit(project_dir)

    if args.format == "json":
        print(json.dumps({
            "score": report.score,
            "checks_passed": report.checks_passed,
            "checks_total": report.checks_total,
            "critical": report.critical_count,
            "warnings": report.warning_count,
            "findings": [
                {
                    "check": f.check,
                    "severity": f.severity.value,
                    "message": f.message,
                    "fix": f.fix,
                }
                for f in report.findings
            ],
        }, indent=2))
    else:
        print(report.summary())

    return 0 if report.critical_count == 0 else 1


def cmd_mcp_validate(args: argparse.Namespace) -> int:
    """Validate MCP tool schemas for injection vectors."""
    from sentinel.mcp_schema_validator import validate_mcp_tools

    if args.stdin:
        raw = sys.stdin.read()
    elif args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Error: provide --file or --stdin", file=sys.stderr)
        return 1

    import json as json_module
    try:
        data = json_module.loads(raw)
    except json_module.JSONDecodeError as e:
        print(f"Error: invalid JSON — {e}", file=sys.stderr)
        return 1

    # Accept either a list of tools or a dict with a "tools" key
    if isinstance(data, dict) and "tools" in data:
        tools = data["tools"]
    elif isinstance(data, list):
        tools = data
    else:
        print("Error: expected a JSON array of tools or an object with a 'tools' key", file=sys.stderr)
        return 1

    report = validate_mcp_tools(tools)

    if args.format == "json":
        print(json.dumps({
            "tools_scanned": report.tools_scanned,
            "safe": report.safe,
            "risk": report.risk.value,
            "critical": report.critical_count,
            "high": report.high_count,
            "findings": [
                {
                    "category": f.category,
                    "description": f.description,
                    "risk": f.risk.value,
                    "tool": f.metadata.get("tool", ""),
                    "field": f.metadata.get("field", ""),
                    "match": f.metadata.get("match", ""),
                }
                for f in report.findings
            ],
        }, indent=2))
    else:
        print(report.summary())

    return 1 if not report.safe else 0


def cmd_dep_scan(args: argparse.Namespace) -> int:
    """Scan dependency files for supply chain attack patterns."""
    from sentinel.scanners.dependency_scanner import DependencyScanner

    scanner = DependencyScanner()
    all_findings = []

    dep_filenames = [
        "requirements.txt", "requirements-dev.txt", "requirements-test.txt",
        "package.json", "pyproject.toml", "Pipfile", "setup.py",
    ]

    if args.file:
        files = [Path(args.file)]
    else:
        project_dir = Path(args.dir) if args.dir else Path.cwd()
        files = [project_dir / f for f in dep_filenames if (project_dir / f).exists()]

    if not files:
        print("No dependency files found.")
        return 0

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            continue

        findings = scanner.scan(content, filename=str(filepath))
        if findings:
            all_findings.extend([(filepath, f) for f in findings])

    if args.format == "json":
        print(json.dumps([
            {
                "file": str(fp),
                "category": f.category,
                "description": f.description,
                "risk": f.risk.value,
                "line": f.metadata.get("line"),
                "package": f.metadata.get("package", ""),
            }
            for fp, f in all_findings
        ], indent=2))
    else:
        if not all_findings:
            print("No supply chain risks found.")
        else:
            from sentinel.core import RiskLevel
            current_file = None
            for fp, f in all_findings:
                if fp != current_file:
                    print(f"\n{fp}:")
                    current_file = fp
                line = f.metadata.get("line", "?")
                print(f"  L{line} [{f.risk.value.upper()}] {f.description}")

            critical_count = sum(1 for _, f in all_findings if f.risk >= RiskLevel.CRITICAL)
            print(f"\n{len(all_findings)} finding(s), {critical_count} critical")

    has_critical = any(f.risk >= RiskLevel.CRITICAL for _, f in all_findings)
    return 1 if has_critical else 0


def cmd_secrets_scan(args: argparse.Namespace) -> int:
    """Scan source files for hardcoded secrets, API keys, and credentials."""
    from sentinel.scanners.secrets_scanner import SecretsScanner
    from sentinel.core import RiskLevel

    scanner = SecretsScanner()

    if args.file:
        findings = scanner.scan_file(Path(args.file))
    else:
        project_dir = Path(args.dir) if args.dir else Path.cwd()
        findings = scanner.scan_directory(
            project_dir,
            max_files=args.max_files,
            recursive=not args.no_recursive,
        )

    if args.format == "json":
        print(json.dumps([
            {
                "category": f.category,
                "description": f.description,
                "risk": f.risk.value,
                "line": f.metadata.get("line"),
                "file": f.metadata.get("filename", ""),
                "match": f.metadata.get("match", ""),
            }
            for f in findings
        ], indent=2))
    else:
        if not findings:
            print("No hardcoded secrets found.")
        else:
            current_file = None
            for f in findings:
                fname = f.metadata.get("filename", "unknown")
                if fname != current_file:
                    print(f"\n{fname}:")
                    current_file = fname
                line = f.metadata.get("line", "?")
                print(f"  L{line} [{f.risk.value.upper()}] {f.description.split(': ', 1)[-1]}")
                if f.metadata.get("match"):
                    print(f"       Match: {f.metadata['match']}")

            critical = sum(1 for f in findings if f.risk >= RiskLevel.CRITICAL)
            print(f"\n{len(findings)} secret(s) found, {critical} critical")

    has_critical = any(f.risk >= RiskLevel.CRITICAL for f in findings)
    return 1 if has_critical else 0


def cmd_project_scan(args: argparse.Namespace) -> int:
    """Run comprehensive security scan on entire project."""
    from sentinel.project_scanner import scan_project

    project_dir = Path(args.dir) if args.dir else None
    report = scan_project(project_dir)

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    return 1 if report.critical_count > 0 else 0


def cmd_badge(args: argparse.Namespace) -> int:
    """Generate a security score badge SVG."""
    from sentinel.badge import generate_badge_for_project

    project_dir = Path(args.dir) if args.dir else None
    output = Path(args.output) if args.output else None

    svg, score = generate_badge_for_project(project_dir, output=output)

    if output:
        print(f"Badge generated: {output} (score: {score}/100)")
    else:
        print(svg)

    return 0


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
    parser.add_argument("--version", action="version", version="sentinel-ai 0.10.0")
    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan text for safety issues")
    scan_parser.add_argument("text", nargs="*", help="Text to scan")
    scan_parser.add_argument("--file", "-f", help="Read text from file")
    scan_parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    scan_parser.add_argument(
        "--format", choices=["text", "json", "sarif"], default="text",
        help="Output format (sarif for GitHub Code Scanning)",
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

    # code-scan command
    cs_parser = subparsers.add_parser(
        "code-scan", help="Scan source code for OWASP vulnerabilities"
    )
    cs_parser.add_argument("--file", "-f", help="Source file to scan")
    cs_parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    cs_parser.add_argument("--filename", help="Filename hint for language detection")
    cs_parser.add_argument(
        "--format", choices=["text", "json", "sarif"], default="text",
        help="Output format (sarif for GitHub Code Scanning)",
    )

    # hook command (for Claude Code hooks integration)
    subparsers.add_parser(
        "hook", help="Run as a Claude Code PreToolUse hook (reads event from stdin)"
    )

    # pre-commit command
    pc_parser = subparsers.add_parser(
        "pre-commit", help="Scan git staged files for code vulnerabilities (use as git pre-commit hook)"
    )
    pc_parser.add_argument(
        "--block-on",
        choices=["low", "medium", "high", "critical"],
        default="high",
        help="Minimum risk level to block commit (default: high)",
    )
    pc_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress output for allowed findings",
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

    # mcp-validate command
    mv_parser = subparsers.add_parser(
        "mcp-validate", help="Validate MCP tool schemas for injection vectors"
    )
    mv_parser.add_argument("--file", "-f", help="JSON file containing MCP tool definitions")
    mv_parser.add_argument("--stdin", action="store_true", help="Read tool definitions from stdin")
    mv_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # proxy command (LLM API firewall)
    fw_parser = subparsers.add_parser(
        "proxy", help="Run as a transparent LLM API firewall (reverse proxy)"
    )
    fw_parser.add_argument(
        "--target", "-t", default="https://api.anthropic.com",
        help="Target LLM API URL (default: https://api.anthropic.com)",
    )
    fw_parser.add_argument(
        "--port", "-p", type=int, default=8330,
        help="Local proxy port (default: 8330)",
    )
    fw_parser.add_argument(
        "--no-input-scan", action="store_true",
        help="Disable input scanning",
    )
    fw_parser.add_argument(
        "--no-output-scan", action="store_true",
        help="Disable output scanning",
    )
    fw_parser.add_argument(
        "--no-redact", action="store_true",
        help="Disable PII redaction in responses",
    )
    fw_parser.add_argument(
        "--block-on",
        choices=["low", "medium", "high", "critical"],
        default="high",
        help="Minimum risk level to block (default: high)",
    )

    # audit command
    audit_parser = subparsers.add_parser(
        "audit", help="Audit project security configuration (hooks, permissions, policy)"
    )
    audit_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    audit_parser.add_argument(
        "--dir", "-d", help="Project directory to audit (default: current directory)"
    )

    # claudemd-scan command
    cm_parser = subparsers.add_parser(
        "claudemd-scan", help="Scan CLAUDE.md and project instruction files for injection vectors"
    )
    cm_parser.add_argument(
        "--file", "-f", help="Specific file to scan (default: auto-detect CLAUDE.md, .cursorrules, etc.)"
    )
    cm_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    cm_parser.add_argument(
        "--dir", "-d", help="Project directory (default: current directory)"
    )

    # dep-scan command
    dep_parser = subparsers.add_parser(
        "dep-scan", help="Scan dependency files for supply chain attacks (typosquatting, malicious packages)"
    )
    dep_parser.add_argument("--file", "-f", help="Dependency file to scan (requirements.txt, package.json, etc.)")
    dep_parser.add_argument("--dir", "-d", help="Project directory to auto-detect dependency files")
    dep_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # secrets-scan command
    ss_parser = subparsers.add_parser(
        "secrets-scan", help="Scan source files for hardcoded secrets, API keys, and credentials"
    )
    ss_parser.add_argument("--file", "-f", help="Single file to scan")
    ss_parser.add_argument("--dir", "-d", help="Project directory to scan (default: current directory)")
    ss_parser.add_argument("--max-files", type=int, default=100, help="Maximum files to scan (default: 100)")
    ss_parser.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories")
    ss_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # project-scan command
    ps_parser = subparsers.add_parser(
        "project-scan", help="Comprehensive security scan of entire project"
    )
    ps_parser.add_argument(
        "--dir", "-d", help="Project directory to scan (default: current directory)"
    )
    ps_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # badge command
    badge_parser = subparsers.add_parser(
        "badge", help="Generate a security score badge SVG for your README"
    )
    badge_parser.add_argument("--dir", "-d", help="Project directory (default: current directory)")
    badge_parser.add_argument("--output", "-o", help="Output file path (default: stdout)")

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
    elif args.command == "pre-commit":
        return cmd_pre_commit(args)
    elif args.command == "code-scan":
        return cmd_code_scan(args)
    elif args.command == "mcp-proxy":
        return cmd_mcp_proxy(args)
    elif args.command == "mcp-validate":
        return cmd_mcp_validate(args)
    elif args.command == "proxy":
        return cmd_proxy(args)
    elif args.command == "claudemd-scan":
        return cmd_claudemd_scan(args)
    elif args.command == "audit":
        return cmd_audit(args)
    elif args.command == "dep-scan":
        return cmd_dep_scan(args)
    elif args.command == "secrets-scan":
        return cmd_secrets_scan(args)
    elif args.command == "project-scan":
        return cmd_project_scan(args)
    elif args.command == "badge":
        return cmd_badge(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
