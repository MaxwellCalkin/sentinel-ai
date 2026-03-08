"""Security configuration auditor for Claude Code projects.

Scans a project for security misconfigurations in Claude Code settings,
MCP servers, git hooks, and environment files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditFinding:
    check: str
    severity: Severity
    message: str
    fix: str | None = None


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0

    @property
    def score(self) -> int:
        if self.checks_total == 0:
            return 0
        return round(100 * self.checks_passed / self.checks_total)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    def summary(self) -> str:
        lines = [f"Sentinel AI Security Audit — Score: {self.score}/100"]
        lines.append(f"  Checks: {self.checks_passed}/{self.checks_total} passed")

        if self.critical_count:
            lines.append(f"  Critical issues: {self.critical_count}")
        if self.warning_count:
            lines.append(f"  Warnings: {self.warning_count}")

        if self.findings:
            lines.append("")
            for f in self.findings:
                icon = "!!" if f.severity == Severity.CRITICAL else "?" if f.severity == Severity.WARNING else "i"
                lines.append(f"  [{icon}] {f.check}: {f.message}")
                if f.fix:
                    lines.append(f"      Fix: {f.fix}")

        if self.score == 100:
            lines.append("\nAll checks passed!")
        elif self.critical_count:
            lines.append("\nCritical issues found — run 'sentinel init' to fix automatically.")

        return "\n".join(lines)


# --- Dangerous tool patterns ---

DANGEROUS_ALLOWLIST_PATTERNS = [
    "Bash(*)",
    "Bash",
    "Write(*)",
    "Edit(*)",
]

RISKY_ALLOWLIST_PATTERNS = [
    "WebFetch(*)",
    "WebSearch(*)",
    "mcp__*",
]


def audit_claude_code_hooks(project_dir: Path, report: AuditReport) -> None:
    """Check if Claude Code PreToolUse hooks are configured."""
    settings_path = project_dir / ".claude" / "settings.json"
    report.checks_total += 1

    if not settings_path.exists():
        report.findings.append(AuditFinding(
            check="claude-code-hooks",
            severity=Severity.CRITICAL,
            message="No .claude/settings.json found — no safety hooks configured",
            fix="Run 'sentinel init' to auto-configure hooks",
        ))
        return

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        report.findings.append(AuditFinding(
            check="claude-code-hooks",
            severity=Severity.CRITICAL,
            message=".claude/settings.json is malformed",
            fix="Check JSON syntax in .claude/settings.json",
        ))
        return

    hooks = settings.get("hooks", {})
    pre_tool = hooks.get("PreToolUse", [])

    has_sentinel_hook = False
    for entry in pre_tool:
        for h in entry.get("hooks", []):
            if "sentinel" in h.get("command", "").lower():
                has_sentinel_hook = True
                break

    if has_sentinel_hook:
        report.checks_passed += 1
    else:
        report.findings.append(AuditFinding(
            check="claude-code-hooks",
            severity=Severity.CRITICAL,
            message="No Sentinel AI PreToolUse hook configured — tool calls are unscanned",
            fix="Run 'sentinel init' or add sentinel hook to .claude/settings.json",
        ))


def audit_allowlist(project_dir: Path, report: AuditReport) -> None:
    """Check if permissions allowlist is overly permissive."""
    settings_path = project_dir / ".claude" / "settings.json"
    report.checks_total += 1

    if not settings_path.exists():
        report.checks_passed += 1  # No settings = no dangerous allowlist
        return

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        report.checks_passed += 1
        return

    permissions = settings.get("permissions", {})
    allow = permissions.get("allow", [])

    dangerous = [p for p in allow if p in DANGEROUS_ALLOWLIST_PATTERNS]
    if dangerous:
        report.findings.append(AuditFinding(
            check="permissions-allowlist",
            severity=Severity.WARNING,
            message=f"Overly permissive allowlist: {', '.join(dangerous)}",
            fix="Use specific tool patterns instead of wildcards (e.g., 'Bash(git *)' instead of 'Bash(*)')",
        ))
    else:
        report.checks_passed += 1


def audit_policy(project_dir: Path, report: AuditReport) -> None:
    """Check if sentinel-policy.yaml exists."""
    report.checks_total += 1
    policy_path = project_dir / "sentinel-policy.yaml"

    if policy_path.exists():
        content = policy_path.read_text(encoding="utf-8")
        if "block_threshold" in content:
            report.checks_passed += 1
        else:
            report.findings.append(AuditFinding(
                check="sentinel-policy",
                severity=Severity.WARNING,
                message="sentinel-policy.yaml exists but missing block_threshold",
                fix="Add 'block_threshold: high' to sentinel-policy.yaml",
            ))
    else:
        report.findings.append(AuditFinding(
            check="sentinel-policy",
            severity=Severity.WARNING,
            message="No sentinel-policy.yaml found — using default scan settings",
            fix="Run 'sentinel init' to create a policy file",
        ))


def audit_env_files(project_dir: Path, report: AuditReport) -> None:
    """Check for secrets in committed .env files."""
    report.checks_total += 1

    gitignore_path = project_dir / ".gitignore"
    has_env_ignore = False
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        has_env_ignore = ".env" in content

    env_files = list(project_dir.glob(".env*"))
    env_files = [f for f in env_files if f.is_file() and f.name != ".env.example"]

    if env_files and not has_env_ignore:
        report.findings.append(AuditFinding(
            check="env-files",
            severity=Severity.CRITICAL,
            message=f".env file(s) found but not in .gitignore: {', '.join(f.name for f in env_files)}",
            fix="Add '.env*' to .gitignore to prevent secret leakage",
        ))
    else:
        report.checks_passed += 1


def audit_pre_commit_hook(project_dir: Path, report: AuditReport) -> None:
    """Check if git pre-commit hook is configured."""
    report.checks_total += 1
    hook_path = project_dir / ".git" / "hooks" / "pre-commit"

    if not (project_dir / ".git").exists():
        report.checks_passed += 1  # Not a git repo, skip
        return

    if hook_path.exists():
        content = hook_path.read_text(encoding="utf-8")
        if "sentinel" in content.lower():
            report.checks_passed += 1
        else:
            report.findings.append(AuditFinding(
                check="pre-commit-hook",
                severity=Severity.INFO,
                message="Git pre-commit hook exists but doesn't use Sentinel AI",
                fix="Run 'sentinel init' to add code vulnerability scanning to pre-commit",
            ))
    else:
        report.findings.append(AuditFinding(
            check="pre-commit-hook",
            severity=Severity.WARNING,
            message="No git pre-commit hook — staged code isn't scanned for vulnerabilities",
            fix="Run 'sentinel init' to install the pre-commit hook",
        ))


def audit_mcp_config(project_dir: Path, report: AuditReport) -> None:
    """Check MCP server configuration for security."""
    report.checks_total += 1

    config_locations = [
        project_dir / ".claude" / "claude_desktop_config.json",
        Path.home() / ".claude" / "claude_desktop_config.json",
    ]

    config_path = None
    for loc in config_locations:
        if loc.exists():
            config_path = loc
            break

    if config_path is None:
        report.checks_passed += 1  # No MCP config = nothing to audit
        return

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        report.findings.append(AuditFinding(
            check="mcp-config",
            severity=Severity.WARNING,
            message="MCP config file is malformed",
        ))
        return

    servers = config.get("mcpServers", {})
    has_sentinel = "sentinel-ai" in servers

    if has_sentinel:
        report.checks_passed += 1
    elif servers:
        report.findings.append(AuditFinding(
            check="mcp-config",
            severity=Severity.WARNING,
            message=f"MCP servers configured ({', '.join(servers.keys())}) but no Sentinel AI safety proxy",
            fix="Run 'sentinel init' to add Sentinel AI as an MCP safety layer",
        ))
    else:
        report.checks_passed += 1  # No MCP servers at all


def run_audit(project_dir: Path | None = None) -> AuditReport:
    """Run all security audit checks."""
    if project_dir is None:
        project_dir = Path.cwd()

    report = AuditReport()

    audit_claude_code_hooks(project_dir, report)
    audit_allowlist(project_dir, report)
    audit_policy(project_dir, report)
    audit_env_files(project_dir, report)
    audit_pre_commit_hook(project_dir, report)
    audit_mcp_config(project_dir, report)

    return report
