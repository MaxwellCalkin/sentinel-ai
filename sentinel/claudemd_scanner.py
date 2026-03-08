"""CLAUDE.md security scanner.

Scans project instruction files (CLAUDE.md, .claude/settings.json) for
hidden injection vectors, suspicious configurations, and security risks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sentinel.core import RiskLevel


@dataclass
class ClaudeMdFinding:
    category: str
    description: str
    risk: RiskLevel
    line: int | None = None
    match: str | None = None


@dataclass
class ClaudeMdReport:
    file_path: str
    findings: list[ClaudeMdFinding] = field(default_factory=list)
    lines_scanned: int = 0

    @property
    def safe(self) -> bool:
        return not any(f.risk >= RiskLevel.HIGH for f in self.findings)

    @property
    def risk(self) -> RiskLevel:
        if not self.findings:
            return RiskLevel.NONE
        return max(f.risk for f in self.findings)

    def summary(self) -> str:
        lines = [f"CLAUDE.md Security Scan: {self.file_path}"]
        lines.append(f"  Lines scanned: {self.lines_scanned}")

        if not self.findings:
            lines.append("  No security issues found.")
            return "\n".join(lines)

        lines.append(f"  Issues found: {len(self.findings)}")
        lines.append(f"  Risk: {self.risk.value.upper()}")
        lines.append("")

        for f in self.findings:
            loc = f"L{f.line}" if f.line else "?"
            lines.append(f"  [{f.risk.value.upper()}] {loc}: {f.description}")
            if f.match:
                lines.append(f"    Match: {f.match[:100]}")

        return "\n".join(lines)


# Patterns to detect in CLAUDE.md files
_CLAUDEMD_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, str]] = [
    # Hidden instructions in HTML comments
    (
        "hidden_html_instruction",
        re.compile(
            r"<!--\s*(SYSTEM|ADMIN|INSTRUCTION|IMPORTANT|OVERRIDE|IGNORE|EXECUTE|PROMPT|"
            r"SECRET|HIDDEN|DO NOT|MUST NOT)[\s:].*?-->",
            re.IGNORECASE | re.DOTALL,
        ),
        RiskLevel.CRITICAL,
        "Hidden instruction in HTML comment — may inject commands during AI code review",
    ),
    # Authority impersonation
    (
        "authority_impersonation",
        re.compile(
            r"(ADMIN\s+MESSAGE\s+FROM\s+ANTHROPIC|"
            r"SYSTEM\s+(UPDATE|OVERRIDE|MESSAGE)\s*:|"
            r"ANTHROPIC\s+(ADMIN|SYSTEM|SECURITY)\s*(MESSAGE|NOTICE)\s*:|"
            r"IMPORTANT\s+SYSTEM\s+(MESSAGE|UPDATE)\s*:)",
            re.IGNORECASE,
        ),
        RiskLevel.CRITICAL,
        "Authority impersonation — fake system/admin message to manipulate AI behavior",
    ),
    # Base URL override (credential exfiltration)
    (
        "base_url_override",
        re.compile(
            r"(ANTHROPIC_BASE_URL|OPENAI_BASE_URL|API_BASE)\s*=\s*"
            r"https?://(?!api\.anthropic\.com|api\.openai\.com|localhost|127\.0\.0\.1)",
            re.IGNORECASE,
        ),
        RiskLevel.CRITICAL,
        "API base URL override — potential credential exfiltration vector",
    ),
    # Dangerous permission grants
    (
        "dangerous_permission",
        re.compile(
            r'["\']?(Bash\(\*\)|Write\(\*\)|Edit\(\*\))["\']?',
        ),
        RiskLevel.HIGH,
        "Overly permissive tool allowlist — grants unrestricted access",
    ),
    # Instruction to disable safety
    (
        "disable_safety",
        re.compile(
            r"(disable|turn\s+off|skip|bypass|ignore)\s+"
            r"(safety|security|scanning|hooks?|guardrails?|checks?|verification|validation)",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        "Instruction to disable safety mechanisms",
    ),
    # Exfiltration commands in instructions
    (
        "exfiltration_command",
        re.compile(
            r"(curl\s+(-X\s+POST\s+)?https?://(?!localhost)|"
            r"wget\s+.*-O|"
            r"nc\s+\S+\s+\d+|"
            r"scp\s+|rsync\s+.*@)",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        "Network command in instructions — potential data exfiltration",
    ),
    # Destructive commands embedded in instructions
    (
        "destructive_command",
        re.compile(
            r"(rm\s+(-rf?|--recursive)\s+[/~]|"
            r"mkfs\b|"
            r"dd\s+if=|"
            r"git\s+push\s+--force|"
            r"git\s+reset\s+--hard)",
        ),
        RiskLevel.HIGH,
        "Destructive command in instructions",
    ),
    # Zero-width characters (steganographic hiding)
    (
        "zero_width_chars",
        re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]"),
        RiskLevel.MEDIUM,
        "Zero-width characters detected — may hide instructions from human review",
    ),
    # Base64 encoded content
    (
        "base64_payload",
        re.compile(
            r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{40,}={0,2})(?![A-Za-z0-9+/=])"
        ),
        RiskLevel.MEDIUM,
        "Long base64-encoded content — may contain hidden instructions",
    ),
    # Unicode homoglyphs (visually similar but different characters)
    (
        "homoglyph_chars",
        re.compile(
            r"[\u0430\u0435\u043e\u0440\u0441\u0443\u0445"  # Cyrillic lookalikes
            r"\u0391\u0392\u0395\u0397\u0399\u039a\u039c\u039d\u039f"  # Greek lookalikes
            r"\u03a1\u03a4\u03a5\u03a7]"
        ),
        RiskLevel.MEDIUM,
        "Unicode homoglyph characters — may disguise malicious content as safe text",
    ),
    # Instruction to run arbitrary code
    (
        "arbitrary_code_execution",
        re.compile(
            r"(eval\s*\(|exec\s*\(|__import__\s*\(|"
            r"subprocess\.(run|call|Popen)|"
            r"os\.(system|popen|exec))",
            re.IGNORECASE,
        ),
        RiskLevel.MEDIUM,
        "Code execution pattern in instructions",
    ),
]


def scan_claudemd(content: str, file_path: str = "CLAUDE.md") -> ClaudeMdReport:
    """Scan a CLAUDE.md file for security issues."""
    report = ClaudeMdReport(file_path=file_path)
    lines = content.splitlines()
    report.lines_scanned = len(lines)

    for name, pattern, risk, description in _CLAUDEMD_PATTERNS:
        for match in pattern.finditer(content):
            # Calculate line number
            line_num = content[:match.start()].count("\n") + 1
            report.findings.append(ClaudeMdFinding(
                category=name,
                description=description,
                risk=risk,
                line=line_num,
                match=match.group(0)[:200],
            ))

    return report


def scan_project_instructions(project_dir: Path | None = None) -> list[ClaudeMdReport]:
    """Scan all project instruction files for security issues."""
    if project_dir is None:
        project_dir = Path.cwd()

    reports = []

    # Files to scan
    instruction_files = [
        project_dir / "CLAUDE.md",
        project_dir / ".claude" / "CLAUDE.md",
        project_dir / ".cursorrules",
        project_dir / ".github" / "copilot-instructions.md",
    ]

    for path in instruction_files:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                report = scan_claudemd(content, str(path))
                reports.append(report)
            except (OSError, UnicodeDecodeError):
                pass

    return reports
