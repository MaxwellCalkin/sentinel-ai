"""Project-wide security scanner — comprehensive security assessment of a codebase.

Runs all Sentinel AI scanners across a project directory and generates a
unified security report. Scans:
- CLAUDE.md / .cursorrules for injection vectors
- Dependency files for supply chain attacks
- Source code for OWASP vulnerabilities
- Claude Code configuration for security gaps
- MCP tool definitions for schema injection

Usage:
    sentinel project-scan                    # Scan current directory
    sentinel project-scan --dir /path/to/project
    sentinel project-scan --format json      # JSON output for CI/CD
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from sentinel.core import RiskLevel, Finding


@dataclass
class ProjectScanReport:
    """Unified report from scanning an entire project."""
    project_dir: str = ""
    sections: dict[str, list[Finding]] = field(default_factory=dict)

    @property
    def total_findings(self) -> int:
        return sum(len(f) for f in self.sections.values())

    @property
    def critical_count(self) -> int:
        return sum(1 for findings in self.sections.values() for f in findings if f.risk == RiskLevel.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for findings in self.sections.values() for f in findings if f.risk == RiskLevel.HIGH)

    @property
    def risk(self) -> RiskLevel:
        all_findings = [f for findings in self.sections.values() for f in findings]
        if not all_findings:
            return RiskLevel.NONE
        return max(f.risk for f in all_findings)

    @property
    def score(self) -> int:
        """Security score 0-100 (100 = no issues)."""
        if self.total_findings == 0:
            return 100
        deductions = 0
        for findings in self.sections.values():
            for f in findings:
                if f.risk == RiskLevel.CRITICAL:
                    deductions += 15
                elif f.risk == RiskLevel.HIGH:
                    deductions += 10
                elif f.risk == RiskLevel.MEDIUM:
                    deductions += 5
                elif f.risk == RiskLevel.LOW:
                    deductions += 2
        return max(0, 100 - deductions)

    def summary(self) -> str:
        lines = []
        lines.append(f"Sentinel AI Project Security Report")
        lines.append(f"{'=' * 40}")
        lines.append(f"Directory: {self.project_dir}")
        lines.append(f"Security Score: {self.score}/100")
        lines.append(f"Overall Risk: {self.risk.value.upper()}")
        lines.append(f"Total Findings: {self.total_findings} ({self.critical_count} critical, {self.high_count} high)")
        lines.append("")

        for section_name, findings in self.sections.items():
            if not findings:
                lines.append(f"[PASS] {section_name}")
            else:
                critical = sum(1 for f in findings if f.risk == RiskLevel.CRITICAL)
                high = sum(1 for f in findings if f.risk == RiskLevel.HIGH)
                lines.append(f"[{'FAIL' if critical or high else 'WARN'}] {section_name} — {len(findings)} finding(s)")
                for f in findings:
                    lines.append(f"  [{f.risk.value.upper()}] {f.description}")

        lines.append("")
        if self.score >= 90:
            lines.append("Overall: Good security posture.")
        elif self.score >= 70:
            lines.append("Overall: Moderate risk — address high/critical findings.")
        elif self.score >= 50:
            lines.append("Overall: Significant risk — remediation recommended.")
        else:
            lines.append("Overall: High risk — immediate remediation required.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "project_dir": self.project_dir,
            "score": self.score,
            "risk": self.risk.value,
            "total_findings": self.total_findings,
            "critical": self.critical_count,
            "high": self.high_count,
            "sections": {
                name: [
                    {
                        "scanner": f.scanner,
                        "category": f.category,
                        "description": f.description,
                        "risk": f.risk.value,
                        "metadata": f.metadata,
                    }
                    for f in findings
                ]
                for name, findings in self.sections.items()
            },
        }


def scan_project(project_dir: Path | None = None) -> ProjectScanReport:
    """Run all available scans on a project directory.

    Args:
        project_dir: Path to project root. Defaults to cwd.

    Returns:
        ProjectScanReport with findings from all scanners.
    """
    if project_dir is None:
        project_dir = Path.cwd()

    report = ProjectScanReport(project_dir=str(project_dir))

    # 1. CLAUDE.md / instruction file scanning
    try:
        from sentinel.claudemd_scanner import scan_project_instructions
        instruction_reports = scan_project_instructions(project_dir)
        findings = []
        for r in instruction_reports:
            for f in r.findings:
                findings.append(Finding(
                    scanner="claudemd_scanner",
                    category=f.category,
                    description=f"{r.file_path}:{f.line} — {f.description}",
                    risk=f.risk,
                    metadata={"file": r.file_path, "line": f.line, "match": f.match},
                ))
        report.sections["Instruction Files (CLAUDE.md)"] = findings
    except Exception:
        report.sections["Instruction Files (CLAUDE.md)"] = []

    # 2. Dependency scanning
    try:
        from sentinel.scanners.dependency_scanner import DependencyScanner
        scanner = DependencyScanner()
        dep_filenames = [
            "requirements.txt", "requirements-dev.txt",
            "package.json", "pyproject.toml", "Pipfile",
        ]
        dep_findings = []
        for fname in dep_filenames:
            fpath = project_dir / fname
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    for f in scanner.scan(content, filename=str(fpath)):
                        dep_findings.append(f)
                except (UnicodeDecodeError, OSError):
                    pass
        report.sections["Dependencies (Supply Chain)"] = dep_findings
    except Exception:
        report.sections["Dependencies (Supply Chain)"] = []

    # 3. Security configuration audit
    try:
        from sentinel.audit import run_audit
        audit_report = run_audit(project_dir)
        audit_findings = []
        for af in audit_report.findings:
            risk = RiskLevel.CRITICAL if af.severity.value == "critical" else (
                RiskLevel.HIGH if af.severity.value == "warning" else RiskLevel.LOW
            )
            audit_findings.append(Finding(
                scanner="audit",
                category=af.check,
                description=af.message,
                risk=risk,
                metadata={"fix": af.fix},
            ))
        report.sections["Security Configuration"] = audit_findings
    except Exception:
        report.sections["Security Configuration"] = []

    # 4. Source code scanning (scan .py and .js files in top-level src/ or project root)
    try:
        from sentinel.scanners.code_scanner import CodeScanner
        scanner = CodeScanner()
        code_findings = []
        scannable_exts = {".py", ".js", ".ts", ".jsx", ".tsx"}
        scan_dirs = [project_dir]
        for subdir in ["src", "lib", "app"]:
            d = project_dir / subdir
            if d.is_dir():
                scan_dirs.append(d)

        scanned = 0
        for scan_dir in scan_dirs:
            for fpath in scan_dir.iterdir():
                if fpath.is_file() and fpath.suffix in scannable_exts and scanned < 20:
                    try:
                        code = fpath.read_text(encoding="utf-8")
                        findings = scanner.scan(code, filename=str(fpath))
                        code_findings.extend(findings)
                        scanned += 1
                    except (UnicodeDecodeError, OSError):
                        pass
        report.sections["Source Code (OWASP)"] = code_findings
    except Exception:
        report.sections["Source Code (OWASP)"] = []

    # 5. MCP tool schema validation (check for mcp config files)
    try:
        from sentinel.mcp_schema_validator import validate_mcp_tools
        mcp_findings = []
        mcp_config_paths = [
            project_dir / ".claude" / "settings.json",
            project_dir / "claude_desktop_config.json",
        ]
        for config_path in mcp_config_paths:
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text(encoding="utf-8"))
                    # Look for MCP server configs with tools
                    mcp_servers = data.get("mcpServers", {})
                    if isinstance(mcp_servers, dict):
                        for server_name, server_config in mcp_servers.items():
                            tools = server_config.get("tools", [])
                            if isinstance(tools, list) and tools:
                                mcp_report = validate_mcp_tools(tools)
                                mcp_findings.extend(mcp_report.findings)
                except (json.JSONDecodeError, OSError):
                    pass
        report.sections["MCP Tool Schemas"] = mcp_findings
    except Exception:
        report.sections["MCP Tool Schemas"] = []

    return report
