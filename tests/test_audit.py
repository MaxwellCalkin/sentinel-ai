"""Tests for sentinel audit — Claude Code security configuration auditor."""

import json
import pytest
from pathlib import Path

from sentinel.audit import (
    run_audit,
    audit_claude_code_hooks,
    audit_allowlist,
    audit_policy,
    audit_env_files,
    audit_pre_commit_hook,
    audit_mcp_config,
    AuditReport,
    AuditFinding,
    Severity,
)


@pytest.fixture
def project(tmp_path):
    """Create a minimal project structure."""
    return tmp_path


class TestAuditReport:
    def test_empty_report_score(self):
        report = AuditReport()
        assert report.score == 0

    def test_perfect_score(self):
        report = AuditReport(checks_passed=6, checks_total=6)
        assert report.score == 100

    def test_partial_score(self):
        report = AuditReport(checks_passed=3, checks_total=6)
        assert report.score == 50

    def test_critical_count(self):
        report = AuditReport(findings=[
            AuditFinding(check="a", severity=Severity.CRITICAL, message="bad"),
            AuditFinding(check="b", severity=Severity.WARNING, message="meh"),
            AuditFinding(check="c", severity=Severity.CRITICAL, message="worse"),
        ])
        assert report.critical_count == 2
        assert report.warning_count == 1

    def test_summary_perfect(self):
        report = AuditReport(checks_passed=6, checks_total=6)
        text = report.summary()
        assert "100/100" in text
        assert "All checks passed" in text

    def test_summary_with_findings(self):
        report = AuditReport(
            checks_passed=4, checks_total=6,
            findings=[
                AuditFinding(
                    check="test", severity=Severity.CRITICAL,
                    message="Something bad", fix="Fix it",
                ),
            ],
        )
        text = report.summary()
        assert "Critical issues" in text
        assert "Fix it" in text


class TestAuditClaudeCodeHooks:
    def test_no_settings_file(self, project):
        report = AuditReport()
        audit_claude_code_hooks(project, report)
        assert report.checks_total == 1
        assert report.checks_passed == 0
        assert any(f.severity == Severity.CRITICAL for f in report.findings)

    def test_malformed_settings(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("not json!!!")
        report = AuditReport()
        audit_claude_code_hooks(project, report)
        assert report.checks_passed == 0
        assert any("malformed" in f.message for f in report.findings)

    def test_no_sentinel_hook(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "echo hi"}]}]},
        }))
        report = AuditReport()
        audit_claude_code_hooks(project, report)
        assert report.checks_passed == 0
        assert any("unscanned" in f.message for f in report.findings)

    def test_sentinel_hook_configured(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "sentinel hook"}]}]},
        }))
        report = AuditReport()
        audit_claude_code_hooks(project, report)
        assert report.checks_passed == 1
        assert len(report.findings) == 0


class TestAuditAllowlist:
    def test_no_settings(self, project):
        report = AuditReport()
        audit_allowlist(project, report)
        assert report.checks_passed == 1  # No settings = safe

    def test_safe_allowlist(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "permissions": {"allow": ["Read", "Bash(git status)"]},
        }))
        report = AuditReport()
        audit_allowlist(project, report)
        assert report.checks_passed == 1

    def test_dangerous_allowlist(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "permissions": {"allow": ["Bash(*)", "Read"]},
        }))
        report = AuditReport()
        audit_allowlist(project, report)
        assert report.checks_passed == 0
        assert any("Bash(*)" in f.message for f in report.findings)


class TestAuditPolicy:
    def test_no_policy(self, project):
        report = AuditReport()
        audit_policy(project, report)
        assert report.checks_passed == 0
        assert any("sentinel-policy" in f.check for f in report.findings)

    def test_valid_policy(self, project):
        (project / "sentinel-policy.yaml").write_text("block_threshold: high\nscanners:\n  - prompt_injection\n")
        report = AuditReport()
        audit_policy(project, report)
        assert report.checks_passed == 1

    def test_incomplete_policy(self, project):
        (project / "sentinel-policy.yaml").write_text("scanners:\n  - prompt_injection\n")
        report = AuditReport()
        audit_policy(project, report)
        assert report.checks_passed == 0
        assert any("block_threshold" in f.message for f in report.findings)


class TestAuditEnvFiles:
    def test_no_env_files(self, project):
        report = AuditReport()
        audit_env_files(project, report)
        assert report.checks_passed == 1

    def test_env_in_gitignore(self, project):
        (project / ".env").write_text("SECRET=123")
        (project / ".gitignore").write_text(".env*\n")
        report = AuditReport()
        audit_env_files(project, report)
        assert report.checks_passed == 1

    def test_env_not_in_gitignore(self, project):
        (project / ".env").write_text("SECRET=123")
        report = AuditReport()
        audit_env_files(project, report)
        assert report.checks_passed == 0
        assert any(f.severity == Severity.CRITICAL for f in report.findings)

    def test_env_example_ignored(self, project):
        (project / ".env.example").write_text("SECRET=changeme")
        report = AuditReport()
        audit_env_files(project, report)
        assert report.checks_passed == 1


class TestAuditPreCommitHook:
    def test_not_git_repo(self, project):
        report = AuditReport()
        audit_pre_commit_hook(project, report)
        assert report.checks_passed == 1  # Skip, not a git repo

    def test_no_hook(self, project):
        (project / ".git" / "hooks").mkdir(parents=True)
        report = AuditReport()
        audit_pre_commit_hook(project, report)
        assert report.checks_passed == 0

    def test_sentinel_hook(self, project):
        hooks_dir = project / ".git" / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "pre-commit").write_text("#!/bin/sh\nsentinel pre-commit\n")
        report = AuditReport()
        audit_pre_commit_hook(project, report)
        assert report.checks_passed == 1

    def test_other_hook(self, project):
        hooks_dir = project / ".git" / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "pre-commit").write_text("#!/bin/sh\necho 'custom'\n")
        report = AuditReport()
        audit_pre_commit_hook(project, report)
        assert report.checks_passed == 0
        assert any(f.severity == Severity.INFO for f in report.findings)


class TestAuditMCPConfig:
    def test_no_config(self, project):
        report = AuditReport()
        audit_mcp_config(project, report)
        assert report.checks_passed == 1

    def test_sentinel_configured(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "claude_desktop_config.json").write_text(json.dumps({
            "mcpServers": {"sentinel-ai": {"command": "sentinel", "args": ["mcp-serve"]}},
        }))
        report = AuditReport()
        audit_mcp_config(project, report)
        assert report.checks_passed == 1

    def test_servers_without_sentinel(self, project):
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "claude_desktop_config.json").write_text(json.dumps({
            "mcpServers": {"filesystem": {"command": "npx", "args": ["@mcp/server-filesystem"]}},
        }))
        report = AuditReport()
        audit_mcp_config(project, report)
        assert report.checks_passed == 0
        assert any("safety proxy" in f.message for f in report.findings)


class TestRunAudit:
    def test_fully_configured_project(self, project):
        # Set up all the things
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "sentinel hook"}]}]},
            "permissions": {"allow": ["Read"]},
        }))
        (project / "sentinel-policy.yaml").write_text("block_threshold: high\n")
        (project / ".gitignore").write_text(".env*\n")
        # Not a git repo — pre-commit skipped
        # No MCP config — mcp skipped

        report = run_audit(project)
        assert report.score == 100
        assert report.critical_count == 0

    def test_unconfigured_project(self, project):
        report = run_audit(project)
        assert report.score < 100
        assert report.critical_count > 0

    def test_checks_total_is_6(self, project):
        report = run_audit(project)
        assert report.checks_total == 6


class TestCLIAudit:
    def test_cli_audit_command(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Set up a configured project
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "sentinel hook"}]}]},
        }))
        (tmp_path / "sentinel-policy.yaml").write_text("block_threshold: high\n")

        from sentinel.cli import main
        code = main(["audit"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Audit" in out or "Score" in out
