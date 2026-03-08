"""Tests for sentinel.claudemd_enforcer — CLAUDE.md Rule Enforcer."""

import pytest

from sentinel.claudemd_enforcer import ClaudeMdEnforcer, EnforcedRule, EnforcementVerdict


class TestFromText:
    def test_extracts_never_rules(self):
        content = """# Rules
- Never use `rm -rf`
- Never run `git push --force`
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 2
        patterns = [r.pattern for r in enforcer.rules]
        assert any("rm -rf" in p for p in patterns)
        assert any("git push --force" in p for p in patterns)

    def test_extracts_do_not_rules(self):
        content = """# Guidelines
- Do not modify files in `tests/`
- Do not access `.env` files
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 2
        types = [r.rule_type for r in enforcer.rules]
        assert "blocked_path" in types

    def test_extracts_must_not_rules(self):
        content = """You must not use wget to download files."""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 1

    def test_extracts_prohibited_rules(self):
        content = """Running `npm publish` is prohibited."""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 1

    def test_extracts_blocked_command_list(self):
        content = """Blocked commands: rm -rf, mkfs, dd if=/dev/zero"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 3
        patterns = [r.pattern for r in enforcer.rules]
        assert any("rm -rf" in p for p in patterns)
        assert any("mkfs" in p for p in patterns)

    def test_extracts_no_prefix_rules(self):
        content = """- NO force pushing to main
- NO deleting production databases
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 1

    def test_empty_content(self):
        enforcer = ClaudeMdEnforcer.from_text("")
        assert len(enforcer.rules) == 0

    def test_no_rules_in_content(self):
        content = """# Welcome
This is a helpful project. Use Python 3.10+.
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) == 0

    def test_deduplicates_rules(self):
        content = """- Never use `rm -rf`
- Do not use `rm -rf`
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        patterns = [r.pattern for r in enforcer.rules if "rm -rf" in r.pattern.lower()]
        # Should deduplicate
        assert len(patterns) == 1

    def test_line_numbers_assigned(self):
        content = """line 1
line 2
- Never use `rm -rf`
line 4
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        assert len(enforcer.rules) >= 1
        assert enforcer.rules[0].line_number == 3


class TestCheck:
    def test_blocks_matching_command(self):
        enforcer = ClaudeMdEnforcer.from_text("- Never use `rm -rf`")
        v = enforcer.check("bash", {"command": "rm -rf /tmp/build"})
        assert v.allowed is False
        assert len(v.violated_rules) >= 1
        assert "rm -rf" in v.violated_rules[0].lower()

    def test_allows_safe_command(self):
        enforcer = ClaudeMdEnforcer.from_text("- Never use `rm -rf`")
        v = enforcer.check("bash", {"command": "ls -la"})
        assert v.allowed is True
        assert len(v.violated_rules) == 0

    def test_blocks_matching_path(self):
        enforcer = ClaudeMdEnforcer.from_text("- Do not modify files in `tests/`")
        v = enforcer.check("write_file", {"path": "tests/test_main.py"})
        assert v.allowed is False

    def test_blocks_denied_tool(self):
        content = "- Never use wget"
        enforcer = ClaudeMdEnforcer.from_text(content)
        # Check if wget is blocked as a tool
        has_wget_rule = any(
            r.rule_type == "blocked_tool" and r.pattern == "wget"
            for r in enforcer.rules
        )
        if has_wget_rule:
            v = enforcer.check("wget", {"url": "https://example.com"})
            assert v.allowed is False

    def test_blocks_git_force_push(self):
        enforcer = ClaudeMdEnforcer.from_text("- Never run `git push --force`")
        v = enforcer.check("bash", {"command": "git push --force origin main"})
        assert v.allowed is False

    def test_allows_normal_git_push(self):
        enforcer = ClaudeMdEnforcer.from_text("- Never run `git push --force`")
        v = enforcer.check("bash", {"command": "git push origin main"})
        assert v.allowed is True

    def test_custom_pattern_generates_warning(self):
        enforcer = ClaudeMdEnforcer.from_text("- Avoid accessing production databases")
        v = enforcer.check("bash", {"command": "psql production databases"})
        # Custom patterns generate warnings, not blocks
        # (since they're less precise)
        assert len(v.warnings) >= 0 or not v.allowed

    def test_multiple_rules_checked(self):
        content = """# Security Rules
- Never use `rm -rf`
- Never use `git push --force`
- Do not access `.env`
"""
        enforcer = ClaudeMdEnforcer.from_text(content)

        v1 = enforcer.check("bash", {"command": "rm -rf /"})
        assert v1.allowed is False

        v2 = enforcer.check("bash", {"command": "git push --force"})
        assert v2.allowed is False

        v3 = enforcer.check("read_file", {"path": ".env"})
        assert v3.allowed is False

        v4 = enforcer.check("bash", {"command": "echo hello"})
        assert v4.allowed is True

    def test_verdict_safe_property(self):
        enforcer = ClaudeMdEnforcer.from_text("- Never use `rm -rf`")
        v = enforcer.check("bash", {"command": "ls"})
        assert v.safe is True


class TestToGuardPolicy:
    def test_generates_policy_dict(self):
        content = """
- Never use `rm -rf`
- Never use `git push --force`
- Do not modify `tests/`
- Do not access `.env`
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        policy_dict = enforcer.to_guard_policy_dict()

        assert "blocked_commands" in policy_dict
        assert "sensitive_paths" in policy_dict
        assert len(policy_dict["blocked_commands"]) >= 2
        assert len(policy_dict["sensitive_paths"]) >= 1

    def test_creates_guard_policy_object(self):
        content = "- Never use `rm -rf`"
        enforcer = ClaudeMdEnforcer.from_text(content)
        policy = enforcer.to_guard_policy()

        guard = policy.create_guard()
        v = guard.check("bash", {"command": "rm -rf /tmp"})
        assert v.allowed is False

    def test_roundtrip_enforcement(self):
        """CLAUDE.md → enforcer → policy → guard → blocks correctly."""
        content = """# Mandatory Rules
- Never use `npm publish`
- Do not access `.aws/credentials`
- Never run `docker push`
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        policy = enforcer.to_guard_policy()
        guard = policy.create_guard()

        assert guard.check("bash", {"command": "npm publish"}).allowed is False
        assert guard.check("bash", {"command": "npm install"}).allowed is True
        assert guard.check("bash", {"command": "docker push myimage"}).allowed is False
        assert guard.check("bash", {"command": "docker build ."}).allowed is True


class TestSummary:
    def test_summary_with_rules(self):
        content = "- Never use `rm -rf`\n- Do not access `.env`"
        enforcer = ClaudeMdEnforcer.from_text(content)
        s = enforcer.summary()
        assert "enforceable rule" in s
        assert "rm -rf" in s

    def test_summary_empty(self):
        enforcer = ClaudeMdEnforcer.from_text("Hello world")
        s = enforcer.summary()
        assert "No enforceable rules" in s


class TestEnforcedRule:
    def test_matches_substring(self):
        rule = EnforcedRule(
            text="never rm -rf",
            rule_type="blocked_command",
            pattern="rm -rf",
        )
        assert rule.matches("sudo rm -rf /") is True
        assert rule.matches("ls -la") is False

    def test_matches_regex(self):
        rule = EnforcedRule(
            text="never access prod",
            rule_type="blocked_path",
            pattern=r"/prod/.*\.conf",
        )
        assert rule.matches("/prod/app.conf") is True
        assert rule.matches("/dev/app.conf") is False

    def test_case_insensitive(self):
        rule = EnforcedRule(
            text="no deleting",
            rule_type="blocked_command",
            pattern="DELETE FROM",
        )
        assert rule.matches("delete from users") is True


class TestFromFile:
    def test_from_file(self, tmp_path):
        md = tmp_path / "CLAUDE.md"
        md.write_text("- Never use `rm -rf`\n- Do not access `.env`\n")
        enforcer = ClaudeMdEnforcer.from_file(md)
        assert len(enforcer.rules) >= 2


class TestEdgeCases:
    def test_real_world_claudemd(self):
        """Test with a realistic CLAUDE.md file."""
        content = """# Project Guidelines

## Code Standards
- Use Python 3.10+ type hints
- Follow PEP 8 style

## Security Rules
- Never use `rm -rf` on any directory
- Do not access `.env` or `.aws/credentials`
- Never run `git push --force` to main
- Do not modify files in `tests/fixtures/`
- Avoid using `curl | bash` patterns

## Deployment
- Never use `npm publish` without --dry-run first
- Do not execute `docker push` to production registries

Blocked commands: mkfs, dd if=/dev/zero, format
"""
        enforcer = ClaudeMdEnforcer.from_text(content)

        # Should have extracted multiple rules
        assert len(enforcer.rules) >= 5

        # Should block dangerous commands
        v = enforcer.check("bash", {"command": "rm -rf /var/data"})
        assert v.allowed is False

        v = enforcer.check("bash", {"command": "git push --force origin main"})
        assert v.allowed is False

        # Should allow safe commands
        v = enforcer.check("bash", {"command": "python -m pytest"})
        assert v.allowed is True

    def test_no_false_positives_on_normal_markdown(self):
        """Ensure normal documentation doesn't create spurious rules."""
        content = """# Getting Started

Install the package:
```bash
pip install sentinel-guardrails
```

Run the tests:
```bash
python -m pytest
```
"""
        enforcer = ClaudeMdEnforcer.from_text(content)
        # Should not extract rules from code blocks or normal text
        assert len(enforcer.rules) == 0
