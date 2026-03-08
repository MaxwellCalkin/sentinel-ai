"""Tests for sentinel init auto-configuration."""

import json
import pytest
from pathlib import Path
from sentinel.init_config import run_init, init_claude_code, init_mcp, init_policy


@pytest.fixture
def tmp_project(tmp_path):
    return tmp_path


class TestInitClaudeCode:
    def test_creates_settings_file(self, tmp_project):
        actions = init_claude_code(tmp_project)
        settings_path = tmp_project / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        hooks = settings["hooks"]["PreToolUse"]
        assert any(
            any(h["command"] == "sentinel hook" for h in entry["hooks"])
            for entry in hooks
        )
        assert any("configured" in a for a in actions)

    def test_skips_if_already_configured(self, tmp_project):
        # First init
        init_claude_code(tmp_project)
        # Second init should skip
        actions = init_claude_code(tmp_project)
        assert any("skipped" in a for a in actions)

    def test_preserves_existing_settings(self, tmp_project):
        claude_dir = tmp_project / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(json.dumps({
            "permissions": {"allow": ["Read"]},
            "hooks": {
                "PostToolUse": [{"matcher": ".*", "hooks": []}],
            },
        }))

        init_claude_code(tmp_project)
        settings = json.loads(settings_path.read_text())
        # Existing settings preserved
        assert settings["permissions"]["allow"] == ["Read"]
        assert "PostToolUse" in settings["hooks"]
        # New hooks added
        assert "PreToolUse" in settings["hooks"]


class TestInitMCP:
    def test_creates_mcp_config(self, tmp_project):
        actions = init_mcp(tmp_project)
        assert len(actions) == 1
        # Either created new or found existing — either way sentinel-ai is configured
        assert any("configured" in a or "skipped" in a for a in actions)

    def test_skips_if_already_configured(self, tmp_project):
        # Create config in project dir so it's found first
        claude_dir = tmp_project / ".claude"
        claude_dir.mkdir()
        config_path = claude_dir / "claude_desktop_config.json"
        config_path.write_text(json.dumps({"mcpServers": {"sentinel-ai": {}}}))
        actions = init_mcp(tmp_project)
        assert any("skipped" in a for a in actions)


class TestInitPolicy:
    def test_creates_policy_yaml(self, tmp_project):
        actions = init_policy(tmp_project)
        policy_path = tmp_project / "sentinel-policy.yaml"
        assert policy_path.exists()
        content = policy_path.read_text()
        assert "block_threshold: high" in content
        assert "prompt_injection" in content

    def test_skips_if_exists(self, tmp_project):
        (tmp_project / "sentinel-policy.yaml").write_text("custom: true")
        actions = init_policy(tmp_project)
        assert any("skipped" in a for a in actions)
        # Original content preserved
        assert (tmp_project / "sentinel-policy.yaml").read_text() == "custom: true"


class TestRunInit:
    def test_full_init(self, tmp_project):
        actions = run_init(project_dir=tmp_project, pre_commit=False)
        assert len(actions) == 3
        assert (tmp_project / ".claude" / "settings.json").exists()
        assert (tmp_project / "sentinel-policy.yaml").exists()

    def test_skip_hooks(self, tmp_project):
        actions = run_init(project_dir=tmp_project, hooks=False, pre_commit=False)
        assert not (tmp_project / ".claude" / "settings.json").exists()
        assert len(actions) == 2

    def test_skip_all(self, tmp_project):
        actions = run_init(
            project_dir=tmp_project, hooks=False, mcp=False, policy=False, pre_commit=False
        )
        assert len(actions) == 0


class TestCLIInit:
    def test_cli_init_command(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from sentinel.cli import main
        code = main(["init"])
        assert code == 0
        out = capsys.readouterr().out
        assert "initialized" in out.lower()
