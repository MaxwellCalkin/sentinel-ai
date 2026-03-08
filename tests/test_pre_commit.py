"""Tests for the pre-commit hook and CLI command."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sentinel.cli import cmd_pre_commit
from sentinel.init_config import init_pre_commit


class TestPreCommitCommand:
    def _make_args(self, block_on="high", quiet=False):
        args = MagicMock()
        args.block_on = block_on
        args.quiet = quiet
        return args

    @patch("subprocess.run")
    def test_no_staged_files(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        assert cmd_pre_commit(self._make_args()) == 0

    @patch("subprocess.run")
    def test_no_code_files(self, mock_run):
        mock_run.return_value = MagicMock(stdout="README.md\ndata.csv\n", returncode=0)
        assert cmd_pre_commit(self._make_args()) == 0

    @patch("subprocess.run")
    def test_clean_code_file(self, mock_run, tmp_path):
        clean_file = tmp_path / "clean.py"
        clean_file.write_text("x = 1 + 2\nprint(x)\n")

        mock_run.return_value = MagicMock(stdout=str(clean_file) + "\n", returncode=0)
        assert cmd_pre_commit(self._make_args()) == 0

    @patch("subprocess.run")
    def test_vulnerable_code_blocked(self, mock_run, tmp_path):
        vuln_file = tmp_path / "vuln.py"
        vuln_file.write_text('import os\nos.system(f"ping {user_input}")\n')

        mock_run.return_value = MagicMock(stdout=str(vuln_file) + "\n", returncode=0)
        assert cmd_pre_commit(self._make_args()) == 1

    @patch("subprocess.run")
    def test_git_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert cmd_pre_commit(self._make_args()) == 1

    @patch("subprocess.run")
    def test_git_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        assert cmd_pre_commit(self._make_args()) == 1

    @patch("subprocess.run")
    def test_quiet_suppresses_output(self, mock_run, tmp_path, capsys):
        clean_file = tmp_path / "clean.py"
        clean_file.write_text("x = 1\n")
        mock_run.return_value = MagicMock(stdout=str(clean_file) + "\n", returncode=0)
        cmd_pre_commit(self._make_args(quiet=True))
        captured = capsys.readouterr()
        assert captured.out == ""

    @patch("subprocess.run")
    def test_nonexistent_file_skipped(self, mock_run):
        mock_run.return_value = MagicMock(stdout="/no/such/file.py\n", returncode=0)
        assert cmd_pre_commit(self._make_args()) == 0


class TestInitPreCommit:
    def test_creates_hook(self, tmp_path):
        git_dir = tmp_path / ".git" / "hooks"
        git_dir.mkdir(parents=True)

        actions = init_pre_commit(tmp_path)
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        assert hook_path.exists()
        content = hook_path.read_text()
        assert "sentinel pre-commit" in content
        assert any("installed" in a for a in actions)

    def test_skips_non_git(self, tmp_path):
        actions = init_pre_commit(tmp_path)
        assert any("not a git repo" in a for a in actions)

    def test_skips_existing_sentinel_hook(self, tmp_path):
        git_dir = tmp_path / ".git" / "hooks"
        git_dir.mkdir(parents=True)
        hook_path = git_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\nsentinel pre-commit\n")

        actions = init_pre_commit(tmp_path)
        assert any("already configured" in a for a in actions)

    def test_skips_existing_other_hook(self, tmp_path):
        git_dir = tmp_path / ".git" / "hooks"
        git_dir.mkdir(parents=True)
        hook_path = git_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\necho 'custom hook'\n")

        actions = init_pre_commit(tmp_path)
        assert any("existing hook" in a for a in actions)
