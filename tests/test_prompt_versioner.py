"""Tests for prompt version control."""

import pytest
from sentinel.prompt_versioner import PromptVersioner, PromptVersion, PromptDiff


# ---------------------------------------------------------------------------
# Basic versioning
# ---------------------------------------------------------------------------

class TestCommit:
    def test_first_commit(self):
        v = PromptVersioner()
        pv = v.commit("sys", "You are helpful.")
        assert pv.version == 1
        assert pv.prompt_id == "sys"
        assert pv.content == "You are helpful."

    def test_multiple_commits(self):
        v = PromptVersioner()
        v.commit("sys", "V1")
        v.commit("sys", "V2")
        pv = v.commit("sys", "V3")
        assert pv.version == 3

    def test_skip_duplicate_content(self):
        v = PromptVersioner()
        v.commit("sys", "Same content")
        pv = v.commit("sys", "Same content")
        assert pv.version == 1  # Not incremented
        assert v.version_count("sys") == 1

    def test_commit_message(self):
        v = PromptVersioner()
        pv = v.commit("sys", "Content", message="Initial version")
        assert pv.message == "Initial version"

    def test_commit_metadata(self):
        v = PromptVersioner()
        pv = v.commit("sys", "Content", metadata={"author": "Alice"})
        assert pv.metadata["author"] == "Alice"

    def test_hash_generated(self):
        v = PromptVersioner()
        pv = v.commit("sys", "Hello")
        assert len(pv.hash) == 12


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_get_latest(self):
        v = PromptVersioner()
        v.commit("sys", "V1")
        v.commit("sys", "V2")
        pv = v.get("sys")
        assert pv is not None
        assert pv.content == "V2"

    def test_get_specific_version(self):
        v = PromptVersioner()
        v.commit("sys", "V1")
        v.commit("sys", "V2")
        pv = v.get("sys", version=1)
        assert pv is not None
        assert pv.content == "V1"

    def test_get_nonexistent(self):
        v = PromptVersioner()
        assert v.get("nope") is None
        assert v.get("nope", version=1) is None

    def test_latest_content(self):
        v = PromptVersioner()
        v.commit("sys", "Latest content")
        assert v.latest("sys") == "Latest content"

    def test_latest_nonexistent(self):
        v = PromptVersioner()
        assert v.latest("nope") is None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history(self):
        v = PromptVersioner()
        v.commit("sys", "V1")
        v.commit("sys", "V2")
        v.commit("sys", "V3")
        h = v.history("sys")
        assert len(h) == 3
        assert h[0].version == 1
        assert h[2].version == 3

    def test_history_empty(self):
        v = PromptVersioner()
        assert v.history("nope") == []

    def test_version_count(self):
        v = PromptVersioner()
        v.commit("sys", "V1")
        v.commit("sys", "V2")
        assert v.version_count("sys") == 2


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_added(self):
        v = PromptVersioner()
        v.commit("sys", "Line 1")
        v.commit("sys", "Line 1\nLine 2")
        d = v.diff("sys", 1, 2)
        assert d.changed
        assert "Line 2" in d.added_lines

    def test_diff_removed(self):
        v = PromptVersioner()
        v.commit("sys", "Line 1\nLine 2")
        v.commit("sys", "Line 1")
        d = v.diff("sys", 1, 2)
        assert d.changed
        assert "Line 2" in d.removed_lines

    def test_diff_no_change(self):
        v = PromptVersioner()
        v.commit("sys", "Same")
        # Force a second version by changing and reverting
        v._versions["sys"].append(PromptVersion(
            prompt_id="sys", version=2, content="Same",
            hash="different",  # Force different hash
        ))
        d = v.diff("sys", 1, 2)
        assert not d.changed

    def test_diff_nonexistent(self):
        v = PromptVersioner()
        d = v.diff("nope", 1, 2)
        assert not d.changed


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback(self):
        v = PromptVersioner()
        v.commit("sys", "V1 content")
        v.commit("sys", "V2 content")
        pv = v.rollback("sys", to_version=1)
        assert pv is not None
        assert pv.content == "V1 content"
        assert pv.version == 3
        assert v.latest("sys") == "V1 content"

    def test_rollback_nonexistent(self):
        v = PromptVersioner()
        assert v.rollback("nope", to_version=1) is None

    def test_rollback_message(self):
        v = PromptVersioner()
        v.commit("sys", "V1")
        v.commit("sys", "V2")
        pv = v.rollback("sys", to_version=1)
        assert "Rollback" in pv.message


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------

class TestManagement:
    def test_prompt_ids(self):
        v = PromptVersioner()
        v.commit("sys", "A")
        v.commit("user", "B")
        assert set(v.prompt_ids) == {"sys", "user"}

    def test_prompt_count(self):
        v = PromptVersioner()
        v.commit("a", "A")
        v.commit("b", "B")
        assert v.prompt_count == 2

    def test_delete(self):
        v = PromptVersioner()
        v.commit("sys", "A")
        assert v.delete("sys")
        assert v.prompt_count == 0
        assert v.get("sys") is None

    def test_delete_nonexistent(self):
        v = PromptVersioner()
        assert not v.delete("nope")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export(self):
        v = PromptVersioner()
        v.commit("sys", "Content", message="Init")
        data = v.export()
        assert "sys" in data
        assert len(data["sys"]) == 1
        assert data["sys"][0]["content"] == "Content"
        assert data["sys"][0]["message"] == "Init"
        assert data["sys"][0]["version"] == 1

    def test_export_empty(self):
        v = PromptVersioner()
        assert v.export() == {}
