"""Tests for prompt history tracking module."""

import pytest
from sentinel.prompt_history import (
    PromptHistory,
    PromptVersion,
    PromptDiff,
    PromptTimeline,
    HistoryStats,
)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_returns_version_one(self):
        history = PromptHistory()
        version = history.create("greeting", "Hello!")
        assert version.version == 1
        assert version.text == "Hello!"

    def test_create_stores_author(self):
        history = PromptHistory()
        version = history.create("greeting", "Hello!", author="alice")
        assert version.author == "alice"

    def test_create_stores_metadata(self):
        history = PromptHistory()
        version = history.create("greeting", "Hello!", metadata={"env": "prod"})
        assert version.metadata == {"env": "prod"}

    def test_create_sets_timestamp(self):
        history = PromptHistory()
        version = history.create("greeting", "Hello!")
        assert version.created_at > 0

    def test_create_duplicate_name_raises_key_error(self):
        history = PromptHistory()
        history.create("greeting", "Hello!")
        with pytest.raises(KeyError, match="already exists"):
            history.create("greeting", "Hi!")


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_creates_new_version(self):
        history = PromptHistory()
        history.create("sys", "V1")
        version = history.update("sys", "V2")
        assert version.version == 2
        assert version.text == "V2"

    def test_update_increments_version_number(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        version = history.update("sys", "V3")
        assert version.version == 3

    def test_update_unknown_prompt_raises_key_error(self):
        history = PromptHistory()
        with pytest.raises(KeyError, match="not found"):
            history.update("nonexistent", "text")

    def test_update_stores_author_and_metadata(self):
        history = PromptHistory()
        history.create("sys", "V1")
        version = history.update("sys", "V2", author="bob", metadata={"reason": "fix"})
        assert version.author == "bob"
        assert version.metadata == {"reason": "fix"}


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_latest_version(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        latest = history.get("sys")
        assert latest.version == 2
        assert latest.text == "V2"

    def test_get_specific_version(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        first = history.get("sys", version=1)
        assert first.text == "V1"

    def test_get_unknown_prompt_raises_key_error(self):
        history = PromptHistory()
        with pytest.raises(KeyError, match="not found"):
            history.get("nonexistent")

    def test_get_unknown_version_raises_key_error(self):
        history = PromptHistory()
        history.create("sys", "V1")
        with pytest.raises(KeyError, match="Version 99"):
            history.get("sys", version=99)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_additions(self):
        history = PromptHistory()
        history.create("sys", "Line 1")
        history.update("sys", "Line 1\nLine 2")
        result = history.diff("sys", 1, 2)
        assert "Line 2" in result.additions
        assert result.deletions == []

    def test_diff_deletions(self):
        history = PromptHistory()
        history.create("sys", "Line 1\nLine 2")
        history.update("sys", "Line 1")
        result = history.diff("sys", 1, 2)
        assert "Line 2" in result.deletions
        assert result.additions == []

    def test_diff_both_additions_and_deletions(self):
        history = PromptHistory()
        history.create("sys", "alpha\nbeta")
        history.update("sys", "alpha\ngamma")
        result = history.diff("sys", 1, 2)
        assert "gamma" in result.additions
        assert "beta" in result.deletions

    def test_diff_similarity_identical(self):
        history = PromptHistory()
        history.create("sys", "exact same text")
        history.update("sys", "exact same text")
        result = history.diff("sys", 1, 2)
        assert result.similarity == 1.0

    def test_diff_similarity_completely_different(self):
        history = PromptHistory()
        history.create("sys", "alpha beta gamma")
        history.update("sys", "delta epsilon zeta")
        result = history.diff("sys", 1, 2)
        assert result.similarity == 0.0

    def test_diff_similarity_partial_overlap(self):
        history = PromptHistory()
        history.create("sys", "the quick brown fox")
        history.update("sys", "the slow brown dog")
        result = history.diff("sys", 1, 2)
        assert 0.0 < result.similarity < 1.0

    def test_diff_change_summary_no_changes(self):
        history = PromptHistory()
        history.create("sys", "same")
        history.update("sys", "same")
        result = history.diff("sys", 1, 2)
        assert result.change_summary == "No changes"

    def test_diff_change_summary_with_changes(self):
        history = PromptHistory()
        history.create("sys", "old line")
        history.update("sys", "new line")
        result = history.diff("sys", 1, 2)
        assert "added" in result.change_summary
        assert "removed" in result.change_summary
        assert "similarity" in result.change_summary

    def test_diff_empty_texts(self):
        history = PromptHistory()
        history.create("sys", "")
        history.update("sys", "")
        result = history.diff("sys", 1, 2)
        assert result.similarity == 1.0
        assert result.additions == []
        assert result.deletions == []


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_creates_new_version_with_old_content(self):
        history = PromptHistory()
        history.create("sys", "V1 content")
        history.update("sys", "V2 content")
        rolled_back = history.rollback("sys", version=1)
        assert rolled_back.text == "V1 content"
        assert rolled_back.version == 3

    def test_rollback_latest_reflects_rolled_back_content(self):
        history = PromptHistory()
        history.create("sys", "V1 content")
        history.update("sys", "V2 content")
        history.rollback("sys", version=1)
        latest = history.get("sys")
        assert latest.text == "V1 content"

    def test_rollback_stores_metadata(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        rolled_back = history.rollback("sys", version=1)
        assert rolled_back.metadata["rollback_from_version"] == 1

    def test_rollback_unknown_version_raises_key_error(self):
        history = PromptHistory()
        history.create("sys", "V1")
        with pytest.raises(KeyError):
            history.rollback("sys", version=99)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


class TestTimeline:
    def test_timeline_contains_all_versions(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        history.update("sys", "V3")
        tl = history.timeline("sys")
        assert tl.total_versions == 3
        assert len(tl.versions) == 3

    def test_timeline_latest_version(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        tl = history.timeline("sys")
        assert tl.latest_version == 2

    def test_timeline_average_length(self):
        history = PromptHistory()
        history.create("sys", "ab")       # length 2
        history.update("sys", "abcdef")   # length 6
        tl = history.timeline("sys")
        assert tl.avg_length == 4.0

    def test_timeline_name(self):
        history = PromptHistory()
        history.create("sys", "V1")
        tl = history.timeline("sys")
        assert tl.name == "sys"

    def test_timeline_unknown_raises_key_error(self):
        history = PromptHistory()
        with pytest.raises(KeyError):
            history.timeline("nonexistent")


# ---------------------------------------------------------------------------
# List prompts
# ---------------------------------------------------------------------------


class TestListPrompts:
    def test_list_prompts_empty(self):
        history = PromptHistory()
        assert history.list_prompts() == []

    def test_list_prompts_multiple(self):
        history = PromptHistory()
        history.create("alpha", "A")
        history.create("beta", "B")
        history.create("gamma", "C")
        names = history.list_prompts()
        assert set(names) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_prompt(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.delete("sys")
        assert "sys" not in history.list_prompts()

    def test_delete_unknown_raises_key_error(self):
        history = PromptHistory()
        with pytest.raises(KeyError, match="not found"):
            history.delete("nonexistent")

    def test_delete_then_get_raises_key_error(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.delete("sys")
        with pytest.raises(KeyError):
            history.get("sys")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_empty(self):
        history = PromptHistory()
        s = history.stats()
        assert s.total_prompts == 0
        assert s.total_versions == 0
        assert s.avg_versions_per_prompt == 0.0

    def test_stats_single_prompt(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        s = history.stats()
        assert s.total_prompts == 1
        assert s.total_versions == 2
        assert s.avg_versions_per_prompt == 2.0

    def test_stats_multiple_prompts(self):
        history = PromptHistory()
        history.create("a", "A")
        history.update("a", "A2")
        history.update("a", "A3")
        history.create("b", "B")
        s = history.stats()
        assert s.total_prompts == 2
        assert s.total_versions == 4
        assert s.avg_versions_per_prompt == 2.0


# ---------------------------------------------------------------------------
# Version numbering
# ---------------------------------------------------------------------------


class TestVersionNumbering:
    def test_versions_are_sequential(self):
        history = PromptHistory()
        v1 = history.create("sys", "First")
        v2 = history.update("sys", "Second")
        v3 = history.update("sys", "Third")
        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

    def test_rollback_continues_sequence(self):
        history = PromptHistory()
        history.create("sys", "V1")
        history.update("sys", "V2")
        history.update("sys", "V3")
        rolled = history.rollback("sys", version=1)
        assert rolled.version == 4


# ---------------------------------------------------------------------------
# Multiple prompts tracked independently
# ---------------------------------------------------------------------------


class TestMultiplePrompts:
    def test_independent_version_numbers(self):
        history = PromptHistory()
        history.create("sys", "SysV1")
        history.create("user", "UserV1")
        history.update("sys", "SysV2")
        assert history.get("sys").version == 2
        assert history.get("user").version == 1

    def test_delete_one_does_not_affect_other(self):
        history = PromptHistory()
        history.create("sys", "SysV1")
        history.create("user", "UserV1")
        history.delete("sys")
        assert history.get("user").text == "UserV1"
        assert history.list_prompts() == ["user"]


# ---------------------------------------------------------------------------
# Dataclass field defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_prompt_version_default_author(self):
        version = PromptVersion(version=1, text="test")
        assert version.author == ""

    def test_prompt_version_default_metadata(self):
        version = PromptVersion(version=1, text="test")
        assert version.metadata == {}

    def test_history_stats_defaults(self):
        stats = HistoryStats()
        assert stats.total_prompts == 0
        assert stats.total_versions == 0
        assert stats.avg_versions_per_prompt == 0.0
