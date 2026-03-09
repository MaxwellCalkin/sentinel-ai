"""Tests for sentinel.prompt_registry."""

from __future__ import annotations

import pytest

from sentinel.prompt_registry import PromptRegistry, RegistryEntry, RegistryStats


# -- TestRegister --


class TestRegister:
    def test_register_basic(self) -> None:
        registry = PromptRegistry()
        entry = registry.register("greeting", "Hello, {name}!", author="alice")

        assert entry.name == "greeting"
        assert entry.template == "Hello, {name}!"
        assert entry.version == 1
        assert entry.author == "alice"
        assert entry.tags == []
        assert entry.approved is False
        assert isinstance(entry.created_at, float)

    def test_register_with_tags(self) -> None:
        registry = PromptRegistry()
        entry = registry.register(
            "summarize", "Summarize: {text}", tags=["nlp", "summarization"]
        )

        assert entry.tags == ["nlp", "summarization"]

    def test_register_duplicate_creates_version(self) -> None:
        registry = PromptRegistry()
        first = registry.register("greeting", "Hello!")
        second = registry.register("greeting", "Hi there!")

        assert first.version == 1
        assert second.version == 2
        assert registry.get("greeting").version == 2


# -- TestGet --


class TestGet:
    def test_get_latest(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "v1")
        registry.register("prompt", "v2")

        latest = registry.get("prompt")

        assert latest.template == "v2"
        assert latest.version == 2

    def test_get_specific_version(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "v1")
        registry.register("prompt", "v2")

        first = registry.get("prompt", version=1)

        assert first.template == "v1"
        assert first.version == 1

    def test_get_missing_raises(self) -> None:
        registry = PromptRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")


# -- TestUpdate --


class TestUpdate:
    def test_update_increments_version(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "original", author="alice")
        updated = registry.update("prompt", "revised", author="bob")

        assert updated.version == 2
        assert updated.template == "revised"
        assert updated.author == "bob"

    def test_update_missing_raises(self) -> None:
        registry = PromptRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.update("nonexistent", "template")


# -- TestApproval --


class TestApproval:
    def test_approve(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "Hello")

        registry.approve("prompt")

        assert registry.get("prompt").approved is True

    def test_render_unapproved_blocked(self) -> None:
        registry = PromptRegistry(require_approval=True)
        registry.register("prompt", "Hello, {name}!")

        with pytest.raises(ValueError, match="not approved"):
            registry.render("prompt", name="World")


# -- TestSearch --


class TestSearch:
    def test_search_by_name(self) -> None:
        registry = PromptRegistry()
        registry.register("greeting_en", "Hello!")
        registry.register("farewell_en", "Goodbye!")

        results = registry.search("greeting")

        assert len(results) == 1
        assert results[0].name == "greeting_en"

    def test_search_by_content(self) -> None:
        registry = PromptRegistry()
        registry.register("a", "Summarize the document")
        registry.register("b", "Translate this text")

        results = registry.search("summarize")

        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_by_tag(self) -> None:
        registry = PromptRegistry()
        registry.register("a", "Hello", tags=["chat"])
        registry.register("b", "Summarize", tags=["nlp"])

        results = registry.search("chat")

        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_no_results(self) -> None:
        registry = PromptRegistry()
        registry.register("a", "Hello")

        results = registry.search("zzzzz_no_match")

        assert results == []


# -- TestRender --


class TestRender:
    def test_render_basic(self) -> None:
        registry = PromptRegistry()
        registry.register("greeting", "Hello, {name}! Welcome to {place}.")

        result = registry.render("greeting", name="Alice", place="Wonderland")

        assert result == "Hello, Alice! Welcome to Wonderland."

    def test_render_missing_vars_preserved(self) -> None:
        registry = PromptRegistry()
        registry.register("greeting", "Hello, {name}! You are in {place}.")

        result = registry.render("greeting", name="Alice")

        assert result == "Hello, Alice! You are in {place}."


# -- TestHistory --


class TestHistory:
    def test_history(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "v1")
        registry.update("prompt", "v2")
        registry.update("prompt", "v3")

        versions = registry.history("prompt")

        assert len(versions) == 3

    def test_history_order(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "first")
        registry.update("prompt", "second")

        versions = registry.history("prompt")

        assert versions[0].version == 1
        assert versions[0].template == "first"
        assert versions[1].version == 2
        assert versions[1].template == "second"


# -- TestDelete --


class TestDelete:
    def test_delete(self) -> None:
        registry = PromptRegistry()
        registry.register("prompt", "Hello")

        registry.delete("prompt")

        assert "prompt" not in registry.list_prompts()

    def test_delete_missing_raises(self) -> None:
        registry = PromptRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.delete("nonexistent")


# -- TestStats --


class TestStats:
    def test_stats(self) -> None:
        registry = PromptRegistry()
        registry.register("a", "Hello", author="alice")
        registry.register("b", "World", author="bob")
        registry.update("a", "Hi", author="alice")
        registry.approve("b")

        result = registry.stats()

        assert isinstance(result, RegistryStats)
        assert result.total_prompts == 2
        assert result.total_versions == 3
        assert result.approved_count == 1
        assert result.authors == ["alice", "bob"]


# -- TestExport --


class TestExport:
    def test_export_all(self) -> None:
        registry = PromptRegistry()
        registry.register("greeting", "Hello, {name}!", author="alice", tags=["chat"])

        exported = registry.export_all()

        assert "greeting" in exported
        assert len(exported["greeting"]) == 1
        entry_dict = exported["greeting"][0]
        assert entry_dict["name"] == "greeting"
        assert entry_dict["template"] == "Hello, {name}!"
        assert entry_dict["version"] == 1
        assert entry_dict["author"] == "alice"
        assert entry_dict["tags"] == ["chat"]
        assert entry_dict["approved"] is False
        assert "created_at" in entry_dict
