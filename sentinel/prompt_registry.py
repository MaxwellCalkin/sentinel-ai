"""Centralized registry for managing, versioning, and auditing prompt templates.

Provides a single source of truth for all prompt templates used in LLM
applications, with version history, approval workflows, search, and
rendering capabilities.

Usage:
    from sentinel.prompt_registry import PromptRegistry, RegistryEntry, RegistryStats

    registry = PromptRegistry(require_approval=True)
    registry.register("greeting", "Hello, {name}!", author="alice", tags=["chat"])
    registry.approve("greeting")
    result = registry.render("greeting", name="World")
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegistryEntry:
    """A single versioned prompt template entry."""

    name: str
    template: str
    version: int
    author: str = ""
    tags: list[str] = field(default_factory=list)
    approved: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class RegistryStats:
    """Summary statistics for the prompt registry."""

    total_prompts: int = 0
    total_versions: int = 0
    approved_count: int = 0
    authors: list[str] = field(default_factory=list)


class PromptRegistry:
    """Centralized prompt template registry with versioning and approval.

    Manages prompt templates with automatic version tracking, optional
    approval workflows, search, and safe rendering with variable
    substitution.
    """

    def __init__(self, require_approval: bool = False) -> None:
        """Initialize the registry.

        Args:
            require_approval: If True, render() raises ValueError for
                unapproved prompt versions.
        """
        self._entries: dict[str, list[RegistryEntry]] = {}
        self._require_approval = require_approval

    def register(
        self,
        name: str,
        template: str,
        author: str = "",
        tags: list[str] | None = None,
    ) -> RegistryEntry:
        """Register a new prompt template or create a new version if it exists.

        Args:
            name: Unique identifier for the prompt.
            template: The prompt template string with {variable} placeholders.
            author: Who created this template.
            tags: Optional categorization tags.

        Returns:
            The newly created RegistryEntry.
        """
        if name in self._entries:
            return self._create_new_version(name, template, author, tags)
        return self._create_first_version(name, template, author, tags)

    def get(self, name: str, version: int | None = None) -> RegistryEntry:
        """Get a prompt entry by name, optionally at a specific version.

        Args:
            name: The prompt identifier.
            version: Specific version number, or None for latest.

        Returns:
            The matching RegistryEntry.

        Raises:
            KeyError: If name is not found or version does not exist.
        """
        self._require_exists(name)
        versions = self._entries[name]
        if version is None:
            return versions[-1]
        return self._find_version(name, versions, version)

    def update(self, name: str, template: str, author: str = "") -> RegistryEntry:
        """Create a new version of an existing prompt.

        Args:
            name: The prompt identifier.
            template: Updated template string.
            author: Who made this update.

        Returns:
            The newly created RegistryEntry.

        Raises:
            KeyError: If name is not found.
        """
        self._require_exists(name)
        previous = self._entries[name][-1]
        tags = list(previous.tags)
        return self._create_new_version(name, template, author, tags)

    def approve(self, name: str, version: int | None = None) -> None:
        """Mark a prompt version as approved.

        Args:
            name: The prompt identifier.
            version: Specific version to approve, or None for latest.

        Raises:
            KeyError: If name or version is not found.
        """
        entry = self.get(name, version)
        entry.approved = True

    def list_prompts(self) -> list[str]:
        """List all registered prompt names.

        Returns:
            Sorted list of prompt names.
        """
        return list(self._entries.keys())

    def history(self, name: str) -> list[RegistryEntry]:
        """Get the full version history for a prompt.

        Args:
            name: The prompt identifier.

        Returns:
            List of all RegistryEntry versions in chronological order.

        Raises:
            KeyError: If name is not found.
        """
        self._require_exists(name)
        return list(self._entries[name])

    def search(self, query: str) -> list[RegistryEntry]:
        """Search prompts by name, template content, or tags.

        Returns the latest version of each matching prompt. Matching is
        case-insensitive: name and template use substring matching, tags
        use exact match.

        Args:
            query: The search term.

        Returns:
            List of matching RegistryEntry objects (latest version each).
        """
        query_lower = query.lower()
        results: list[RegistryEntry] = []
        for versions in self._entries.values():
            latest = versions[-1]
            if self._entry_matches_query(latest, query_lower):
                results.append(latest)
        return results

    def delete(self, name: str) -> None:
        """Remove a prompt and all its versions.

        Args:
            name: The prompt identifier.

        Raises:
            KeyError: If name is not found.
        """
        self._require_exists(name)
        del self._entries[name]

    def render(self, name: str, /, **kwargs: Any) -> str:
        """Render a prompt template with variable substitution.

        Missing variables are left as literal {key} placeholders in the output.

        Args:
            name: The prompt identifier.
            **kwargs: Variable values to substitute.

        Returns:
            The rendered prompt string.

        Raises:
            KeyError: If name is not found.
            ValueError: If require_approval is True and prompt is not approved.
        """
        entry = self.get(name)
        self._enforce_approval(entry)
        return self._render_template(entry.template, kwargs)

    def export_all(self) -> dict[str, Any]:
        """Export the entire registry as a dictionary.

        Returns:
            Dict mapping prompt names to lists of version dicts.
        """
        result: dict[str, Any] = {}
        for name, versions in self._entries.items():
            result[name] = [
                {
                    "name": entry.name,
                    "template": entry.template,
                    "version": entry.version,
                    "author": entry.author,
                    "tags": entry.tags,
                    "approved": entry.approved,
                    "created_at": entry.created_at,
                }
                for entry in versions
            ]
        return result

    def stats(self) -> RegistryStats:
        """Compute registry statistics.

        Returns:
            RegistryStats with counts and author list.
        """
        total_versions = 0
        approved_count = 0
        authors_seen: set[str] = set()
        for versions in self._entries.values():
            total_versions += len(versions)
            for entry in versions:
                if entry.approved:
                    approved_count += 1
                if entry.author:
                    authors_seen.add(entry.author)
        return RegistryStats(
            total_prompts=len(self._entries),
            total_versions=total_versions,
            approved_count=approved_count,
            authors=sorted(authors_seen),
        )

    # -- Private helpers --

    def _create_first_version(
        self,
        name: str,
        template: str,
        author: str,
        tags: list[str] | None,
    ) -> RegistryEntry:
        entry = RegistryEntry(
            name=name,
            template=template,
            version=1,
            author=author,
            tags=tags or [],
        )
        self._entries[name] = [entry]
        return entry

    def _create_new_version(
        self,
        name: str,
        template: str,
        author: str,
        tags: list[str] | None,
    ) -> RegistryEntry:
        versions = self._entries[name]
        next_version = versions[-1].version + 1
        entry = RegistryEntry(
            name=name,
            template=template,
            version=next_version,
            author=author,
            tags=tags or [],
        )
        versions.append(entry)
        return entry

    def _require_exists(self, name: str) -> None:
        if name not in self._entries:
            raise KeyError(f"Prompt '{name}' not found in registry")

    def _find_version(
        self, name: str, versions: list[RegistryEntry], version: int
    ) -> RegistryEntry:
        for entry in versions:
            if entry.version == version:
                return entry
        raise KeyError(
            f"Version {version} not found for prompt '{name}'"
        )

    def _entry_matches_query(self, entry: RegistryEntry, query_lower: str) -> bool:
        if query_lower in entry.name.lower():
            return True
        if query_lower in entry.template.lower():
            return True
        for tag in entry.tags:
            if tag.lower() == query_lower:
                return True
        return False

    def _enforce_approval(self, entry: RegistryEntry) -> None:
        if self._require_approval and not entry.approved:
            raise ValueError(
                f"Prompt '{entry.name}' version {entry.version} is not approved"
            )

    def _render_template(self, template: str, variables: dict[str, Any]) -> str:
        """Render template, preserving unknown {key} placeholders."""
        return template.format_map(_MissingKeyDict(variables))


class _MissingKeyDict(dict):
    """Dict subclass that returns '{key}' for missing keys during format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
