"""Prompt evolution tracking with version diffs and trend analysis.

Track how prompts change across iterations, compute diffs between
versions, rollback to previous states, and analyze change trends.

Usage:
    from sentinel.prompt_history import PromptHistory

    history = PromptHistory()
    history.create("system", "You are a helpful assistant.", author="alice")
    history.update("system", "You are a helpful, harmless assistant.", author="bob")
    diff = history.diff("system", 1, 2)
    print(diff.additions, diff.deletions, diff.similarity)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    """A single versioned snapshot of a prompt."""

    version: int
    text: str
    author: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class PromptDiff:
    """Diff result between two prompt versions."""

    version_from: int
    version_to: int
    additions: list[str]
    deletions: list[str]
    similarity: float
    change_summary: str


@dataclass
class PromptTimeline:
    """Full timeline view of a prompt's evolution."""

    name: str
    versions: list[PromptVersion]
    total_versions: int
    latest_version: int
    avg_length: float


@dataclass
class HistoryStats:
    """Aggregate statistics across all tracked prompts."""

    total_prompts: int = 0
    total_versions: int = 0
    avg_versions_per_prompt: float = 0.0


class PromptHistory:
    """Track prompt evolution with version diffs and rollback.

    Maintains a versioned history for named prompts, supporting
    diff computation, rollback to any prior version, timeline
    views, and cumulative statistics.
    """

    def __init__(self) -> None:
        self._prompts: dict[str, list[PromptVersion]] = {}

    def create(
        self,
        name: str,
        text: str,
        author: str = "",
        metadata: dict | None = None,
    ) -> PromptVersion:
        """Create a new prompt with version 1.

        Args:
            name: Unique identifier for the prompt.
            text: The prompt content.
            author: Who created this prompt.
            metadata: Optional metadata dict.

        Returns:
            The newly created PromptVersion (version 1).

        Raises:
            KeyError: If a prompt with this name already exists.
        """
        if name in self._prompts:
            raise KeyError(f"Prompt '{name}' already exists")
        version = PromptVersion(
            version=1,
            text=text,
            author=author,
            metadata=metadata or {},
        )
        self._prompts[name] = [version]
        return version

    def update(
        self,
        name: str,
        text: str,
        author: str = "",
        metadata: dict | None = None,
    ) -> PromptVersion:
        """Create a new version of an existing prompt.

        Args:
            name: The prompt identifier.
            text: Updated prompt content.
            author: Who made this update.
            metadata: Optional metadata dict.

        Returns:
            The newly created PromptVersion.

        Raises:
            KeyError: If the prompt does not exist.
        """
        self._require_exists(name)
        versions = self._prompts[name]
        next_version_number = versions[-1].version + 1
        version = PromptVersion(
            version=next_version_number,
            text=text,
            author=author,
            metadata=metadata or {},
        )
        versions.append(version)
        return version

    def get(self, name: str, version: int | None = None) -> PromptVersion:
        """Get a specific version or the latest version of a prompt.

        Args:
            name: The prompt identifier.
            version: Specific version number, or None for latest.

        Returns:
            The matching PromptVersion.

        Raises:
            KeyError: If prompt not found or version does not exist.
        """
        self._require_exists(name)
        versions = self._prompts[name]
        if version is None:
            return versions[-1]
        return self._find_version(name, version)

    def diff(self, name: str, version_from: int, version_to: int) -> PromptDiff:
        """Compute a diff between two versions of a prompt.

        Additions are lines present in version_to but not in version_from.
        Deletions are lines present in version_from but not in version_to.
        Similarity is computed as Jaccard similarity on word sets.

        Args:
            name: The prompt identifier.
            version_from: Source version number.
            version_to: Target version number.

        Returns:
            PromptDiff with additions, deletions, similarity, and summary.

        Raises:
            KeyError: If prompt or either version is not found.
        """
        source = self.get(name, version_from)
        target = self.get(name, version_to)
        additions = _compute_additions(source.text, target.text)
        deletions = _compute_deletions(source.text, target.text)
        similarity = _jaccard_similarity(source.text, target.text)
        change_summary = _generate_change_summary(additions, deletions, similarity)
        return PromptDiff(
            version_from=version_from,
            version_to=version_to,
            additions=additions,
            deletions=deletions,
            similarity=similarity,
            change_summary=change_summary,
        )

    def rollback(self, name: str, version: int) -> PromptVersion:
        """Create a new version with the content from an old version.

        Args:
            name: The prompt identifier.
            version: The version number to rollback to.

        Returns:
            A new PromptVersion containing the rolled-back content.

        Raises:
            KeyError: If prompt or version is not found.
        """
        target = self.get(name, version)
        return self.update(
            name,
            target.text,
            author=target.author,
            metadata={"rollback_from_version": version},
        )

    def timeline(self, name: str) -> PromptTimeline:
        """Get the full timeline of a prompt's evolution.

        Args:
            name: The prompt identifier.

        Returns:
            PromptTimeline with all versions and summary stats.

        Raises:
            KeyError: If prompt is not found.
        """
        self._require_exists(name)
        versions = list(self._prompts[name])
        total = len(versions)
        latest = versions[-1].version
        avg_length = _average_length(versions)
        return PromptTimeline(
            name=name,
            versions=versions,
            total_versions=total,
            latest_version=latest,
            avg_length=avg_length,
        )

    def list_prompts(self) -> list[str]:
        """List all tracked prompt names.

        Returns:
            List of prompt names in insertion order.
        """
        return list(self._prompts.keys())

    def delete(self, name: str) -> None:
        """Delete a prompt and all its versions.

        Args:
            name: The prompt identifier.

        Raises:
            KeyError: If prompt is not found.
        """
        self._require_exists(name)
        del self._prompts[name]

    def stats(self) -> HistoryStats:
        """Compute cumulative statistics across all tracked prompts.

        Returns:
            HistoryStats with totals and averages.
        """
        total_prompts = len(self._prompts)
        total_versions = sum(len(v) for v in self._prompts.values())
        avg = total_versions / total_prompts if total_prompts > 0 else 0.0
        return HistoryStats(
            total_prompts=total_prompts,
            total_versions=total_versions,
            avg_versions_per_prompt=avg,
        )

    # -- Private helpers --

    def _require_exists(self, name: str) -> None:
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found")

    def _find_version(self, name: str, version: int) -> PromptVersion:
        for entry in self._prompts[name]:
            if entry.version == version:
                return entry
        raise KeyError(f"Version {version} not found for prompt '{name}'")


# -- Module-level helper functions --


def _compute_additions(source_text: str, target_text: str) -> list[str]:
    """Return lines in target that are not in source."""
    source_lines = set(source_text.splitlines())
    target_lines = target_text.splitlines()
    return [line for line in target_lines if line not in source_lines]


def _compute_deletions(source_text: str, target_text: str) -> list[str]:
    """Return lines in source that are not in target."""
    target_lines = set(target_text.splitlines())
    source_lines = source_text.splitlines()
    return [line for line in source_lines if line not in target_lines]


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity on word sets."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _generate_change_summary(
    additions: list[str],
    deletions: list[str],
    similarity: float,
) -> str:
    """Generate a human-readable summary of changes."""
    if not additions and not deletions:
        return "No changes"
    parts: list[str] = []
    if additions:
        parts.append(f"{len(additions)} line(s) added")
    if deletions:
        parts.append(f"{len(deletions)} line(s) removed")
    parts.append(f"similarity: {similarity:.0%}")
    return ", ".join(parts)


def _average_length(versions: list[PromptVersion]) -> float:
    """Compute the average text length across versions."""
    if not versions:
        return 0.0
    total = sum(len(v.text) for v in versions)
    return total / len(versions)
