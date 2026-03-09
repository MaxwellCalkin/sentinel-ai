"""Prompt version control and diff tracking.

Track changes to prompts over time, compare versions, and
roll back to previous versions. Essential for prompt engineering
in production.

Usage:
    from sentinel.prompt_versioner import PromptVersioner

    v = PromptVersioner()
    v.commit("system_prompt", "You are a helpful assistant.")
    v.commit("system_prompt", "You are a helpful, harmless assistant.")
    print(v.history("system_prompt"))
    print(v.diff("system_prompt", 1, 2))
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptVersion:
    """A single version of a prompt."""
    prompt_id: str
    version: int
    content: str
    hash: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class PromptDiff:
    """Diff between two prompt versions."""
    prompt_id: str
    from_version: int
    to_version: int
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    changed: bool = False


class PromptVersioner:
    """Version control for prompts.

    Track prompt changes over time with commit messages,
    diff between versions, and rollback support.
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[PromptVersion]] = {}

    def commit(
        self,
        prompt_id: str,
        content: str,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Commit a new version of a prompt.

        Args:
            prompt_id: Identifier for the prompt.
            content: The prompt content.
            message: Optional commit message.
            metadata: Optional metadata.

        Returns:
            The new PromptVersion.
        """
        if prompt_id not in self._versions:
            self._versions[prompt_id] = []

        versions = self._versions[prompt_id]
        version_num = len(versions) + 1
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

        # Skip if content hasn't changed
        if versions and versions[-1].hash == content_hash:
            return versions[-1]

        pv = PromptVersion(
            prompt_id=prompt_id,
            version=version_num,
            content=content,
            hash=content_hash,
            message=message,
            metadata=metadata or {},
        )
        versions.append(pv)
        return pv

    def get(self, prompt_id: str, version: int | None = None) -> PromptVersion | None:
        """Get a specific version (latest if version=None)."""
        versions = self._versions.get(prompt_id, [])
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for v in versions:
            if v.version == version:
                return v
        return None

    def latest(self, prompt_id: str) -> str | None:
        """Get the latest content for a prompt."""
        v = self.get(prompt_id)
        return v.content if v else None

    def history(self, prompt_id: str) -> list[PromptVersion]:
        """Get full version history for a prompt."""
        return list(self._versions.get(prompt_id, []))

    def version_count(self, prompt_id: str) -> int:
        """Number of versions for a prompt."""
        return len(self._versions.get(prompt_id, []))

    @property
    def prompt_ids(self) -> list[str]:
        """All tracked prompt IDs."""
        return list(self._versions.keys())

    @property
    def prompt_count(self) -> int:
        """Number of tracked prompts."""
        return len(self._versions)

    def diff(self, prompt_id: str, from_ver: int, to_ver: int) -> PromptDiff:
        """Compute diff between two versions.

        Args:
            prompt_id: The prompt identifier.
            from_ver: Source version number.
            to_ver: Target version number.

        Returns:
            PromptDiff with added/removed lines.
        """
        v_from = self.get(prompt_id, from_ver)
        v_to = self.get(prompt_id, to_ver)

        if v_from is None or v_to is None:
            return PromptDiff(
                prompt_id=prompt_id,
                from_version=from_ver,
                to_version=to_ver,
                changed=False,
            )

        from_lines = set(v_from.content.splitlines())
        to_lines = set(v_to.content.splitlines())

        added = sorted(to_lines - from_lines)
        removed = sorted(from_lines - to_lines)

        return PromptDiff(
            prompt_id=prompt_id,
            from_version=from_ver,
            to_version=to_ver,
            added_lines=added,
            removed_lines=removed,
            changed=bool(added or removed),
        )

    def rollback(self, prompt_id: str, to_version: int) -> PromptVersion | None:
        """Rollback to a previous version (creates a new version with old content).

        Args:
            prompt_id: The prompt identifier.
            to_version: Version to roll back to.

        Returns:
            New PromptVersion with the rolled-back content, or None if not found.
        """
        target = self.get(prompt_id, to_version)
        if target is None:
            return None

        return self.commit(
            prompt_id,
            target.content,
            message=f"Rollback to version {to_version}",
        )

    def delete(self, prompt_id: str) -> bool:
        """Delete all versions of a prompt."""
        if prompt_id in self._versions:
            del self._versions[prompt_id]
            return True
        return False

    def export(self) -> dict[str, Any]:
        """Export all prompts and versions as a dict."""
        result: dict[str, Any] = {}
        for pid, versions in self._versions.items():
            result[pid] = [
                {
                    "version": v.version,
                    "content": v.content,
                    "hash": v.hash,
                    "timestamp": v.timestamp,
                    "message": v.message,
                    "metadata": v.metadata,
                }
                for v in versions
            ]
        return result
