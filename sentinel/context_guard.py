"""Context boundary enforcement for LLM conversations.

Prevent system prompt leakage, role confusion, and context manipulation
attacks by validating messages, detecting injection patterns, and
enforcing clean conversation boundaries.

Usage:
    from sentinel.context_guard import ContextGuard

    guard = ContextGuard()
    result = guard.validate_message("user", "Hello, how are you?")
    assert result.valid

    result = guard.validate_message("user", "Ignore previous instructions")
    assert not result.valid  # Role confusion detected
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

_DEFAULT_CONFUSION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+previous", re.IGNORECASE), "ignore previous"),
    (re.compile(r"you\s+are\s+now", re.IGNORECASE), "you are now"),
    (re.compile(r"act\s+as", re.IGNORECASE), "act as"),
    (re.compile(r"pretend\s+you", re.IGNORECASE), "pretend you"),
    (re.compile(r"new\s+instructions", re.IGNORECASE), "new instructions"),
    (re.compile(r"(?:^|\s)system\s*:", re.IGNORECASE), "system:"),
    (re.compile(r"(?:^|\s)assistant\s*:", re.IGNORECASE), "assistant:"),
    (re.compile(r"forget\s+everything", re.IGNORECASE), "forget everything"),
]

_MIN_LEAK_SUBSTRING_LENGTH = 21


@dataclass
class ContextCheckResult:
    """Result of validating a single message."""

    valid: bool
    issues: list[str]
    role: str
    content_length: int


@dataclass
class ConversationCheckResult:
    """Result of validating an entire conversation."""

    valid: bool
    total_messages: int
    flagged_messages: int
    issues: list[str]
    total_tokens_estimate: int


@dataclass
class GuardStats:
    """Cumulative statistics for a ContextGuard instance."""

    messages_checked: int
    conversations_checked: int
    issues_found: int
    blocked_count: int


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class ContextGuard:
    """Enforce context boundaries in LLM conversations.

    Validates roles, detects role-confusion injection patterns,
    checks for system prompt leakage, and enforces total context
    length limits.
    """

    def __init__(
        self,
        max_context_length: int = 100_000,
        enforce_roles: bool = True,
    ) -> None:
        self._max_context_length = max_context_length
        self._enforce_roles = enforce_roles
        self._custom_patterns: list[tuple[re.Pattern[str], str]] = []
        self._messages_checked = 0
        self._conversations_checked = 0
        self._issues_found = 0
        self._blocked_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_message(self, role: str, content: str) -> ContextCheckResult:
        """Validate a single message for role and content safety."""
        self._messages_checked += 1
        issues: list[str] = []

        self._check_role(role, issues)
        self._check_role_confusion(content, issues)

        if issues:
            self._issues_found += len(issues)
            self._blocked_count += 1

        return ContextCheckResult(
            valid=len(issues) == 0,
            issues=issues,
            role=role,
            content_length=len(content),
        )

    def validate_conversation(
        self, messages: list[dict[str, str]]
    ) -> ConversationCheckResult:
        """Validate every message in a conversation and check total context length."""
        self._conversations_checked += 1
        all_issues: list[str] = []
        flagged = 0
        total_tokens = 0

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total_tokens += _estimate_tokens(content)
            result = self.validate_message(role, content)
            if not result.valid:
                flagged += 1
                all_issues.extend(result.issues)

        if total_tokens > self._max_context_length:
            overflow_issue = (
                f"Context length {total_tokens} exceeds limit {self._max_context_length}"
            )
            all_issues.append(overflow_issue)
            self._issues_found += 1

        return ConversationCheckResult(
            valid=len(all_issues) == 0,
            total_messages=len(messages),
            flagged_messages=flagged,
            issues=all_issues,
            total_tokens_estimate=total_tokens,
        )

    def detect_role_confusion(self, content: str) -> list[str]:
        """Return descriptions of any role-confusion patterns found in *content*."""
        matches: list[str] = []
        for pattern, description in _DEFAULT_CONFUSION_PATTERNS:
            if pattern.search(content):
                matches.append(description)
        for pattern, description in self._custom_patterns:
            if pattern.search(content):
                matches.append(description)
        return matches

    def detect_context_leak(self, content: str, system_prompt: str) -> bool:
        """Return True if *content* contains a significant substring of *system_prompt*."""
        if len(system_prompt) <= 20:
            return False
        content_lower = content.lower()
        prompt_lower = system_prompt.lower()
        return _has_leaking_substring(content_lower, prompt_lower)

    def enforce_boundaries(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
    ) -> list[dict[str, str]]:
        """Return a cleaned copy of *messages*.

        Messages containing role-confusion patterns are removed entirely.
        Messages that leak the system prompt have the leaked substring
        redacted.
        """
        cleaned: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if self.detect_role_confusion(content):
                self._blocked_count += 1
                continue

            if system_prompt and self.detect_context_leak(content, system_prompt):
                content = _redact_leaked_content(content, system_prompt)

            cleaned.append({"role": role, "content": content})
        return cleaned

    def add_pattern(self, pattern: str, description: str) -> None:
        """Register a custom detection pattern."""
        self._custom_patterns.append(
            (re.compile(pattern, re.IGNORECASE), description)
        )

    def stats(self) -> GuardStats:
        """Return cumulative guard statistics."""
        return GuardStats(
            messages_checked=self._messages_checked,
            conversations_checked=self._conversations_checked,
            issues_found=self._issues_found,
            blocked_count=self._blocked_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_role(self, role: str, issues: list[str]) -> None:
        if self._enforce_roles and role not in VALID_ROLES:
            issues.append(f"Invalid role: {role}")

    def _check_role_confusion(self, content: str, issues: list[str]) -> None:
        for description in self.detect_role_confusion(content):
            issues.append(f"Role confusion detected: {description}")


def _has_leaking_substring(content_lower: str, prompt_lower: str) -> bool:
    """Check whether any prompt substring longer than 20 chars appears in content."""
    for start in range(len(prompt_lower)):
        end = start + _MIN_LEAK_SUBSTRING_LENGTH
        if end > len(prompt_lower):
            break
        if prompt_lower[start:end] in content_lower:
            return True
    return False


def _redact_leaked_content(content: str, system_prompt: str) -> str:
    """Replace the longest leaked substring with [REDACTED]."""
    content_lower = content.lower()
    prompt_lower = system_prompt.lower()

    longest_start = -1
    longest_length = 0

    for start in range(len(prompt_lower)):
        for end in range(len(prompt_lower), start + 20, -1):
            substring = prompt_lower[start:end]
            idx = content_lower.find(substring)
            if idx != -1 and len(substring) > longest_length:
                longest_start = idx
                longest_length = len(substring)
                break

    if longest_start == -1:
        return content

    return (
        content[:longest_start]
        + "[REDACTED]"
        + content[longest_start + longest_length :]
    )
