"""Topic enforcement for LLM conversations.

Ensures LLM responses stay on-topic by checking against allowed
and blocked topic definitions. Useful for customer support bots,
educational assistants, and domain-specific applications.

Usage:
    from sentinel.topic_guard import TopicGuard

    guard = TopicGuard()
    guard.add_allowed_topic("python programming", keywords=["python", "code", "function", "class"])
    guard.add_blocked_topic("politics", keywords=["election", "democrat", "republican", "vote"])

    result = guard.check("How do I write a Python function?")
    assert result.on_topic
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopicMatch:
    """A topic match."""
    topic: str
    score: float
    matched_keywords: list[str]
    is_allowed: bool


@dataclass
class TopicResult:
    """Result of topic checking."""
    text: str
    on_topic: bool
    matches: list[TopicMatch] = field(default_factory=list)
    blocked_topics: list[str] = field(default_factory=list)
    top_topic: str = ""
    top_score: float = 0.0

    @property
    def is_off_topic(self) -> bool:
        return not self.on_topic


@dataclass
class TopicDef:
    """Definition of a topic."""
    name: str
    keywords: list[str]
    weight: float = 1.0
    is_allowed: bool = True


_WORD_RE = re.compile(r"[a-z0-9]+")


class TopicGuard:
    """Enforce on-topic conversations.

    Define allowed and blocked topics with keyword sets.
    Checks text against all topics and determines if it's
    on-topic based on keyword overlap.
    """

    def __init__(
        self,
        require_topic: bool = False,
        threshold: float = 0.1,
    ) -> None:
        """
        Args:
            require_topic: If True, text must match an allowed topic.
            threshold: Minimum score to consider a topic matched.
        """
        self._require_topic = require_topic
        self._threshold = threshold
        self._allowed: list[TopicDef] = []
        self._blocked: list[TopicDef] = []

    def add_allowed_topic(
        self,
        name: str,
        keywords: list[str],
        weight: float = 1.0,
    ) -> None:
        """Add an allowed topic."""
        self._allowed.append(TopicDef(
            name=name,
            keywords=[k.lower() for k in keywords],
            weight=weight,
            is_allowed=True,
        ))

    def add_blocked_topic(
        self,
        name: str,
        keywords: list[str],
        weight: float = 1.0,
    ) -> None:
        """Add a blocked topic."""
        self._blocked.append(TopicDef(
            name=name,
            keywords=[k.lower() for k in keywords],
            weight=weight,
            is_allowed=False,
        ))

    @property
    def topic_count(self) -> int:
        return len(self._allowed) + len(self._blocked)

    def check(self, text: str) -> TopicResult:
        """Check if text is on-topic.

        Returns:
            TopicResult with topic matches and on-topic status.
        """
        words = set(_WORD_RE.findall(text.lower()))
        all_topics = self._allowed + self._blocked
        matches: list[TopicMatch] = []
        blocked: list[str] = []

        for topic_def in all_topics:
            matched_kw = [k for k in topic_def.keywords if k in words]
            if not matched_kw:
                # Also check multi-word keywords
                text_lower = text.lower()
                matched_kw = [k for k in topic_def.keywords if k in text_lower]

            if matched_kw:
                score = (len(matched_kw) / len(topic_def.keywords)) * topic_def.weight
                score = min(1.0, score)

                if score >= self._threshold:
                    matches.append(TopicMatch(
                        topic=topic_def.name,
                        score=round(score, 4),
                        matched_keywords=matched_kw,
                        is_allowed=topic_def.is_allowed,
                    ))

                    if not topic_def.is_allowed:
                        blocked.append(topic_def.name)

        # Determine on-topic status
        has_allowed = any(m.is_allowed for m in matches)
        has_blocked = len(blocked) > 0

        if has_blocked:
            on_topic = False
        elif self._require_topic:
            on_topic = has_allowed
        else:
            on_topic = True

        top_topic = ""
        top_score = 0.0
        if matches:
            best = max(matches, key=lambda m: m.score)
            top_topic = best.topic
            top_score = best.score

        return TopicResult(
            text=text,
            on_topic=on_topic,
            matches=matches,
            blocked_topics=blocked,
            top_topic=top_topic,
            top_score=top_score,
        )

    def clear(self) -> None:
        """Remove all topics."""
        self._allowed.clear()
        self._blocked.clear()
