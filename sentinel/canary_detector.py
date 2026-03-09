"""Canary token detection for LLM outputs.

Detects if LLM outputs contain planted canary tokens that indicate
training data memorization, data leakage, or unauthorized data access.

Usage:
    from sentinel.canary_detector import CanaryDetector

    detector = CanaryDetector()
    detector.add_canary("CANARY-abc123-TOKEN")

    result = detector.check("The model output contains CANARY-abc123-TOKEN leaked")
    assert result.detected
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class CanaryMatch:
    """A detected canary token match."""
    canary_id: str
    matched_text: str
    match_type: str       # exact, partial, hash
    span: tuple[int, int] | None = None
    confidence: float = 1.0


@dataclass
class CanaryResult:
    """Result of canary token checking."""
    text: str
    detected: bool
    matches: list[CanaryMatch] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def canary_ids(self) -> list[str]:
        return list(set(m.canary_id for m in self.matches))


class CanaryDetector:
    """Detect canary tokens in LLM outputs.

    Supports exact match, regex patterns, hash-based detection,
    and fuzzy matching for canary tokens.
    """

    def __init__(self) -> None:
        self._exact: dict[str, str] = {}       # token -> canary_id
        self._patterns: list[tuple[str, re.Pattern]] = []  # (canary_id, pattern)
        self._hashes: dict[str, str] = {}       # hash -> canary_id
        self._case_insensitive: dict[str, str] = {}  # lower token -> canary_id

    def add_canary(
        self,
        token: str,
        canary_id: str | None = None,
        case_sensitive: bool = True,
    ) -> str:
        """Add a canary token to detect.

        Args:
            token: The canary token string.
            canary_id: Optional identifier (auto-generated if None).
            case_sensitive: Whether matching is case-sensitive.

        Returns:
            The canary_id for this token.
        """
        cid = canary_id or f"canary_{hashlib.sha256(token.encode()).hexdigest()[:8]}"

        self._exact[token] = cid
        self._hashes[hashlib.sha256(token.encode()).hexdigest()] = cid
        self._hashes[hashlib.md5(token.encode()).hexdigest()] = cid

        if not case_sensitive:
            self._case_insensitive[token.lower()] = cid

        return cid

    def add_pattern(self, pattern: str, canary_id: str | None = None) -> str:
        """Add a regex pattern to detect.

        Args:
            pattern: Regex pattern to match.
            canary_id: Optional identifier.

        Returns:
            The canary_id for this pattern.
        """
        cid = canary_id or f"pattern_{hashlib.sha256(pattern.encode()).hexdigest()[:8]}"
        self._patterns.append((cid, re.compile(pattern)))
        return cid

    def add_canaries(self, tokens: Sequence[str]) -> list[str]:
        """Add multiple canary tokens. Returns list of canary_ids."""
        return [self.add_canary(t) for t in tokens]

    @property
    def canary_count(self) -> int:
        """Number of canary tokens registered."""
        return len(self._exact) + len(self._patterns)

    def check(self, text: str) -> CanaryResult:
        """Check text for canary tokens."""
        matches: list[CanaryMatch] = []

        # Exact matches
        for token, cid in self._exact.items():
            idx = text.find(token)
            while idx >= 0:
                matches.append(CanaryMatch(
                    canary_id=cid,
                    matched_text=token,
                    match_type="exact",
                    span=(idx, idx + len(token)),
                    confidence=1.0,
                ))
                idx = text.find(token, idx + 1)

        # Case-insensitive matches
        text_lower = text.lower()
        for token_lower, cid in self._case_insensitive.items():
            idx = text_lower.find(token_lower)
            while idx >= 0:
                # Don't duplicate exact matches
                if not any(m.span == (idx, idx + len(token_lower)) for m in matches):
                    matches.append(CanaryMatch(
                        canary_id=cid,
                        matched_text=text[idx:idx + len(token_lower)],
                        match_type="case_insensitive",
                        span=(idx, idx + len(token_lower)),
                        confidence=0.95,
                    ))
                idx = text_lower.find(token_lower, idx + 1)

        # Pattern matches
        for cid, pattern in self._patterns:
            for m in pattern.finditer(text):
                matches.append(CanaryMatch(
                    canary_id=cid,
                    matched_text=m.group(),
                    match_type="pattern",
                    span=(m.start(), m.end()),
                    confidence=0.9,
                ))

        # Hash matches (check if any word hashes match)
        words = re.findall(r"\S+", text)
        for word in words:
            word_hash = hashlib.sha256(word.encode()).hexdigest()
            if word_hash in self._hashes:
                cid = self._hashes[word_hash]
                if not any(m.matched_text == word and m.match_type == "exact" for m in matches):
                    matches.append(CanaryMatch(
                        canary_id=cid,
                        matched_text=word,
                        match_type="hash",
                        confidence=0.85,
                    ))

            md5_hash = hashlib.md5(word.encode()).hexdigest()
            if md5_hash in self._hashes:
                cid = self._hashes[md5_hash]
                if not any(m.matched_text == word for m in matches):
                    matches.append(CanaryMatch(
                        canary_id=cid,
                        matched_text=word,
                        match_type="hash",
                        confidence=0.85,
                    ))

        return CanaryResult(
            text=text,
            detected=len(matches) > 0,
            matches=matches,
        )

    def check_batch(self, texts: list[str]) -> list[CanaryResult]:
        """Check multiple texts for canary tokens."""
        return [self.check(text) for text in texts]

    def clear(self) -> None:
        """Remove all canary tokens."""
        self._exact.clear()
        self._patterns.clear()
        self._hashes.clear()
        self._case_insensitive.clear()

    @staticmethod
    def generate_token(prefix: str = "CANARY", length: int = 16) -> str:
        """Generate a random canary token."""
        import secrets
        random_part = secrets.token_hex(length // 2)
        return f"{prefix}-{random_part}-TOKEN"
