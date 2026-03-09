"""Multi-turn conversation safety analyzer.

Detect safety drift, gradual escalation, topic steering,
and manipulation patterns across conversation turns.

Usage:
    from sentinel.multi_turn import MultiTurnAnalyzer

    analyzer = MultiTurnAnalyzer()
    analyzer.add_turn("user", "Tell me about chemistry")
    analyzer.add_turn("assistant", "Chemistry is the study of matter...")
    analyzer.add_turn("user", "What about dangerous reactions?")
    report = analyzer.analyze()
    print(report.risk_trend)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    risk_score: float = 0.0
    flags: list[str] = field(default_factory=list)


@dataclass
class MultiTurnReport:
    """Analysis of multi-turn conversation safety."""
    turns_analyzed: int
    risk_trend: str  # "escalating", "stable", "de-escalating"
    avg_risk: float
    max_risk: float
    escalation_points: list[int]  # turn indices where risk increased significantly
    flags: list[str]
    safe: bool


# Escalation signals — user turns that may indicate manipulation
_ESCALATION_PATTERNS = [
    (r'(?i)\bnow\s+(?:tell|show|give)\s+me\b', "directive_escalation"),
    (r'(?i)\bforget\s+(?:all|everything|your|the)\b', "memory_wipe"),
    (r'(?i)\bignore\s+(?:previous|prior|above|all)\b', "instruction_override"),
    (r'(?i)\bpretend\s+(?:you|to)\b', "role_manipulation"),
    (r'(?i)\bact\s+as\s+(?:if|a|an)\b', "role_manipulation"),
    (r'(?i)\bjust\s+this\s+once\b', "boundary_pushing"),
    (r'(?i)\bno\s+one\s+will\s+(?:know|see|find)\b', "secrecy_appeal"),
    (r'(?i)\byou\s+(?:must|have\s+to|need\s+to)\b', "authority_assertion"),
    (r'(?i)\bdon\'?t\s+(?:worry|care)\s+about\b', "safety_dismissal"),
    (r'(?i)\bactually,?\s+(?:can|could)\s+you\b', "gradual_steering"),
]

# Topic risk signals
_RISK_KEYWORDS = [
    (r'(?i)\b(?:hack|exploit|bypass|crack)\b', 0.3, "security_risk"),
    (r'(?i)\b(?:weapon|bomb|poison|explosive)\b', 0.5, "dangerous_content"),
    (r'(?i)\b(?:steal|fraud|scam|illegal)\b', 0.4, "illegal_activity"),
    (r'(?i)\b(?:harm|hurt|kill|attack|destroy)\b', 0.4, "violence"),
    (r'(?i)\b(?:password|credential|private\s+key)\b', 0.3, "data_extraction"),
]


class MultiTurnAnalyzer:
    """Analyze multi-turn conversations for safety drift."""

    def __init__(
        self,
        escalation_threshold: float = 0.3,
        max_turns: int = 500,
    ) -> None:
        self._turns: list[Turn] = []
        self._escalation_threshold = escalation_threshold
        self._max_turns = max_turns

    def add_turn(self, role: str, content: str, timestamp: float | None = None) -> Turn:
        """Add a conversation turn."""
        risk_score, flags = self._score_turn(role, content)
        turn = Turn(
            role=role, content=content,
            timestamp=timestamp or time.time(),
            risk_score=risk_score, flags=flags,
        )
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]
        return turn

    def analyze(self) -> MultiTurnReport:
        """Analyze the conversation for safety patterns."""
        if not self._turns:
            return MultiTurnReport(
                turns_analyzed=0, risk_trend="stable", avg_risk=0.0,
                max_risk=0.0, escalation_points=[], flags=[], safe=True,
            )

        n = len(self._turns)
        risks = [t.risk_score for t in self._turns]
        avg_risk = sum(risks) / n
        max_risk = max(risks)

        # Find escalation points
        escalation_points = []
        for i in range(1, n):
            if risks[i] - risks[i - 1] > self._escalation_threshold:
                escalation_points.append(i)

        # Determine trend
        if n < 3:
            trend = "stable"
        else:
            mid = n // 2
            first_half = sum(risks[:mid]) / mid if mid > 0 else 0
            second_half = sum(risks[mid:]) / (n - mid) if (n - mid) > 0 else 0
            diff = second_half - first_half
            if diff > 0.15:
                trend = "escalating"
            elif diff < -0.15:
                trend = "de-escalating"
            else:
                trend = "stable"

        # Collect all flags
        all_flags = []
        for t in self._turns:
            for f in t.flags:
                if f not in all_flags:
                    all_flags.append(f)

        # Check for multi-turn manipulation patterns
        user_turns = [t for t in self._turns if t.role == "user"]
        if len(user_turns) >= 3:
            flagged = sum(1 for t in user_turns if t.flags)
            if flagged / len(user_turns) > 0.5:
                if "sustained_manipulation" not in all_flags:
                    all_flags.append("sustained_manipulation")

        safe = max_risk < 0.5 and trend != "escalating" and "sustained_manipulation" not in all_flags

        return MultiTurnReport(
            turns_analyzed=n,
            risk_trend=trend,
            avg_risk=round(avg_risk, 4),
            max_risk=round(max_risk, 4),
            escalation_points=escalation_points,
            flags=all_flags,
            safe=safe,
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self._turns.clear()

    def get_turns(self) -> list[Turn]:
        """Get all recorded turns."""
        return list(self._turns)

    def _score_turn(self, role: str, content: str) -> tuple[float, list[str]]:
        """Score a turn for risk."""
        score = 0.0
        flags: list[str] = []

        if role == "user":
            # Check escalation patterns
            for pattern, flag in _ESCALATION_PATTERNS:
                if re.search(pattern, content):
                    score += 0.2
                    if flag not in flags:
                        flags.append(flag)

        # Check risk keywords (both roles)
        for pattern, weight, flag in _RISK_KEYWORDS:
            if re.search(pattern, content):
                score += weight
                if flag not in flags:
                    flags.append(flag)

        return min(1.0, score), flags
