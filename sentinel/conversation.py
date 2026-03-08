"""Multi-turn conversation safety tracking.

Detects gradual jailbreak escalation, topic persistence attacks, and
conversation-level risk patterns that single-message scanning misses.

Usage:
    from sentinel.conversation import ConversationGuard

    conv = ConversationGuard()
    r1 = conv.add_message("user", "Tell me about chemistry")
    r2 = conv.add_message("assistant", "Chemistry is the study of matter...")
    r3 = conv.add_message("user", "What about dangerous reactions?")
    r4 = conv.add_message("user", "How do I actually synthesize explosives?")
    print(r4.escalation_detected)  # True
    print(conv.conversation_risk)   # RiskLevel.HIGH
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from sentinel.core import SentinelGuard, ScanResult, RiskLevel, Finding


@dataclass
class TurnResult:
    """Result of scanning a single conversation turn."""

    role: str
    scan: ScanResult
    turn_number: int
    conversation_risk: RiskLevel
    escalation_detected: bool
    escalation_reason: str | None = None
    risk_trend: list[str] = field(default_factory=list)


@dataclass
class ConversationSummary:
    """Summary of conversation-level safety analysis."""

    total_turns: int
    user_turns: int
    assistant_turns: int
    conversation_risk: RiskLevel
    escalations: int
    blocked_turns: int
    category_counts: dict[str, int]
    risk_trajectory: list[str]
    flags: list[str]


# Categories that indicate adversarial intent when they persist
_ESCALATION_CATEGORIES = {
    "prompt_injection",
    "harmful_content",
    "tool_use",
    "toxicity",
}

# Number of recent turns to consider for pattern detection
_WINDOW_SIZE = 6

# Threshold: if this fraction of recent user turns have findings, flag escalation
_PERSISTENCE_THRESHOLD = 0.5

# Risk level ordering for comparison
_RISK_ORDER = list(RiskLevel)


class ConversationGuard:
    """Track safety across a multi-turn conversation.

    Detects patterns that single-message scanning misses:
    - Gradual escalation: risk level increases over successive turns
    - Topic persistence: repeated probing of dangerous topics
    - Role confusion: attempts to make the model adopt unsafe personas
    - Sandwich attacks: dangerous content wrapped in benign turns
    """

    def __init__(self, guard: SentinelGuard | None = None):
        self._guard = guard or SentinelGuard.default()
        self._turns: list[TurnResult] = []
        self._user_scans: list[ScanResult] = []
        self._category_history: list[set[str]] = []
        self._risk_history: list[RiskLevel] = []
        self._escalation_count = 0
        self._blocked_count = 0

    @property
    def conversation_risk(self) -> RiskLevel:
        """Current conversation-level risk assessment."""
        if not self._risk_history:
            return RiskLevel.NONE
        # Conversation risk is the max risk seen, but also considers patterns
        max_risk = max(self._risk_history, key=lambda r: _RISK_ORDER.index(r))
        # Escalate if we detect persistent probing
        if self._escalation_count >= 2 and max_risk < RiskLevel.HIGH:
            return RiskLevel.HIGH
        return max_risk

    @property
    def turns(self) -> list[TurnResult]:
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def add_message(
        self,
        role: str,
        text: str,
        context: dict | None = None,
    ) -> TurnResult:
        """Scan a message and update conversation state.

        Args:
            role: "user" or "assistant"
            text: The message content
            context: Optional context for scanners

        Returns:
            TurnResult with per-turn and conversation-level assessment
        """
        scan = self._guard.scan(text, context)
        turn_number = len(self._turns) + 1

        # Track risk history
        self._risk_history.append(scan.risk)

        # Track categories for this turn
        turn_categories = {f.category for f in scan.findings}
        self._category_history.append(turn_categories)

        if scan.blocked:
            self._blocked_count += 1

        # Only track user turns for escalation detection
        if role == "user":
            self._user_scans.append(scan)

        # Check for escalation patterns
        escalation_detected = False
        escalation_reason = None

        if role == "user" and len(self._user_scans) >= 2:
            escalation_detected, escalation_reason = self._check_escalation(
                scan, turn_categories
            )
            if escalation_detected:
                self._escalation_count += 1

        result = TurnResult(
            role=role,
            scan=scan,
            turn_number=turn_number,
            conversation_risk=self.conversation_risk,
            escalation_detected=escalation_detected,
            escalation_reason=escalation_reason,
            risk_trend=[r.value for r in self._risk_history[-_WINDOW_SIZE:]],
        )
        self._turns.append(result)
        return result

    def summarize(self) -> ConversationSummary:
        """Generate a conversation-level safety summary."""
        category_counts: dict[str, int] = {}
        for cats in self._category_history:
            for c in cats:
                category_counts[c] = category_counts.get(c, 0) + 1

        user_turns = sum(1 for t in self._turns if t.role == "user")
        assistant_turns = sum(1 for t in self._turns if t.role == "assistant")

        flags = []
        if self._escalation_count > 0:
            flags.append(
                f"Escalation detected {self._escalation_count} time(s)"
            )
        if self._blocked_count > 0:
            flags.append(f"{self._blocked_count} turn(s) blocked")
        if self._detect_topic_persistence():
            flags.append("Persistent probing of sensitive topics detected")
        if self._detect_sandwich_pattern():
            flags.append("Potential sandwich attack pattern detected")

        return ConversationSummary(
            total_turns=len(self._turns),
            user_turns=user_turns,
            assistant_turns=assistant_turns,
            conversation_risk=self.conversation_risk,
            escalations=self._escalation_count,
            blocked_turns=self._blocked_count,
            category_counts=category_counts,
            risk_trajectory=[r.value for r in self._risk_history],
            flags=flags,
        )

    def reset(self) -> None:
        """Clear conversation state."""
        self._turns.clear()
        self._user_scans.clear()
        self._category_history.clear()
        self._risk_history.clear()
        self._escalation_count = 0
        self._blocked_count = 0

    def _check_escalation(
        self, current_scan: ScanResult, current_categories: set[str]
    ) -> tuple[bool, str | None]:
        """Check if the current turn represents an escalation."""
        # Pattern 1: Risk level increase
        if len(self._risk_history) >= 2:
            prev_risk = self._risk_history[-2]
            curr_risk = self._risk_history[-1]
            if (
                curr_risk > prev_risk
                and curr_risk >= RiskLevel.MEDIUM
                and current_categories & _ESCALATION_CATEGORIES
            ):
                return True, (
                    f"Risk escalated from {prev_risk.value} to "
                    f"{curr_risk.value}"
                )

        # Pattern 2: Persistent category probing
        if len(self._category_history) >= 3:
            recent = self._category_history[-_WINDOW_SIZE:]
            dangerous_turns = sum(
                1 for cats in recent
                if cats & _ESCALATION_CATEGORIES
            )
            if (
                dangerous_turns >= len(recent) * _PERSISTENCE_THRESHOLD
                and current_categories & _ESCALATION_CATEGORIES
            ):
                persistent_cats = set()
                for cats in recent:
                    persistent_cats |= (cats & _ESCALATION_CATEGORIES)
                return True, (
                    f"Persistent probing in categories: "
                    f"{', '.join(sorted(persistent_cats))}"
                )

        # Pattern 3: Re-attempt after block
        if self._blocked_count > 0 and current_scan.findings:
            prev_blocked_cats = set()
            for t in self._turns:
                if t.scan.blocked:
                    prev_blocked_cats |= {f.category for f in t.scan.findings}
            overlap = current_categories & prev_blocked_cats
            if overlap:
                return True, (
                    f"Re-attempt after block in categories: "
                    f"{', '.join(sorted(overlap))}"
                )

        return False, None

    def _detect_topic_persistence(self) -> bool:
        """Detect sustained probing of the same dangerous topic."""
        if len(self._category_history) < 3:
            return False
        recent_user_cats = [
            cats for cats, turn in zip(
                self._category_history, self._turns
            )
            if turn.role == "user" and cats & _ESCALATION_CATEGORIES
        ]
        if len(recent_user_cats) < 3:
            return False
        # Check if any dangerous category appears in 3+ user turns
        from collections import Counter
        cat_counts: Counter[str] = Counter()
        for cats in recent_user_cats:
            for c in cats & _ESCALATION_CATEGORIES:
                cat_counts[c] += 1
        return any(count >= 3 for count in cat_counts.values())

    def _detect_sandwich_pattern(self) -> bool:
        """Detect benign-dangerous-benign pattern (sandwich attack)."""
        if len(self._turns) < 3:
            return False
        user_turns = [t for t in self._turns if t.role == "user"]
        if len(user_turns) < 3:
            return False
        for i in range(len(user_turns) - 2):
            t1, t2, t3 = user_turns[i], user_turns[i + 1], user_turns[i + 2]
            cats1 = {f.category for f in t1.scan.findings}
            cats2 = {f.category for f in t2.scan.findings}
            cats3 = {f.category for f in t3.scan.findings}
            if (
                not (cats1 & _ESCALATION_CATEGORIES)
                and cats2 & _ESCALATION_CATEGORIES
                and not (cats3 & _ESCALATION_CATEGORIES)
                and t2.scan.risk >= RiskLevel.HIGH
            ):
                return True
        return False

    def detect_split_injection(self) -> list[Finding]:
        """Detect injection payloads split across multiple user messages.

        Concatenates recent user messages and scans the combined text.
        If the combined text triggers findings that individual messages
        don't, this indicates a split injection attack.
        """
        findings: list[Finding] = []
        user_turns = [t for t in self._turns if t.role == "user"]
        if len(user_turns) < 2:
            return findings

        recent = user_turns[-min(5, len(user_turns)):]
        combined = " ".join(t.scan.text for t in recent)

        combined_result = self._guard.scan(combined)

        if combined_result.risk >= RiskLevel.HIGH:
            latest_risk = recent[-1].scan.risk
            if latest_risk < RiskLevel.HIGH:
                for f in combined_result.findings:
                    if f.risk >= RiskLevel.HIGH:
                        findings.append(Finding(
                            scanner="conversation",
                            category="split_injection",
                            description=(
                                f"Split injection: attack payload split across "
                                f"{len(recent)} messages (detected: {f.description})"
                            ),
                            risk=RiskLevel.CRITICAL,
                            metadata={
                                "turns_combined": len(recent),
                                "original_category": f.category,
                            },
                        ))
                        break
        return findings

    def detect_context_manipulation(self) -> list[Finding]:
        """Detect false authority claims followed by exploitation.

        Looks for patterns where a user claims authority (developer,
        admin) in one turn and then attempts to leverage it later.
        """
        findings: list[Finding] = []
        user_turns = [t for t in self._turns if t.role == "user"]
        if len(user_turns) < 2:
            return findings

        authority_patterns = [
            re.compile(
                r"\b(?:I\s+am|I'm)\s+(?:your|the|an?)\s+"
                r"(?:developer|admin|owner|creator|supervisor|manager|operator|maintainer)\b",
                re.I,
            ),
            re.compile(
                r"\b(?:system\s+override|admin\s+mode|debug\s+mode|"
                r"maintenance\s+mode|developer\s+mode)\b",
                re.I,
            ),
            re.compile(
                r"\b(?:new\s+(?:instructions|rules|policy)|"
                r"updated\s+(?:instructions|guidelines|policy))\b",
                re.I,
            ),
        ]

        authority_claimed = False
        for turn in user_turns[:-1]:
            for pat in authority_patterns:
                if pat.search(turn.scan.text):
                    authority_claimed = True
                    break
            if authority_claimed:
                break

        if not authority_claimed:
            return findings

        latest_text = user_turns[-1].scan.text
        exploit_signals = [
            re.compile(r"\b(?:therefore|so\s+now|now\s+(?:that|you|please)|given\s+that|since\s+I)\b", re.I),
            re.compile(r"\b(?:override|disable|bypass|ignore|skip|turn\s+off)\b", re.I),
            re.compile(
                r"\b(?:show|reveal|display|dump|print|output)\s+(?:the\s+)?"
                r"(?:system|hidden|internal|secret|private)\b",
                re.I,
            ),
        ]
        for pat in exploit_signals:
            if pat.search(latest_text):
                findings.append(Finding(
                    scanner="conversation",
                    category="context_manipulation",
                    description=(
                        "Context manipulation: false authority claimed in "
                        "earlier turn, now being leveraged"
                    ),
                    risk=RiskLevel.CRITICAL,
                    metadata={"turns": len(user_turns)},
                ))
                break
        return findings

    def detect_progressive_jailbreak(self) -> list[Finding]:
        """Detect jailbreak persona established early, then exploited later.

        Looks for DAN-style persona assignment in earlier turns followed
        by exploitation in the latest turn.
        """
        findings: list[Finding] = []
        user_turns = [t for t in self._turns if t.role == "user"]
        if len(user_turns) < 2:
            return findings

        persona_patterns = [
            re.compile(
                r"\b(?:you\s+are|act\s+as|pretend\s+to\s+be|role\s*play\s+as)\b"
                r".*?\b(?:DAN|unrestricted|unfiltered|jailbreak|evil|uncensored)\b",
                re.I,
            ),
            re.compile(r"\b(?:DAN|unrestricted|unfiltered|evil)\s+mode\b", re.I),
            re.compile(r"\bno\s+(?:rules|restrictions|limits|boundaries|filters|guidelines)\b", re.I),
        ]

        persona_established = False
        for turn in user_turns[:-1]:
            for pat in persona_patterns:
                if pat.search(turn.scan.text):
                    persona_established = True
                    break
            if persona_established:
                break

        if not persona_established:
            return findings

        latest = user_turns[-1].scan.text
        exploit_patterns = [
            r"\bnow\b.*?\b(?:tell|show|explain|help|do)\b",
            r"\bas\s+(?:DAN|that\s+character|that\s+persona)\b",
            r"\bin\s+(?:that|this)\s+(?:mode|role|character)\b",
            r"\b(?:remember|recall)\s+(?:you\s+are|your\s+role)\b",
        ]
        for pat in exploit_patterns:
            if re.search(pat, latest, re.I):
                findings.append(Finding(
                    scanner="conversation",
                    category="progressive_jailbreak",
                    description=(
                        "Progressive jailbreak: persona established in "
                        "earlier turn, now being exploited"
                    ),
                    risk=RiskLevel.CRITICAL,
                    metadata={"turns": len(user_turns)},
                ))
                break
        return findings

    def full_scan(self) -> list[Finding]:
        """Run all cross-turn detection methods and return combined findings."""
        findings: list[Finding] = []
        findings.extend(self.detect_split_injection())
        findings.extend(self.detect_context_manipulation())
        findings.extend(self.detect_progressive_jailbreak())
        return findings
