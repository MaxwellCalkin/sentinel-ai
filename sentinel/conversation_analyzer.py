"""Deep conversation analysis for safety, engagement, and quality.

Analyze multi-turn conversations for safety metrics, engagement patterns,
topic drift, repetition, escalation, and overall quality scoring.

Usage:
    from sentinel.conversation_analyzer import ConversationAnalyzer

    analyzer = ConversationAnalyzer()
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]
    analysis = analyzer.analyze(messages)
    print(analysis.total_turns)
    print(analysis.topics_detected)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ConversationAnalysis:
    """Full conversation analysis results."""
    total_turns: int
    user_turns: int
    assistant_turns: int
    avg_user_length: float
    avg_assistant_length: float
    total_tokens_estimate: int
    duration_turns: int
    topics_detected: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A detected conversation pattern."""
    pattern_type: str
    description: str
    severity: str
    start_turn: int
    end_turn: int


@dataclass
class QualityScore:
    """Conversation quality assessment."""
    overall: float
    relevance: float
    coherence: float
    safety: float
    grade: str


@dataclass
class EngagementMetrics:
    """Conversation engagement statistics."""
    response_ratio: float
    avg_turn_gap: float
    user_initiative_rate: float
    question_count: int
    follow_up_rate: float


_STOP_WORDS = frozenset({
    "about", "above", "after", "again", "along", "already", "always",
    "among", "another", "around", "because", "before", "being", "below",
    "between", "cannot", "could", "during", "every", "from", "going",
    "great", "having", "itself", "maybe", "might", "never", "other",
    "really", "shall", "should", "since", "still", "their", "there",
    "these", "thing", "things", "those", "through", "under", "until",
    "using", "very", "want", "was", "were", "what", "when", "where",
    "which", "while", "will", "with", "without", "would", "your",
})

_HARMFUL_KEYWORDS = re.compile(
    r"\b(?:hack|exploit|weapon|bomb|poison|steal|fraud|scam|illegal"
    r"|kill|attack|destroy|malware|virus|ransomware)\b",
    re.IGNORECASE,
)

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")


def _extract_words(text: str) -> list[str]:
    """Extract lowercase alphabetic words from text."""
    return [w.lower() for w in _WORD_PATTERN.findall(text)]


def _extract_keywords(messages: list[dict]) -> list[str]:
    """Extract frequent meaningful keywords from messages."""
    word_counts: dict[str, int] = {}
    for message in messages:
        for word in _extract_words(message.get("content", "")):
            if len(word) > 5 and word not in _STOP_WORDS:
                word_counts[word] = word_counts.get(word, 0) + 1

    return sorted(
        [word for word, count in word_counts.items() if count >= 3],
        key=lambda w: word_counts[w],
        reverse=True,
    )


def _keyword_set_for_messages(messages: list[dict]) -> set[str]:
    """Build a set of meaningful words from a list of messages."""
    words: set[str] = set()
    for message in messages:
        for word in _extract_words(message.get("content", "")):
            if len(word) > 5 and word not in _STOP_WORDS:
                words.add(word)
    return words


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _is_question(text: str) -> bool:
    """Check whether a message is a question."""
    return "?" in text


def _count_punctuation_marks(text: str, char: str) -> int:
    """Count occurrences of a specific punctuation character."""
    return text.count(char)


def _compute_grade(score: float) -> str:
    """Convert a 0-1 score to a letter grade."""
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"


def _estimate_tokens(text: str) -> int:
    """Estimate token count as len(content) / 4."""
    return len(text) // 4


class ConversationAnalyzer:
    """Analyze multi-turn conversations for safety, quality, and engagement."""

    def __init__(self, max_turns: int = 100) -> None:
        self._max_turns = max_turns

    def analyze(self, messages: list[dict]) -> ConversationAnalysis:
        """Analyze a conversation and return comprehensive metrics."""
        truncated = messages[:self._max_turns]

        user_messages = [m for m in truncated if m.get("role") == "user"]
        assistant_messages = [m for m in truncated if m.get("role") == "assistant"]

        avg_user_length = _average_content_length(user_messages)
        avg_assistant_length = _average_content_length(assistant_messages)

        total_tokens = sum(
            _estimate_tokens(m.get("content", "")) for m in truncated
        )

        topics = _extract_keywords(truncated)

        return ConversationAnalysis(
            total_turns=len(truncated),
            user_turns=len(user_messages),
            assistant_turns=len(assistant_messages),
            avg_user_length=avg_user_length,
            avg_assistant_length=avg_assistant_length,
            total_tokens_estimate=total_tokens,
            duration_turns=len(truncated),
            topics_detected=topics,
        )

    def detect_patterns(self, messages: list[dict]) -> list[Pattern]:
        """Detect conversation patterns: repetition, escalation, topic drift."""
        truncated = messages[:self._max_turns]
        patterns: list[Pattern] = []

        patterns.extend(_detect_repetition(truncated))
        patterns.extend(_detect_escalation(truncated))
        patterns.extend(_detect_topic_drift(truncated))

        return patterns

    def quality_score(self, messages: list[dict]) -> QualityScore:
        """Score conversation quality on a 0-1 scale with letter grade."""
        truncated = messages[:self._max_turns]

        relevance = _compute_relevance(truncated)
        coherence = _compute_coherence(truncated)
        safety = _compute_safety(truncated)

        overall = (relevance + coherence + safety) / 3.0
        overall = max(0.0, min(1.0, overall))

        return QualityScore(
            overall=round(overall, 4),
            relevance=round(relevance, 4),
            coherence=round(coherence, 4),
            safety=round(safety, 4),
            grade=_compute_grade(overall),
        )

    def engagement_metrics(self, messages: list[dict]) -> EngagementMetrics:
        """Compute engagement statistics for the conversation."""
        truncated = messages[:self._max_turns]

        user_messages = [m for m in truncated if m.get("role") == "user"]
        assistant_messages = [m for m in truncated if m.get("role") == "assistant"]

        response_ratio = _compute_response_ratio(user_messages, assistant_messages)
        question_count = sum(
            1 for m in user_messages if _is_question(m.get("content", ""))
        )
        user_initiative_rate = _compute_initiative_rate(user_messages)
        follow_up_rate = _compute_follow_up_rate(truncated)
        avg_turn_gap = _compute_avg_turn_gap(truncated)

        return EngagementMetrics(
            response_ratio=round(response_ratio, 4),
            avg_turn_gap=round(avg_turn_gap, 4),
            user_initiative_rate=round(user_initiative_rate, 4),
            question_count=question_count,
            follow_up_rate=round(follow_up_rate, 4),
        )

    def summarize(self, messages: list[dict]) -> str:
        """Generate a brief text summary of the conversation."""
        truncated = messages[:self._max_turns]

        if not truncated:
            return "Empty conversation."

        user_messages = [m for m in truncated if m.get("role") == "user"]
        assistant_messages = [m for m in truncated if m.get("role") == "assistant"]
        topics = _extract_keywords(truncated)

        parts = [
            f"Conversation with {len(truncated)} turns"
            f" ({len(user_messages)} user, {len(assistant_messages)} assistant).",
        ]

        if topics:
            parts.append(f"Topics: {', '.join(topics[:5])}.")

        quality = self.quality_score(truncated)
        parts.append(f"Quality grade: {quality.grade} ({quality.overall:.2f}).")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _average_content_length(messages: list[dict]) -> float:
    """Compute average content length for a list of messages."""
    if not messages:
        return 0.0
    total = sum(len(m.get("content", "")) for m in messages)
    return total / len(messages)


def _detect_repetition(messages: list[dict]) -> list[Pattern]:
    """Flag messages whose content appears more than once (case-insensitive)."""
    seen: dict[str, int] = {}
    patterns: list[Pattern] = []
    flagged: set[str] = set()

    for index, message in enumerate(messages):
        normalized = message.get("content", "").strip().lower()
        if not normalized:
            continue
        if normalized in seen and normalized not in flagged:
            patterns.append(Pattern(
                pattern_type="repetition",
                description=f"Message repeated at turns {seen[normalized]} and {index}",
                severity="medium",
                start_turn=seen[normalized],
                end_turn=index,
            ))
            flagged.add(normalized)
        elif normalized not in seen:
            seen[normalized] = index

    return patterns


def _detect_escalation(messages: list[dict]) -> list[Pattern]:
    """Detect increasing question marks or exclamation marks across user turns."""
    user_turns = [
        (i, m) for i, m in enumerate(messages) if m.get("role") == "user"
    ]
    if len(user_turns) < 3:
        return []

    patterns: list[Pattern] = []
    _check_punctuation_escalation(user_turns, "?", "question_escalation", patterns)
    _check_punctuation_escalation(user_turns, "!", "exclamation_escalation", patterns)

    return patterns


def _check_punctuation_escalation(
    user_turns: list[tuple[int, dict]],
    char: str,
    escalation_type: str,
    patterns: list[Pattern],
) -> None:
    """Check if a punctuation character is increasing across user turns."""
    counts = [
        _count_punctuation_marks(m.get("content", ""), char)
        for _, m in user_turns
    ]

    increasing_streak = 0
    for i in range(1, len(counts)):
        if counts[i] > counts[i - 1]:
            increasing_streak += 1
        else:
            increasing_streak = 0

    if increasing_streak >= 2:
        patterns.append(Pattern(
            pattern_type="escalation",
            description=f"Increasing '{char}' usage across user turns",
            severity="low",
            start_turn=user_turns[0][0],
            end_turn=user_turns[-1][0],
        ))


def _detect_topic_drift(messages: list[dict]) -> list[Pattern]:
    """Detect topic drift when first-half and second-half keywords diverge."""
    if len(messages) < 4:
        return []

    mid = len(messages) // 2
    first_half_keywords = _keyword_set_for_messages(messages[:mid])
    second_half_keywords = _keyword_set_for_messages(messages[mid:])

    if not first_half_keywords and not second_half_keywords:
        return []

    similarity = _jaccard_similarity(first_half_keywords, second_half_keywords)

    if similarity < 0.3:
        return [Pattern(
            pattern_type="topic_drift",
            description=f"Topic drift detected (Jaccard similarity {similarity:.2f})",
            severity="low",
            start_turn=0,
            end_turn=len(messages) - 1,
        )]

    return []


def _compute_relevance(messages: list[dict]) -> float:
    """Score relevance based on response-to-question length ratio.

    If the average assistant response length is reasonably proportional
    to user messages, relevance is high.
    """
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

    if not user_msgs or not assistant_msgs:
        return 0.5

    avg_user = _average_content_length(user_msgs)
    avg_assistant = _average_content_length(assistant_msgs)

    if avg_user == 0:
        return 0.7 if avg_assistant > 0 else 0.5

    ratio = avg_assistant / avg_user
    # Ideal ratio around 1-3x; penalize very low or very high
    if ratio < 0.2:
        return 0.3
    if ratio > 10:
        return 0.6
    return min(1.0, 0.5 + ratio * 0.15)


def _compute_coherence(messages: list[dict]) -> float:
    """Score coherence; penalize repetition."""
    if not messages:
        return 1.0

    seen: set[str] = set()
    repetition_count = 0
    for message in messages:
        normalized = message.get("content", "").strip().lower()
        if normalized in seen:
            repetition_count += 1
        seen.add(normalized)

    penalty = repetition_count * 0.15
    return max(0.0, 1.0 - penalty)


def _compute_safety(messages: list[dict]) -> float:
    """Score safety; penalize harmful keywords."""
    if not messages:
        return 1.0

    harmful_count = 0
    for message in messages:
        content = message.get("content", "")
        harmful_count += len(_HARMFUL_KEYWORDS.findall(content))

    penalty = harmful_count * 0.1
    return max(0.0, 1.0 - penalty)


def _compute_response_ratio(
    user_messages: list[dict],
    assistant_messages: list[dict],
) -> float:
    """Ratio of assistant turns to user turns."""
    if not user_messages:
        return 0.0
    return len(assistant_messages) / len(user_messages)


def _compute_initiative_rate(user_messages: list[dict]) -> float:
    """Fraction of user messages that are questions."""
    if not user_messages:
        return 0.0
    question_count = sum(
        1 for m in user_messages if _is_question(m.get("content", ""))
    )
    return question_count / len(user_messages)


def _compute_follow_up_rate(messages: list[dict]) -> float:
    """Fraction of consecutive user turns (follow-ups without waiting for response)."""
    if len(messages) < 2:
        return 0.0

    follow_ups = 0
    opportunities = 0
    for i in range(1, len(messages)):
        if messages[i].get("role") == "user":
            opportunities += 1
            if messages[i - 1].get("role") == "user":
                follow_ups += 1

    if opportunities == 0:
        return 0.0
    return follow_ups / opportunities


def _compute_avg_turn_gap(messages: list[dict]) -> float:
    """Average character length difference between consecutive messages."""
    if len(messages) < 2:
        return 0.0

    gaps = []
    for i in range(1, len(messages)):
        length_a = len(messages[i - 1].get("content", ""))
        length_b = len(messages[i].get("content", ""))
        gaps.append(abs(length_b - length_a))

    return sum(gaps) / len(gaps)
