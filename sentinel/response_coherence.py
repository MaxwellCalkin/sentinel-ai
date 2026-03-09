"""LLM response coherence and internal consistency analysis.

Measures coherence across four dimensions: topic consistency,
structural quality, contradiction detection, and repetition.
Detects contradictions, topic drift, and structural issues.

Zero external dependencies — uses word-level analysis for
language-agnostic coherence scoring.

Usage:
    from sentinel.response_coherence import ResponseCoherence

    checker = ResponseCoherence()
    report = checker.analyze("The sky is blue. It is a clear day.")
    print(report.score.overall)   # 0.0-1.0
    print(report.score.grade)     # "A"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CoherenceScore:
    """Multi-dimensional coherence score for a text."""
    text: str
    topic_consistency: float
    structural_quality: float
    contradiction_score: float
    repetition_score: float
    overall: float
    grade: str


@dataclass
class CoherenceIssue:
    """A specific coherence problem found in the text."""
    issue_type: str
    description: str
    severity: str
    location: str = ""


@dataclass
class CoherenceReport:
    """Full coherence analysis report."""
    score: CoherenceScore
    issues: list[CoherenceIssue]
    recommendations: list[str]


@dataclass
class CoherenceStats:
    """Cumulative statistics across multiple analyses."""
    total_analyzed: int = 0
    avg_overall: float = 0.0
    by_grade: dict[str, int] = field(default_factory=dict)
    total_issues: int = 0


_ANTONYM_PAIRS = [
    ("true", "false"),
    ("yes", "no"),
    ("good", "bad"),
    ("always", "never"),
    ("all", "none"),
    ("possible", "impossible"),
    ("safe", "unsafe"),
    ("correct", "incorrect"),
    ("valid", "invalid"),
    ("allow", "deny"),
    ("accept", "reject"),
    ("increase", "decrease"),
    ("above", "below"),
    ("before", "after"),
    ("success", "failure"),
]

_NEGATION_PAIRS = [
    ("is", "is not"),
    ("can", "cannot"),
    ("will", "will not"),
    ("should", "should not"),
    ("does", "does not"),
    ("has", "has not"),
    ("was", "was not"),
    ("are", "are not"),
    ("do", "do not"),
    ("could", "could not"),
]


def _tokenize_words(text: str) -> list[str]:
    """Extract lowercase words from text."""
    return re.findall(r"\b\w+\b", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    raw = re.split(r"[.!?]+", text)
    return [s.strip() for s in raw if s.strip()]


def _compute_topic_consistency(sentences: list[str]) -> float:
    """Measure vocabulary overlap between first and second half (Jaccard)."""
    if len(sentences) < 2:
        return 1.0

    midpoint = len(sentences) // 2
    first_half = " ".join(sentences[:midpoint])
    second_half = " ".join(sentences[midpoint:])

    words_first = set(_tokenize_words(first_half))
    words_second = set(_tokenize_words(second_half))

    if not words_first or not words_second:
        return 1.0

    intersection = len(words_first & words_second)
    union = len(words_first | words_second)

    return intersection / union if union > 0 else 1.0


def _compute_structural_quality(sentences: list[str]) -> float:
    """Score based on sentence length variance (lower variance = higher)."""
    if len(sentences) < 2:
        return 1.0

    lengths = [len(_tokenize_words(s)) for s in sentences]
    mean_length = sum(lengths) / len(lengths)

    if mean_length == 0:
        return 1.0

    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    normalized_std = (variance ** 0.5) / mean_length

    # Map normalized std to 0-1 (std/mean of 0 => 1.0, std/mean of 2+ => 0.0)
    return max(0.0, min(1.0, 1.0 - normalized_std / 2.0))


def _compute_contradiction_score(sentences: list[str]) -> float:
    """Detect contradictions via antonym pairs and negation patterns."""
    if len(sentences) < 2:
        return 1.0

    contradiction_count = 0
    window_size = 3

    for i in range(len(sentences)):
        window_end = min(i + window_size, len(sentences))
        for j in range(i + 1, window_end):
            if _sentences_contradict(sentences[i], sentences[j]):
                contradiction_count += 1

    max_pairs = sum(min(window_size - 1, len(sentences) - i - 1)
                    for i in range(len(sentences) - 1))

    if max_pairs == 0:
        return 1.0

    contradiction_ratio = contradiction_count / max_pairs
    return max(0.0, min(1.0, 1.0 - contradiction_ratio))


def _sentences_contradict(sentence_a: str, sentence_b: str) -> bool:
    """Check if two sentences contain contradicting patterns."""
    lower_a = sentence_a.lower()
    lower_b = sentence_b.lower()

    for positive, negative in _NEGATION_PAIRS:
        a_has_positive = re.search(r"\b" + re.escape(positive) + r"\b", lower_a)
        a_has_negative = re.search(r"\b" + re.escape(negative) + r"\b", lower_a)
        b_has_positive = re.search(r"\b" + re.escape(positive) + r"\b", lower_b)
        b_has_negative = re.search(r"\b" + re.escape(negative) + r"\b", lower_b)

        if (a_has_positive and not a_has_negative and b_has_negative) or \
           (a_has_negative and b_has_positive and not b_has_negative):
            return True

    words_a = set(_tokenize_words(lower_a))
    words_b = set(_tokenize_words(lower_b))

    for word_pos, word_neg in _ANTONYM_PAIRS:
        if (word_pos in words_a and word_neg in words_b) or \
           (word_neg in words_a and word_pos in words_b):
            return True

    return False


def _compute_repetition_score(sentences: list[str]) -> float:
    """Score based on duplicate sentence ratio (1 = no repetition)."""
    if len(sentences) <= 1:
        return 1.0

    normalized = [" ".join(_tokenize_words(s)) for s in sentences]
    unique_count = len(set(normalized))
    duplicate_ratio = 1.0 - (unique_count / len(normalized))

    return max(0.0, min(1.0, 1.0 - duplicate_ratio))


def _compute_overall(
    topic: float,
    structural: float,
    contradiction: float,
    repetition: float,
) -> float:
    """Weighted average of all dimensions."""
    return (
        0.3 * topic
        + 0.2 * structural
        + 0.3 * contradiction
        + 0.2 * repetition
    )


def _grade_from_score(score: float) -> str:
    """Map a 0-1 score to a letter grade."""
    if score >= 0.8:
        return "A"
    if score >= 0.6:
        return "B"
    if score >= 0.4:
        return "C"
    if score >= 0.2:
        return "D"
    return "F"


def _build_issues(
    topic: float,
    structural: float,
    contradiction: float,
    repetition: float,
) -> list[CoherenceIssue]:
    """Generate issues for dimensions scoring below 0.5."""
    issues: list[CoherenceIssue] = []
    threshold = 0.5

    if topic < threshold:
        severity = "high" if topic < 0.2 else "medium"
        issues.append(CoherenceIssue(
            issue_type="topic_drift",
            description="Significant vocabulary shift between first and second half of text",
            severity=severity,
        ))

    if structural < threshold:
        severity = "high" if structural < 0.2 else "medium"
        issues.append(CoherenceIssue(
            issue_type="structural",
            description="High variance in sentence lengths reduces readability",
            severity=severity,
        ))

    if contradiction < threshold:
        severity = "high" if contradiction < 0.2 else "medium"
        issues.append(CoherenceIssue(
            issue_type="contradiction",
            description="Contradicting statements detected in nearby sentences",
            severity=severity,
        ))

    if repetition < threshold:
        severity = "high" if repetition < 0.2 else "medium"
        issues.append(CoherenceIssue(
            issue_type="repetition",
            description="Significant sentence-level repetition detected",
            severity=severity,
        ))

    return issues


def _build_recommendations(issues: list[CoherenceIssue]) -> list[str]:
    """Generate actionable recommendations based on detected issues."""
    recommendations: list[str] = []

    issue_types = {issue.issue_type for issue in issues}

    if "topic_drift" in issue_types:
        recommendations.append(
            "Improve topic focus by maintaining consistent vocabulary throughout the response"
        )

    if "structural" in issue_types:
        recommendations.append(
            "Balance sentence lengths for better readability and flow"
        )

    if "contradiction" in issue_types:
        recommendations.append(
            "Review and resolve contradicting statements within the response"
        )

    if "repetition" in issue_types:
        recommendations.append(
            "Reduce repeated sentences by consolidating or removing duplicates"
        )

    return recommendations


class ResponseCoherence:
    """Analyze coherence and internal consistency of LLM responses.

    Measures four dimensions:
    - Topic consistency: vocabulary overlap between text halves (Jaccard)
    - Structural quality: sentence length variance (lower = better)
    - Contradiction score: antonym/negation pattern detection
    - Repetition score: duplicate sentence ratio

    Zero external dependencies.
    """

    def __init__(self) -> None:
        self._total_analyzed = 0
        self._sum_overall = 0.0
        self._grade_counts: dict[str, int] = {}
        self._total_issues = 0

    def analyze(self, text: str) -> CoherenceReport:
        """Run full coherence analysis on text.

        Returns a CoherenceReport with scores, issues, and recommendations.
        """
        sentences = _split_sentences(text)

        topic = _compute_topic_consistency(sentences)
        structural = _compute_structural_quality(sentences)
        contradiction = _compute_contradiction_score(sentences)
        repetition = _compute_repetition_score(sentences)

        overall = _compute_overall(topic, structural, contradiction, repetition)
        grade = _grade_from_score(overall)

        score = CoherenceScore(
            text=text,
            topic_consistency=round(topic, 4),
            structural_quality=round(structural, 4),
            contradiction_score=round(contradiction, 4),
            repetition_score=round(repetition, 4),
            overall=round(overall, 4),
            grade=grade,
        )

        issues = _build_issues(topic, structural, contradiction, repetition)
        recommendations = _build_recommendations(issues)

        self._track_stats(overall, grade, len(issues))

        return CoherenceReport(
            score=score,
            issues=issues,
            recommendations=recommendations,
        )

    def quick_score(self, text: str) -> float:
        """Return just the overall coherence score (0-1)."""
        report = self.analyze(text)
        return report.score.overall

    def analyze_batch(self, texts: list[str]) -> list[CoherenceReport]:
        """Analyze multiple texts and return a list of reports."""
        return [self.analyze(text) for text in texts]

    def stats(self) -> CoherenceStats:
        """Return cumulative analysis statistics."""
        avg = self._sum_overall / self._total_analyzed if self._total_analyzed > 0 else 0.0
        return CoherenceStats(
            total_analyzed=self._total_analyzed,
            avg_overall=round(avg, 4),
            by_grade=dict(self._grade_counts),
            total_issues=self._total_issues,
        )

    def _track_stats(self, overall: float, grade: str, issue_count: int) -> None:
        """Update cumulative statistics after an analysis."""
        self._total_analyzed += 1
        self._sum_overall += overall
        self._grade_counts[grade] = self._grade_counts.get(grade, 0) + 1
        self._total_issues += issue_count
