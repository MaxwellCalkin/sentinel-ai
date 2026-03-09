"""Semantic validation between prompts and LLM responses.

Detects when an LLM response doesn't match the intent of the prompt,
including off-topic answers, refusal disguises, hallucinated topics,
question dodges, and partial answers.

Usage:
    from sentinel.semantic_validator import SemanticValidator

    validator = SemanticValidator()
    result = validator.validate(
        "What is photosynthesis?",
        "Photosynthesis is the process by which plants convert sunlight into energy."
    )
    assert result.is_valid
    assert result.grade in ("A", "B")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "or", "but", "nor", "not", "so", "yet", "both", "either",
    "if", "then", "than", "because", "as", "while", "when", "where",
    "what", "which", "who", "whom", "how", "why",
    "all", "each", "every", "any", "some", "no", "other", "such",
    "just", "also", "very", "much", "more", "most", "only", "own",
    "same", "too", "even", "still", "already",
    "up", "out", "off", "over", "under", "again", "further",
    "here", "there", "once", "twice",
})

_QUESTION_WORDS = frozenset({"what", "how", "why", "when", "where", "who", "which"})

_INSTRUCTION_WORDS = frozenset({
    "explain", "describe", "list", "compare", "define", "summarize",
    "analyze", "discuss", "outline", "evaluate", "calculate", "tell",
})

_REFUSAL_PATTERNS = [
    re.compile(r"\bi (?:cannot|can't|am unable|am not able)\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bi'?m (?:unable|not able|not capable)\b", re.I),
    re.compile(r"\bi (?:don't|do not) have (?:the ability|access|capability)\b", re.I),
    re.compile(r"\bit(?:'s| is) (?:not possible|beyond my)\b", re.I),
    re.compile(r"\bi (?:must|have to) (?:decline|refuse)\b", re.I),
]

_DIRECT_REFUSAL_PATTERNS = [
    re.compile(r"\bi can'?t help\b", re.I),
    re.compile(r"\bi'?m sorry,? (?:but )?i can'?t\b", re.I),
]

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")


@dataclass
class SemanticMatch:
    """Scores for semantic alignment between prompt and response."""
    prompt: str
    response: str
    relevance_score: float
    topic_overlap: float
    intent_alignment: float
    overall: float


@dataclass
class SemanticIssue:
    """A detected semantic issue in the response."""
    issue_type: str
    confidence: float
    description: str


@dataclass
class ValidationResult:
    """Complete validation result for a prompt-response pair."""
    match: SemanticMatch
    issues: list[SemanticIssue]
    is_valid: bool
    grade: str


@dataclass
class ValidatorStats:
    """Cumulative statistics for the validator."""
    total_validated: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    by_issue: dict[str, int] = field(default_factory=dict)
    avg_overall: float = 0.0


def _extract_words(text: str) -> list[str]:
    """Extract lowercase alphabetic words from text."""
    return [w.lower() for w in _WORD_PATTERN.findall(text)]


def _significant_words(words: list[str]) -> set[str]:
    """Filter out stopwords to get content-bearing words."""
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _topic_words(words: list[str]) -> set[str]:
    """Extract noun-like words (longer than 4 characters)."""
    return {w for w in words if w not in _STOPWORDS and len(w) > 4}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _compute_relevance(prompt_words: set[str], response_words: set[str]) -> float:
    return _jaccard_similarity(prompt_words, response_words)


def _compute_topic_overlap(prompt_topics: set[str], response_topics: set[str]) -> float:
    if not prompt_topics:
        return 0.0
    if not response_topics:
        return 0.0
    return _jaccard_similarity(prompt_topics, response_topics)


def _compute_intent_alignment(prompt_raw_words: list[str], response_raw_words: list[str]) -> float:
    prompt_lower = {w.lower() for w in prompt_raw_words}

    question_words_in_prompt = prompt_lower & _QUESTION_WORDS
    instruction_words_in_prompt = prompt_lower & _INSTRUCTION_WORDS

    if not question_words_in_prompt and not instruction_words_in_prompt:
        return 0.5

    response_significant = _significant_words(response_raw_words)
    prompt_significant = _significant_words(prompt_raw_words)

    if not prompt_significant:
        return 0.5

    overlap = response_significant & prompt_significant
    coverage = len(overlap) / len(prompt_significant) if prompt_significant else 0.0
    return min(1.0, coverage * 1.5)


def _has_refusal_disguise(response: str) -> bool:
    """Detect refusal language that isn't a direct refusal."""
    has_indirect = any(p.search(response) for p in _REFUSAL_PATTERNS)
    has_direct = any(p.search(response) for p in _DIRECT_REFUSAL_PATTERNS)
    return has_indirect and not has_direct


def _has_question(text: str) -> bool:
    """Check if text contains a question."""
    words = {w.lower() for w in _WORD_PATTERN.findall(text)}
    return bool(words & _QUESTION_WORDS) or "?" in text


def _compute_overall(relevance: float, topic: float, intent: float) -> float:
    return round(0.4 * relevance + 0.3 * topic + 0.3 * intent, 4)


def _assign_grade(overall: float) -> str:
    if overall >= 0.8:
        return "A"
    if overall >= 0.6:
        return "B"
    if overall >= 0.4:
        return "C"
    if overall >= 0.2:
        return "D"
    return "F"


class SemanticValidator:
    """Validate semantic consistency between prompts and LLM responses.

    Detects off-topic answers, refusal disguises, hallucinated topics,
    question dodges, and partial answers using word-overlap heuristics.
    """

    def __init__(self, min_relevance: float = 0.3) -> None:
        self._min_relevance = min_relevance
        self._total_validated = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._issue_counts: dict[str, int] = {}
        self._overall_sum = 0.0

    def validate(self, prompt: str, response: str) -> ValidationResult:
        """Validate a prompt-response pair for semantic consistency."""
        match = self._compute_match(prompt, response)
        issues = self._detect_issues(prompt, response, match)
        is_valid = self._determine_validity(match, issues)
        grade = _assign_grade(match.overall)

        self._update_stats(is_valid, issues, match.overall)

        return ValidationResult(
            match=match,
            issues=issues,
            is_valid=is_valid,
            grade=grade,
        )

    def validate_batch(self, pairs: list[tuple[str, str]]) -> list[ValidationResult]:
        """Validate multiple prompt-response pairs."""
        return [self.validate(prompt, response) for prompt, response in pairs]

    def stats(self) -> ValidatorStats:
        """Return cumulative validation statistics."""
        avg = self._overall_sum / self._total_validated if self._total_validated else 0.0
        return ValidatorStats(
            total_validated=self._total_validated,
            valid_count=self._valid_count,
            invalid_count=self._invalid_count,
            by_issue=dict(self._issue_counts),
            avg_overall=round(avg, 4),
        )

    def _compute_match(self, prompt: str, response: str) -> SemanticMatch:
        prompt_raw = _extract_words(prompt)
        response_raw = _extract_words(response)

        prompt_significant = _significant_words(prompt_raw)
        response_significant = _significant_words(response_raw)

        prompt_topics = _topic_words(prompt_raw)
        response_topics = _topic_words(response_raw)

        relevance = _compute_relevance(prompt_significant, response_significant)
        topic = _compute_topic_overlap(prompt_topics, response_topics)
        intent = _compute_intent_alignment(prompt_raw, response_raw)
        overall = _compute_overall(relevance, topic, intent)

        return SemanticMatch(
            prompt=prompt,
            response=response,
            relevance_score=round(relevance, 4),
            topic_overlap=round(topic, 4),
            intent_alignment=round(intent, 4),
            overall=round(overall, 4),
        )

    def _detect_issues(
        self, prompt: str, response: str, match: SemanticMatch
    ) -> list[SemanticIssue]:
        issues: list[SemanticIssue] = []

        if match.relevance_score < 0.1:
            issues.append(SemanticIssue(
                issue_type="off_topic",
                confidence=0.9,
                description="Response has very low word overlap with the prompt",
            ))
        elif match.topic_overlap < 0.15:
            issues.append(SemanticIssue(
                issue_type="partial_answer",
                confidence=0.6,
                description="Response partially addresses the prompt but misses key topics",
            ))

        if _has_refusal_disguise(response):
            issues.append(SemanticIssue(
                issue_type="refusal_disguise",
                confidence=0.8,
                description="Response contains refusal language disguised as a helpful answer",
            ))

        if self._has_topic_hallucination(prompt, response):
            issues.append(SemanticIssue(
                issue_type="topic_hallucination",
                confidence=0.7,
                description="Response introduces many topics not present in the prompt",
            ))

        if self._is_question_dodge(prompt, match):
            issues.append(SemanticIssue(
                issue_type="question_dodge",
                confidence=0.7,
                description="Response does not address the question asked in the prompt",
            ))

        return issues

    def _has_topic_hallucination(self, prompt: str, response: str) -> bool:
        response_words = _significant_words(_extract_words(response))
        prompt_words = _significant_words(_extract_words(prompt))

        if len(response_words) < 3:
            return False

        new_words = response_words - prompt_words
        new_ratio = len(new_words) / len(response_words)
        return new_ratio > 0.80

    def _is_question_dodge(self, prompt: str, match: SemanticMatch) -> bool:
        if not _has_question(prompt):
            return False
        return match.intent_alignment < 0.2

    def _determine_validity(
        self, match: SemanticMatch, issues: list[SemanticIssue]
    ) -> bool:
        if match.overall < self._min_relevance:
            return False

        high_confidence_issues = [i for i in issues if i.confidence >= 0.8]
        return len(high_confidence_issues) == 0

    def _update_stats(
        self, is_valid: bool, issues: list[SemanticIssue], overall: float
    ) -> None:
        self._total_validated += 1
        self._overall_sum += overall

        if is_valid:
            self._valid_count += 1
        else:
            self._invalid_count += 1

        for issue in issues:
            self._issue_counts[issue.issue_type] = (
                self._issue_counts.get(issue.issue_type, 0) + 1
            )
