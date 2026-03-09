"""Multi-dimensional LLM response quality assessment.

Score responses across helpfulness, completeness, clarity,
accuracy signals, formatting, and conciseness. Produces weighted
overall scores with letter grades.

Usage:
    from sentinel.response_quality import ResponseQuality

    rq = ResponseQuality()
    result = rq.assess("The capital of France is Paris. It is located...")
    print(result.overall)  # 0.82
    print(result.grade)    # "A"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QualityDimension:
    """Score for a single quality dimension."""
    name: str
    score: float
    weight: float = 1.0
    details: str = ""


@dataclass
class QualityAssessment:
    """Full quality assessment of a response."""
    text: str
    dimensions: list[QualityDimension]
    overall: float
    grade: str
    word_count: int
    sentence_count: int


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    min_length: int = 10
    ideal_min_words: int = 20
    ideal_max_words: int = 500
    check_formatting: bool = True


@dataclass
class QualityStats:
    """Cumulative quality assessment statistics."""
    total_assessed: int = 0
    avg_overall: float = 0.0
    by_grade: dict[str, int] = field(default_factory=dict)


_CONCLUSION_WORDS = frozenset([
    "therefore", "thus", "consequently", "in conclusion", "overall",
    "in summary", "to summarize", "finally", "hence", "as a result",
])

_TRANSITION_WORDS = frozenset([
    "however", "moreover", "furthermore", "additionally", "nevertheless",
    "meanwhile", "although", "because", "therefore", "consequently",
    "specifically", "for example", "in contrast", "on the other hand",
    "similarly", "likewise", "in addition", "as a result", "first",
    "second", "third", "next", "finally",
])

_FILLER_WORDS = frozenset([
    "very", "really", "just", "basically", "actually", "literally",
    "simply", "quite", "rather", "somewhat", "perhaps", "maybe",
    "possibly", "certainly", "definitely", "probably", "essentially",
])

_VAGUE_WORDS = frozenset([
    "thing", "things", "stuff", "something", "somehow", "somewhat",
    "various", "several", "many", "some", "kind of", "sort of",
])

_ACTION_VERBS = frozenset([
    "create", "build", "implement", "configure", "install", "run",
    "execute", "define", "set", "use", "apply", "enable", "disable",
    "add", "remove", "update", "check", "verify", "ensure", "follow",
    "start", "stop", "open", "close", "select", "click", "navigate",
])


def _count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r'[.!?]+', text)
    return [s.strip() for s in raw if s.strip()]


def _words_lower(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'\b\w+\b', text)]


def _compute_grade(overall: float) -> str:
    if overall >= 0.8:
        return "A"
    if overall >= 0.6:
        return "B"
    if overall >= 0.4:
        return "C"
    if overall >= 0.2:
        return "D"
    return "F"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class ResponseQuality:
    """Assess LLM response quality across multiple dimensions."""

    def __init__(self, config: QualityConfig | None = None) -> None:
        self._config = config or QualityConfig()
        self._history: list[QualityAssessment] = []

    def assess(self, text: str) -> QualityAssessment:
        """Assess response quality across five dimensions."""
        word_count = _count_words(text)
        sentences = _split_sentences(text)
        sentence_count = len(sentences)
        words = _words_lower(text)

        if not text.strip():
            dimensions = self._empty_dimensions()
            overall = 0.0
            grade = "F"
        else:
            dimensions = [
                self._score_completeness(word_count, sentences, text),
                self._score_clarity(sentences, text),
                self._score_helpfulness(words, text),
                self._score_formatting(text, word_count),
                self._score_conciseness(words, word_count),
            ]
            overall = self._weighted_average(dimensions)
            grade = _compute_grade(overall)

        assessment = QualityAssessment(
            text=text,
            dimensions=dimensions,
            overall=round(overall, 4),
            grade=grade,
            word_count=word_count,
            sentence_count=sentence_count,
        )
        self._history.append(assessment)
        return assessment

    def assess_batch(self, texts: list[str]) -> list[QualityAssessment]:
        """Assess multiple responses."""
        return [self.assess(text) for text in texts]

    def compare(self, text_a: str, text_b: str) -> dict:
        """Compare two responses and return which is better."""
        assessment_a = self.assess(text_a)
        assessment_b = self.assess(text_b)
        diff = round(assessment_a.overall - assessment_b.overall, 4)

        if diff > 0:
            winner = "a"
        elif diff < 0:
            winner = "b"
        else:
            winner = "tie"

        return {
            "winner": winner,
            "a": assessment_a,
            "b": assessment_b,
            "difference": abs(diff),
        }

    def stats(self) -> QualityStats:
        """Return cumulative assessment statistics."""
        total = len(self._history)
        if total == 0:
            return QualityStats()

        avg = round(sum(a.overall for a in self._history) / total, 4)
        by_grade: dict[str, int] = {}
        for assessment in self._history:
            by_grade[assessment.grade] = by_grade.get(assessment.grade, 0) + 1

        return QualityStats(
            total_assessed=total,
            avg_overall=avg,
            by_grade=by_grade,
        )

    @staticmethod
    def _empty_dimensions() -> list[QualityDimension]:
        return [
            QualityDimension(name="completeness", score=0.0, weight=1.5, details="empty"),
            QualityDimension(name="clarity", score=0.0, weight=1.0, details="empty"),
            QualityDimension(name="helpfulness", score=0.0, weight=1.5, details="empty"),
            QualityDimension(name="formatting", score=0.0, weight=0.5, details="empty"),
            QualityDimension(name="conciseness", score=0.0, weight=1.0, details="empty"),
        ]

    # -- Dimension scorers --

    def _score_completeness(
        self, word_count: int, sentences: list[str], text: str,
    ) -> QualityDimension:
        score = 0.0
        details_parts: list[str] = []

        length_score = self._score_word_count_range(word_count)
        score += length_score * 0.5
        details_parts.append(f"length={length_score:.2f}")

        variety_score = self._score_sentence_variety(sentences)
        score += variety_score * 0.3
        details_parts.append(f"variety={variety_score:.2f}")

        conclusion_score = self._score_conclusion_presence(text)
        score += conclusion_score * 0.2
        details_parts.append(f"conclusion={conclusion_score:.2f}")

        return QualityDimension(
            name="completeness",
            score=round(_clamp(score), 4),
            weight=1.5,
            details=", ".join(details_parts),
        )

    def _score_clarity(
        self, sentences: list[str], text: str,
    ) -> QualityDimension:
        score = 0.0
        details_parts: list[str] = []

        sentence_length_score = self._score_sentence_length(sentences)
        score += sentence_length_score * 0.4
        details_parts.append(f"sent_len={sentence_length_score:.2f}")

        paragraph_score = self._score_paragraph_breaks(text)
        score += paragraph_score * 0.3
        details_parts.append(f"paragraphs={paragraph_score:.2f}")

        transition_score = self._score_transition_words(text)
        score += transition_score * 0.3
        details_parts.append(f"transitions={transition_score:.2f}")

        return QualityDimension(
            name="clarity",
            score=round(_clamp(score), 4),
            weight=1.0,
            details=", ".join(details_parts),
        )

    def _score_helpfulness(
        self, words: list[str], text: str,
    ) -> QualityDimension:
        score = 0.0
        details_parts: list[str] = []

        action_score = self._score_actionable_content(words, text)
        score += action_score * 0.5
        details_parts.append(f"actionable={action_score:.2f}")

        specificity_score = self._score_specificity(words)
        score += specificity_score * 0.5
        details_parts.append(f"specificity={specificity_score:.2f}")

        return QualityDimension(
            name="helpfulness",
            score=round(_clamp(score), 4),
            weight=1.5,
            details=", ".join(details_parts),
        )

    def _score_formatting(
        self, text: str, word_count: int,
    ) -> QualityDimension:
        if not self._config.check_formatting:
            return QualityDimension(
                name="formatting", score=1.0, weight=0.5,
                details="formatting checks disabled",
            )

        score = 0.0
        details_parts: list[str] = []

        cap_score = self._score_capitalization(text)
        score += cap_score * 0.3
        details_parts.append(f"capitalization={cap_score:.2f}")

        punct_score = self._score_punctuation(text)
        score += punct_score * 0.3
        details_parts.append(f"punctuation={punct_score:.2f}")

        structure_score = self._score_structure(text, word_count)
        score += structure_score * 0.4
        details_parts.append(f"structure={structure_score:.2f}")

        return QualityDimension(
            name="formatting",
            score=round(_clamp(score), 4),
            weight=0.5,
            details=", ".join(details_parts),
        )

    def _score_conciseness(
        self, words: list[str], word_count: int,
    ) -> QualityDimension:
        score = 0.0
        details_parts: list[str] = []

        length_penalty = self._score_excessive_length(word_count)
        score += length_penalty * 0.4
        details_parts.append(f"length_ok={length_penalty:.2f}")

        filler_penalty = self._score_filler_ratio(words)
        score += filler_penalty * 0.3
        details_parts.append(f"low_filler={filler_penalty:.2f}")

        repetition_penalty = self._score_repetition(words)
        score += repetition_penalty * 0.3
        details_parts.append(f"low_repetition={repetition_penalty:.2f}")

        return QualityDimension(
            name="conciseness",
            score=round(_clamp(score), 4),
            weight=1.0,
            details=", ".join(details_parts),
        )

    # -- Sub-scorers --

    def _score_word_count_range(self, word_count: int) -> float:
        ideal_min = self._config.ideal_min_words
        ideal_max = self._config.ideal_max_words
        if ideal_min <= word_count <= ideal_max:
            return 1.0
        if word_count < ideal_min:
            return max(0.0, word_count / ideal_min)
        overshoot = (word_count - ideal_max) / ideal_max
        return max(0.0, 1.0 - overshoot * 0.5)

    def _score_sentence_variety(self, sentences: list[str]) -> float:
        if len(sentences) <= 1:
            return 0.3
        lengths = [_count_words(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        if avg_len == 0:
            return 0.0
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / avg_len
        if coefficient_of_variation < 0.1:
            return 0.5
        if coefficient_of_variation > 0.8:
            return 0.6
        return min(1.0, 0.5 + coefficient_of_variation)

    def _score_conclusion_presence(self, text: str) -> float:
        text_lower = text.lower()
        for word in _CONCLUSION_WORDS:
            if word in text_lower:
                return 1.0
        return 0.0

    def _score_sentence_length(self, sentences: list[str]) -> float:
        if not sentences:
            return 0.0
        lengths = [_count_words(s) for s in sentences]
        avg = sum(lengths) / len(lengths)
        if 10 <= avg <= 25:
            return 1.0
        if avg < 5:
            return 0.3
        if avg < 10:
            return 0.5 + (avg - 5) * 0.1
        if avg <= 35:
            return max(0.3, 1.0 - (avg - 25) * 0.07)
        return 0.2

    def _score_paragraph_breaks(self, text: str) -> float:
        if not text.strip():
            return 0.0
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        word_count = _count_words(text)
        if word_count < 50:
            return 0.8
        if len(paragraphs) >= 2:
            return 1.0
        if "\n" in text.strip():
            return 0.7
        return 0.3

    def _score_transition_words(self, text: str) -> float:
        text_lower = text.lower()
        found = sum(1 for w in _TRANSITION_WORDS if w in text_lower)
        if found == 0:
            return 0.2
        if found >= 5:
            return 1.0
        return min(1.0, 0.2 + found * 0.2)

    def _score_actionable_content(self, words: list[str], text: str) -> float:
        word_set = set(words)
        action_count = len(word_set & _ACTION_VERBS)

        has_examples = bool(re.search(
            r'(?i)\b(?:for example|e\.g\.|such as|for instance)\b', text,
        ))
        has_steps = bool(re.search(r'(?:^|\n)\s*(?:\d+[\.\):]|[-*])\s', text))
        has_list = bool(re.search(r'(?:^|\n)\s*[-*]\s', text))

        score = min(1.0, action_count * 0.15)
        if has_examples:
            score += 0.25
        if has_steps:
            score += 0.25
        if has_list:
            score += 0.15
        return min(1.0, score)

    def _score_specificity(self, words: list[str]) -> float:
        if not words:
            return 0.0
        word_set = set(words)
        vague_count = len(word_set & _VAGUE_WORDS)
        filler_count = len(word_set & _FILLER_WORDS)
        penalty = (vague_count + filler_count) / max(1, len(word_set))
        return _clamp(1.0 - penalty * 3.0)

    def _score_capitalization(self, text: str) -> float:
        sentences = _split_sentences(text)
        if not sentences:
            return 0.0
        capitalized = sum(1 for s in sentences if s and s[0].isupper())
        return capitalized / len(sentences)

    def _score_punctuation(self, text: str) -> float:
        stripped = text.rstrip()
        if not stripped:
            return 0.0
        if stripped[-1] in ".!?":
            return 1.0
        if stripped[-1] in ":;,":
            return 0.5
        return 0.2

    def _score_structure(self, text: str, word_count: int) -> float:
        if word_count < 50:
            return 0.8
        has_headers = bool(re.search(r'(?:^|\n)#{1,6}\s', text))
        has_lists = bool(re.search(r'(?:^|\n)\s*[-*]\s', text))
        has_numbered = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', text))
        score = 0.4
        if has_headers:
            score += 0.3
        if has_lists or has_numbered:
            score += 0.3
        return min(1.0, score)

    def _score_excessive_length(self, word_count: int) -> float:
        ideal_max = self._config.ideal_max_words
        if word_count <= ideal_max:
            return 1.0
        ratio = word_count / ideal_max
        if ratio <= 1.5:
            return 0.7
        if ratio <= 2.0:
            return 0.4
        return 0.2

    def _score_filler_ratio(self, words: list[str]) -> float:
        if not words:
            return 1.0
        filler_count = sum(1 for w in words if w in _FILLER_WORDS)
        ratio = filler_count / len(words)
        if ratio < 0.02:
            return 1.0
        if ratio < 0.05:
            return 0.8
        if ratio < 0.1:
            return 0.5
        return 0.2

    def _score_repetition(self, words: list[str]) -> float:
        if len(words) < 5:
            return 1.0
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if not bigrams:
            return 1.0
        unique_ratio = len(set(bigrams)) / len(bigrams)
        if unique_ratio > 0.9:
            return 1.0
        if unique_ratio > 0.7:
            return 0.7
        return 0.3

    @staticmethod
    def _weighted_average(dimensions: list[QualityDimension]) -> float:
        total_weight = sum(d.weight for d in dimensions)
        if total_weight == 0:
            return 0.0
        return sum(d.score * d.weight for d in dimensions) / total_weight
