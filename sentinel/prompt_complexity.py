"""Prompt complexity analyzer for intelligent model routing.

Analyze prompt complexity to route simple prompts to cheaper/faster
models and complex prompts to more capable ones.

Usage:
    from sentinel.prompt_complexity import PromptComplexity

    analyzer = PromptComplexity()
    score = analyzer.analyze("Explain quantum computing and compare it to classical computing")
    print(score.level)          # "complex"
    print(analyzer.suggest_model("What is 2+2?"))  # "fast"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ComplexityScore:
    """Full complexity analysis of a prompt."""
    prompt: str
    word_count: int
    sentence_count: int
    avg_word_length: float
    vocabulary_diversity: float
    structural_complexity: float
    instruction_density: float
    overall: float
    level: str


@dataclass
class ComplexityComparison:
    """Side-by-side comparison of two prompts."""
    prompt_a_score: float
    prompt_b_score: float
    delta: float
    recommendation: str


@dataclass
class ComplexityStats:
    """Cumulative analysis statistics."""
    total_analyzed: int = 0
    by_level: dict[str, int] = field(default_factory=dict)
    avg_overall: float = 0.0


_INSTRUCTION_WORDS = frozenset({
    "analyze", "explain", "compare", "create", "implement", "design",
    "evaluate", "synthesize", "summarize", "describe", "list", "calculate",
    "assess", "develop", "formulate", "integrate", "justify", "critique",
    "propose", "recommend", "classify", "differentiate", "illustrate",
    "interpret", "predict", "validate", "construct", "debug", "refactor",
    "optimize", "generate", "translate", "transform", "review", "outline",
})

_LIST_BULLET_PATTERN = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)]\s)")
_CODE_BLOCK_PATTERN = re.compile(r"```")
_NESTED_INSTRUCTION_PATTERN = re.compile(
    r"(?i)\b(?:then|after that|next|finally|additionally|moreover|furthermore)\b"
)


def _extract_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
    return max(1, sum(1 for s in sentences if s.strip()))


class PromptComplexity:
    """Analyze prompt complexity for model routing decisions.

    Classifies prompts into four levels (simple, moderate, complex,
    expert) based on vocabulary diversity, structural complexity,
    instruction density, and normalized length.
    """

    def __init__(
        self,
        simple_threshold: float = 0.3,
        complex_threshold: float = 0.6,
        expert_threshold: float = 0.8,
    ) -> None:
        self._simple_threshold = simple_threshold
        self._complex_threshold = complex_threshold
        self._expert_threshold = expert_threshold
        self._history: list[ComplexityScore] = []

    def analyze(self, prompt: str) -> ComplexityScore:
        """Analyze a prompt and return its complexity score."""
        words = _extract_words(prompt)
        word_count = len(words)
        sentence_count = _count_sentences(prompt)
        avg_word_length = self._compute_avg_word_length(words)
        vocabulary_diversity = self._compute_vocabulary_diversity(words)
        structural_complexity = self._compute_structural_complexity(prompt, sentence_count)
        instruction_density = self._compute_instruction_density(words)
        normalized_length = self._compute_normalized_length(word_count)
        overall = self._compute_overall(
            vocabulary_diversity, structural_complexity,
            instruction_density, normalized_length,
        )
        level = self._classify_level(overall)

        score = ComplexityScore(
            prompt=prompt,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=round(avg_word_length, 4),
            vocabulary_diversity=round(vocabulary_diversity, 4),
            structural_complexity=round(structural_complexity, 4),
            instruction_density=round(instruction_density, 4),
            overall=round(overall, 4),
            level=level,
        )
        self._history.append(score)
        return score

    def compare(self, prompt_a: str, prompt_b: str) -> ComplexityComparison:
        """Compare the complexity of two prompts."""
        score_a = self.analyze(prompt_a)
        score_b = self.analyze(prompt_b)
        delta = round(score_a.overall - score_b.overall, 4)
        recommendation = self._build_comparison_recommendation(score_a, score_b, delta)
        return ComplexityComparison(
            prompt_a_score=score_a.overall,
            prompt_b_score=score_b.overall,
            delta=delta,
            recommendation=recommendation,
        )

    def analyze_batch(self, prompts: list[str]) -> list[ComplexityScore]:
        """Analyze multiple prompts in a batch."""
        return [self.analyze(p) for p in prompts]

    def suggest_model(self, prompt: str) -> str:
        """Suggest a model tier based on prompt complexity."""
        score = self.analyze(prompt)
        return self._model_tier_for_level(score.level)

    def stats(self) -> ComplexityStats:
        """Return cumulative analysis statistics."""
        total = len(self._history)
        if total == 0:
            return ComplexityStats()

        by_level: dict[str, int] = {}
        overall_sum = 0.0
        for score in self._history:
            by_level[score.level] = by_level.get(score.level, 0) + 1
            overall_sum += score.overall

        return ComplexityStats(
            total_analyzed=total,
            by_level=by_level,
            avg_overall=round(overall_sum / total, 4),
        )

    # --- Private helpers ---

    def _compute_avg_word_length(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _compute_vocabulary_diversity(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _compute_structural_complexity(self, text: str, sentence_count: int) -> float:
        signals = 0.0
        total_checks = 4

        # Sentence variety: more sentences = more structure
        if sentence_count >= 5:
            signals += 1.0
        elif sentence_count >= 3:
            signals += 0.5

        # Lists or bullet points
        if _LIST_BULLET_PATTERN.search(text):
            signals += 1.0

        # Code blocks
        code_block_count = len(_CODE_BLOCK_PATTERN.findall(text))
        if code_block_count >= 2:
            signals += 1.0
        elif code_block_count >= 1:
            signals += 0.5

        # Nested/sequential instructions
        nested_matches = len(_NESTED_INSTRUCTION_PATTERN.findall(text))
        if nested_matches >= 3:
            signals += 1.0
        elif nested_matches >= 1:
            signals += 0.5

        return min(1.0, signals / total_checks)

    def _compute_instruction_density(self, words: list[str]) -> float:
        if not words:
            return 0.0
        instruction_count = sum(1 for w in words if w in _INSTRUCTION_WORDS)
        raw_density = instruction_count / len(words)
        # Scale so that ~10% instruction words maps to ~1.0
        return min(1.0, raw_density * 10)

    def _compute_normalized_length(self, word_count: int) -> float:
        # 200+ words is considered maximum complexity for length
        return min(1.0, word_count / 200)

    def _compute_overall(
        self,
        vocabulary_diversity: float,
        structural_complexity: float,
        instruction_density: float,
        normalized_length: float,
    ) -> float:
        return (
            0.2 * vocabulary_diversity
            + 0.3 * structural_complexity
            + 0.3 * instruction_density
            + 0.2 * normalized_length
        )

    def _classify_level(self, overall: float) -> str:
        if overall >= self._expert_threshold:
            return "expert"
        if overall >= self._complex_threshold:
            return "complex"
        if overall >= self._simple_threshold:
            return "moderate"
        return "simple"

    def _model_tier_for_level(self, level: str) -> str:
        tiers = {
            "simple": "fast",
            "moderate": "balanced",
            "complex": "capable",
            "expert": "frontier",
        }
        return tiers[level]

    def _build_comparison_recommendation(
        self,
        score_a: ComplexityScore,
        score_b: ComplexityScore,
        delta: float,
    ) -> str:
        if abs(delta) < 0.05:
            return "Both prompts have similar complexity; same model tier is appropriate."
        if delta > 0:
            return (
                f"Prompt A is more complex ({score_a.level}) than Prompt B ({score_b.level}). "
                f"Consider a more capable model for Prompt A."
            )
        return (
            f"Prompt B is more complex ({score_b.level}) than Prompt A ({score_a.level}). "
            f"Consider a more capable model for Prompt B."
        )
