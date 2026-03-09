"""Parse and classify instructions within prompts.

Identifies instruction types (directive, constraint, context, query, meta),
detects conflicting instructions, and measures instruction density.

Usage:
    from sentinel.instruction_parser import InstructionParser

    parser = InstructionParser()
    result = parser.parse("Write a poem. Never use profanity. What style do you prefer?")
    print(result.instructions)
    print(result.instruction_density)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Instruction:
    """A single classified instruction extracted from text."""
    text: str
    instruction_type: str  # "directive" | "constraint" | "context" | "query" | "meta"
    position: int
    confidence: float


@dataclass
class ConflictPair:
    """A detected conflict between two instructions."""
    instruction_a: Instruction
    instruction_b: Instruction
    conflict_type: str  # "contradiction" | "ambiguity" | "redundancy"
    description: str


@dataclass
class ParseResult:
    """Complete result of parsing a prompt for instructions."""
    text: str
    instructions: list[Instruction]
    conflicts: list[ConflictPair]
    instruction_density: float
    complexity_level: str  # "simple" | "moderate" | "complex"


@dataclass
class ParserStats:
    """Cumulative statistics across all parse calls."""
    total_parsed: int = 0
    total_instructions: int = 0
    total_conflicts: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


_DIRECTIVE_VERBS = {
    "write", "create", "generate", "make", "build",
    "explain", "describe", "analyze", "list", "summarize", "translate",
}

_CONSTRAINT_PATTERNS = [
    r"\bmust\b",
    r"\bshould\b",
    r"\bdon'?t\b",
    r"\bnever\b",
    r"\balways\b",
    r"\bonly\b",
    r"\bat\s+most\b",
    r"\bat\s+least\b",
    r"\bno\s+more\s+than\b",
    r"\bwithin\b",
]

_CONTEXT_PATTERNS = [
    r"\bgiven\s+that\b",
    r"\bassuming\b",
    r"\bin\s+the\s+context\s+of\b",
    r"\bbased\s+on\b",
    r"\bconsidering\b",
]

_QUERY_WORDS = {"what", "how", "why", "when", "where", "who", "which"}

_META_PATTERNS = [
    r"\bignore\b",
    r"\bforget\b",
    r"\boverride\b",
    r"\bsystem\b",
    r"\binstruction\b",
    r"\bprompt\b",
]

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out empty strings."""
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _matches_any_pattern(sentence: str, patterns: list[str]) -> bool:
    lower = sentence.lower()
    return any(re.search(p, lower) for p in patterns)


def _starts_with_directive_verb(sentence: str) -> bool:
    first_word = sentence.strip().split()[0].lower().rstrip(".:,") if sentence.strip().split() else ""
    return first_word in _DIRECTIVE_VERBS


def _is_question(sentence: str) -> bool:
    if "?" in sentence:
        return True
    first_word = sentence.strip().split()[0].lower().rstrip(".:,?") if sentence.strip().split() else ""
    return first_word in _QUERY_WORDS


class InstructionParser:
    """Parse and classify instructions within prompts."""

    def __init__(self) -> None:
        self._stats = ParserStats()

    def parse(self, text: str) -> ParseResult:
        """Parse a prompt for instructions, conflicts, and complexity."""
        sentences = _split_sentences(text)
        instructions = self._classify_sentences(sentences)
        conflicts = self._detect_conflicts(instructions)
        density = self._compute_density(len(instructions), len(sentences))
        complexity = self._compute_complexity(len(instructions))

        self._update_stats(instructions, conflicts)

        return ParseResult(
            text=text,
            instructions=instructions,
            conflicts=conflicts,
            instruction_density=density,
            complexity_level=complexity,
        )

    def parse_batch(self, texts: list[str]) -> list[ParseResult]:
        """Parse multiple prompts."""
        return [self.parse(t) for t in texts]

    def stats(self) -> ParserStats:
        """Return cumulative parsing statistics."""
        return self._stats

    def _classify_sentences(self, sentences: list[str]) -> list[Instruction]:
        instructions: list[Instruction] = []
        for position, sentence in enumerate(sentences):
            classification = self._classify_single(sentence)
            if classification is not None:
                instruction_type, confidence = classification
                instructions.append(Instruction(
                    text=sentence,
                    instruction_type=instruction_type,
                    position=position,
                    confidence=confidence,
                ))
        return instructions

    def _classify_single(self, sentence: str) -> tuple[str, float] | None:
        """Classify a sentence, returning (type, confidence) or None."""
        scores = self._compute_type_scores(sentence)
        if not scores:
            return None
        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best_type, scores[best_type]

    def _compute_type_scores(self, sentence: str) -> dict[str, float]:
        scores: dict[str, float] = {}

        if _matches_any_pattern(sentence, _META_PATTERNS):
            scores["meta"] = self._meta_confidence(sentence)

        if _starts_with_directive_verb(sentence):
            scores["directive"] = 0.9
        elif self._contains_directive_verb(sentence):
            scores["directive"] = 0.6

        if _matches_any_pattern(sentence, _CONSTRAINT_PATTERNS):
            scores["constraint"] = self._constraint_confidence(sentence)

        if _matches_any_pattern(sentence, _CONTEXT_PATTERNS):
            scores["context"] = 0.85

        if _is_question(sentence):
            scores["query"] = 0.95 if "?" in sentence else 0.7

        return scores

    def _contains_directive_verb(self, sentence: str) -> bool:
        words = {w.lower().rstrip(".:,") for w in sentence.split()}
        return bool(words & _DIRECTIVE_VERBS)

    def _constraint_confidence(self, sentence: str) -> float:
        lower = sentence.lower()
        strong_markers = [r"\bnever\b", r"\balways\b", r"\bmust\b"]
        if any(re.search(p, lower) for p in strong_markers):
            return 0.95
        return 0.75

    def _meta_confidence(self, sentence: str) -> float:
        lower = sentence.lower()
        strong_meta = [r"\bignore\b", r"\bforget\b", r"\boverride\b"]
        if any(re.search(p, lower) for p in strong_meta):
            return 0.95
        return 0.6

    def _detect_conflicts(self, instructions: list[Instruction]) -> list[ConflictPair]:
        conflicts: list[ConflictPair] = []
        for i, inst_a in enumerate(instructions):
            for inst_b in instructions[i + 1:]:
                conflict = self._check_pair_for_conflict(inst_a, inst_b)
                if conflict is not None:
                    conflicts.append(conflict)
        return conflicts

    def _check_pair_for_conflict(self, inst_a: Instruction, inst_b: Instruction) -> ConflictPair | None:
        contradiction = self._check_contradiction(inst_a, inst_b)
        if contradiction is not None:
            return contradiction
        redundancy = self._check_redundancy(inst_a, inst_b)
        if redundancy is not None:
            return redundancy
        return None

    def _check_contradiction(self, inst_a: Instruction, inst_b: Instruction) -> ConflictPair | None:
        """Detect 'always X' vs 'never X' contradictions."""
        always_subject = self._extract_after_keyword(inst_a.text, "always")
        never_subject = self._extract_after_keyword(inst_b.text, "never")

        if always_subject and never_subject and self._subjects_overlap(always_subject, never_subject):
            return ConflictPair(
                instruction_a=inst_a,
                instruction_b=inst_b,
                conflict_type="contradiction",
                description=f"'always' conflicts with 'never' regarding '{always_subject}'",
            )

        always_subject = self._extract_after_keyword(inst_b.text, "always")
        never_subject = self._extract_after_keyword(inst_a.text, "never")

        if always_subject and never_subject and self._subjects_overlap(always_subject, never_subject):
            return ConflictPair(
                instruction_a=inst_a,
                instruction_b=inst_b,
                conflict_type="contradiction",
                description=f"'never' conflicts with 'always' regarding '{never_subject}'",
            )

        return None

    def _extract_after_keyword(self, text: str, keyword: str) -> str:
        lower = text.lower()
        match = re.search(rf"\b{keyword}\b\s+(.*)", lower)
        if match:
            return match.group(1).strip().rstrip(".")
        return ""

    def _subjects_overlap(self, subject_a: str, subject_b: str) -> bool:
        filler = {"use", "be", "have", "do", "make", "get", "go", "take", "give", "it", "the", "a", "an"}
        words_a = set(subject_a.lower().split()) - filler
        words_b = set(subject_b.lower().split()) - filler
        if not words_a or not words_b:
            return False
        common = words_a & words_b
        if not common:
            return False
        smaller_set = min(len(words_a), len(words_b))
        return len(common) / smaller_set >= 0.5 if smaller_set > 0 else False

    def _check_redundancy(self, inst_a: Instruction, inst_b: Instruction) -> ConflictPair | None:
        """Detect same-type instructions with overlapping content words."""
        if inst_a.instruction_type != inst_b.instruction_type:
            return None

        overlap = self._keyword_overlap(inst_a.text, inst_b.text)
        if overlap < 0.5:
            return None

        return ConflictPair(
            instruction_a=inst_a,
            instruction_b=inst_b,
            conflict_type="redundancy",
            description=f"Similar {inst_a.instruction_type} instructions (overlap: {overlap:.0%})",
        )

    def _keyword_overlap(self, text_a: str, text_b: str) -> float:
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and", "in", "on", "for", "it", "this", "that"}
        words_a = {w.lower().strip(".,!?;:") for w in text_a.split()} - stopwords
        words_b = {w.lower().strip(".,!?;:") for w in text_b.split()} - stopwords
        if not words_a or not words_b:
            return 0.0
        common = words_a & words_b
        smaller = min(len(words_a), len(words_b))
        return len(common) / smaller if smaller > 0 else 0.0

    def _compute_density(self, instruction_count: int, sentence_count: int) -> float:
        if sentence_count == 0:
            return 0.0
        return round(instruction_count / sentence_count, 4)

    def _compute_complexity(self, instruction_count: int) -> str:
        if instruction_count <= 2:
            return "simple"
        if instruction_count <= 5:
            return "moderate"
        return "complex"

    def _update_stats(self, instructions: list[Instruction], conflicts: list[ConflictPair]) -> None:
        self._stats.total_parsed += 1
        self._stats.total_instructions += len(instructions)
        self._stats.total_conflicts += len(conflicts)
        for inst in instructions:
            self._stats.by_type[inst.instruction_type] = self._stats.by_type.get(inst.instruction_type, 0) + 1
