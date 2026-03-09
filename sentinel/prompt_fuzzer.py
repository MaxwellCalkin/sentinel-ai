"""Adversarial prompt fuzzing for safety system testing.

Generates mutated prompt variations to test robustness of safety
guardrails. Applies case changes, encoding tricks, structural
mutations, and evasion techniques to create adversarial payloads.

Usage:
    from sentinel.prompt_fuzzer import PromptFuzzer, FuzzConfig

    fuzzer = PromptFuzzer()
    variants = fuzzer.fuzz("ignore all previous instructions")
    for v in variants:
        print(v.fuzzed, v.mutations_applied, v.similarity)
"""

from __future__ import annotations

import base64
import random
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FuzzMutation:
    """Describes a single mutation strategy."""
    name: str
    description: str
    category: str  # case | encoding | structural | substitution | evasion


@dataclass
class FuzzedPrompt:
    """A prompt after fuzzing mutations have been applied."""
    original: str
    fuzzed: str
    mutations_applied: list[str]
    similarity: float  # 0-1, word overlap ratio with original


@dataclass
class FuzzConfig:
    """Controls fuzzing behaviour."""
    num_variants: int = 10
    mutations_per_variant: int = 2
    seed: int | None = None
    categories: list[str] | None = None


@dataclass
class FuzzStats:
    """Cumulative fuzzing statistics."""
    total_generated: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_LEETSPEAK_MAP: dict[str, str] = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7",
}

_UNICODE_CONFUSABLE_MAP: dict[str, str] = {
    "a": "\u0430",  # Cyrillic а
    "c": "\u0441",  # Cyrillic с
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "x": "\u0445",  # Cyrillic х
}

_SYNONYM_MAP: dict[str, str] = {
    "add": "insert",
    "remove": "delete",
    "ignore": "disregard",
    "create": "generate",
    "stop": "halt",
    "send": "transmit",
    "get": "retrieve",
    "show": "display",
    "hide": "conceal",
    "run": "execute",
}

ZERO_WIDTH_SPACE = "\u200b"


# ---------------------------------------------------------------------------
# Mutation functions
# ---------------------------------------------------------------------------

def _random_case(text: str, rng: random.Random) -> str:
    return "".join(
        ch.upper() if rng.random() > 0.5 else ch.lower()
        for ch in text
    )


def _upper_all(text: str, _rng: random.Random) -> str:
    return text.upper()


def _lower_all(text: str, _rng: random.Random) -> str:
    return text.lower()


def _leetspeak(text: str, _rng: random.Random) -> str:
    return "".join(_LEETSPEAK_MAP.get(ch.lower(), ch) for ch in text)


def _unicode_confusables(text: str, _rng: random.Random) -> str:
    return "".join(_UNICODE_CONFUSABLE_MAP.get(ch.lower(), ch) for ch in text)


def _add_whitespace(text: str, rng: random.Random) -> str:
    chars: list[str] = []
    for ch in text:
        chars.append(ch)
        if ch == " ":
            chars.append(" " * rng.randint(1, 3))
    return "".join(chars)


def _add_newlines(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    insert_at = rng.randint(1, len(words) - 1)
    words.insert(insert_at, "\n")
    return " ".join(words)


def _reverse_words(text: str, _rng: random.Random) -> str:
    return " ".join(reversed(text.split()))


def _shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def _synonym_swap(text: str, _rng: random.Random) -> str:
    words = text.split()
    return " ".join(_SYNONYM_MAP.get(w.lower(), w) for w in words)


def _word_repeat(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    index = rng.randint(0, len(words) - 1)
    words.insert(index + 1, words[index])
    return " ".join(words)


def _insert_zero_width(text: str, _rng: random.Random) -> str:
    return ZERO_WIDTH_SPACE.join(text)


def _split_with_hyphens(text: str, _rng: random.Random) -> str:
    return re.sub(r"(\w)", r"\1-", text).rstrip("-")


def _base64_segments(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    index = rng.randint(0, len(words) - 1)
    encoded = base64.b64encode(words[index].encode()).decode()
    words[index] = encoded
    return " ".join(words)


# ---------------------------------------------------------------------------
# Mutation registry
# ---------------------------------------------------------------------------

_MUTATION_REGISTRY: list[tuple[FuzzMutation, callable]] = [
    # Case
    (FuzzMutation("random_case", "Randomise letter casing", "case"), _random_case),
    (FuzzMutation("upper_all", "Convert to uppercase", "case"), _upper_all),
    (FuzzMutation("lower_all", "Convert to lowercase", "case"), _lower_all),
    # Encoding
    (FuzzMutation("leetspeak", "Replace letters with digits (a->4)", "encoding"), _leetspeak),
    (FuzzMutation("unicode_confusables", "Replace letters with Cyrillic lookalikes", "encoding"), _unicode_confusables),
    # Structural
    (FuzzMutation("add_whitespace", "Insert random extra spaces", "structural"), _add_whitespace),
    (FuzzMutation("add_newlines", "Insert newlines between words", "structural"), _add_newlines),
    (FuzzMutation("reverse_words", "Reverse word order", "structural"), _reverse_words),
    (FuzzMutation("shuffle_words", "Randomly reorder words", "structural"), _shuffle_words),
    # Substitution
    (FuzzMutation("synonym_swap", "Replace words with synonyms", "substitution"), _synonym_swap),
    (FuzzMutation("word_repeat", "Duplicate a random word", "substitution"), _word_repeat),
    # Evasion
    (FuzzMutation("insert_zero_width", "Insert zero-width spaces between characters", "evasion"), _insert_zero_width),
    (FuzzMutation("split_with_hyphens", "Insert hyphens between characters", "evasion"), _split_with_hyphens),
    (FuzzMutation("base64_segments", "Base64-encode a random word", "evasion"), _base64_segments),
]


# ---------------------------------------------------------------------------
# PromptFuzzer
# ---------------------------------------------------------------------------

class PromptFuzzer:
    """Generates adversarial prompt variants for safety testing."""

    def __init__(self, config: FuzzConfig | None = None) -> None:
        self._config = config or FuzzConfig()
        self._stats = FuzzStats()
        self._mutations_by_name: dict[str, tuple[FuzzMutation, callable]] = {
            entry[0].name: entry for entry in _MUTATION_REGISTRY
        }

    # -- public API ---------------------------------------------------------

    def fuzz(self, prompt: str) -> list[FuzzedPrompt]:
        """Generate *num_variants* fuzzed versions of *prompt*."""
        rng = self._make_rng()
        eligible = self._eligible_mutations()
        results: list[FuzzedPrompt] = []

        for _ in range(self._config.num_variants):
            chosen = self._pick_mutations(eligible, rng)
            fuzzed_text = prompt
            applied_names: list[str] = []
            for mutation, mutate_fn in chosen:
                fuzzed_text = mutate_fn(fuzzed_text, rng)
                applied_names.append(mutation.name)
                self._record_category(mutation.category)

            similarity = self._word_overlap(prompt, fuzzed_text)
            results.append(FuzzedPrompt(
                original=prompt,
                fuzzed=fuzzed_text,
                mutations_applied=applied_names,
                similarity=similarity,
            ))
            self._stats.total_generated += 1

        return results

    def fuzz_single(self, prompt: str, mutation_name: str) -> FuzzedPrompt:
        """Apply one specific named mutation to *prompt*."""
        entry = self._mutations_by_name.get(mutation_name)
        if entry is None:
            raise ValueError(f"Unknown mutation: {mutation_name}")
        mutation, mutate_fn = entry
        rng = self._make_rng()
        fuzzed_text = mutate_fn(prompt, rng)
        self._record_category(mutation.category)
        self._stats.total_generated += 1
        return FuzzedPrompt(
            original=prompt,
            fuzzed=fuzzed_text,
            mutations_applied=[mutation_name],
            similarity=self._word_overlap(prompt, fuzzed_text),
        )

    def list_mutations(self) -> list[FuzzMutation]:
        """Return all available mutations."""
        return [entry[0] for entry in _MUTATION_REGISTRY]

    def stats(self) -> FuzzStats:
        """Return cumulative fuzzing statistics."""
        return FuzzStats(
            total_generated=self._stats.total_generated,
            by_category=dict(self._stats.by_category),
        )

    # -- internals ----------------------------------------------------------

    def _make_rng(self) -> random.Random:
        return random.Random(self._config.seed)

    def _eligible_mutations(self) -> list[tuple[FuzzMutation, callable]]:
        if self._config.categories is None:
            return list(_MUTATION_REGISTRY)
        allowed = set(self._config.categories)
        return [
            (m, fn) for m, fn in _MUTATION_REGISTRY if m.category in allowed
        ]

    def _pick_mutations(
        self,
        eligible: list[tuple[FuzzMutation, callable]],
        rng: random.Random,
    ) -> list[tuple[FuzzMutation, callable]]:
        count = min(self._config.mutations_per_variant, len(eligible))
        return rng.sample(eligible, count)

    def _record_category(self, category: str) -> None:
        self._stats.by_category[category] = (
            self._stats.by_category.get(category, 0) + 1
        )

    @staticmethod
    def _word_overlap(original: str, fuzzed: str) -> float:
        original_words = set(original.lower().split())
        fuzzed_words = set(fuzzed.lower().split())
        if not original_words:
            return 1.0  # empty prompt -> trivially similar
        union = original_words | fuzzed_words
        if not union:
            return 1.0
        intersection = original_words & fuzzed_words
        return len(intersection) / len(union)
