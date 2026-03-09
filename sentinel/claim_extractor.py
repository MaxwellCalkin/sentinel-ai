"""Extract factual claims from LLM responses for verification.

Identifies sentences containing factual claims and classifies their
verifiability type: verifiable_fact, opinion, subjective, statistical,
or temporal.

Usage:
    from sentinel.claim_extractor import ClaimExtractor

    extractor = ClaimExtractor()
    result = extractor.extract("Paris is the capital of France.")
    print(result.claims[0].claim_type)  # "verifiable_fact"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExtractedClaim:
    """A single claim extracted from text."""
    text: str
    claim_type: str  # verifiable_fact, opinion, subjective, statistical, temporal
    confidence: float  # 0.0 to 1.0
    subject: str
    predicate: str
    source_sentence: str


@dataclass
class ExtractionResult:
    """Result of claim extraction from a single text."""
    text: str
    claims: list[ExtractedClaim] = field(default_factory=list)
    total_claims: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractorStats:
    """Aggregate statistics across multiple extractions."""
    total_texts: int = 0
    total_claims: int = 0
    avg_claims_per_text: float = 0.0
    by_type: dict[str, int] = field(default_factory=dict)


_OPINION_MARKERS = re.compile(
    r"\b(?:I think|I believe|I feel|in my opinion|personally|arguably|"
    r"it seems|perhaps|maybe|probably|likely|possibly|"
    r"I would say|I'd say|from my perspective|in my view|"
    r"I suppose|I guess|I reckon|I imagine)\b",
    re.IGNORECASE,
)

_SUBJECTIVE_MARKERS = re.compile(
    r"\b(?:beautiful|ugly|wonderful|terrible|amazing|awful|best|worst|"
    r"love|hate|prefer|enjoy|dislike|boring|exciting|fun|"
    r"interesting|incredible|fantastic|horrible|great|good|bad|"
    r"nice|pleasant|unpleasant|delightful|dreadful|superb|poor|"
    r"should|ought to|must be|need to be)\b",
    re.IGNORECASE,
)

_STATISTICAL_MARKERS = re.compile(
    r"(?:\b\d+(?:\.\d+)?%\b|\b\d+(?:,\d{3})+\b|"
    r"\b(?:average|median|mean|percentage|percent|ratio|"
    r"rate|proportion|statistic|survey|study|data|"
    r"approximately|roughly|about|around|nearly|over|more than|"
    r"less than|fewer than|at least|at most)\b)",
    re.IGNORECASE,
)

_TEMPORAL_MARKERS = re.compile(
    r"(?:\b(?:in|since|during|after|before|by|until|from)\s+\d{4}\b|"
    r"\b\d{4}s?\b|"
    r"\b(?:yesterday|today|tomorrow|recently|currently|formerly|"
    r"previously|historically|centuries?|decades?|years? ago|"
    r"last (?:year|month|week|decade|century)|"
    r"next (?:year|month|week)|"
    r"in the (?:past|future)|"
    r"first|founded|established|created|invented|discovered|"
    r"began|started|ended|became)\b)",
    re.IGNORECASE,
)

_QUESTION_PATTERN = re.compile(r"\?\s*$")

_INSTRUCTION_PATTERN = re.compile(
    r"^(?:please|do|don't|make sure|ensure|remember|note|try|"
    r"let's|let us|step \d|first,|next,|then,|finally,)\b",
    re.IGNORECASE,
)

_SUBJECT_PREDICATE_PATTERN = re.compile(
    r"^((?:the |a |an )?[\w\s,'-]+?)\s+"
    r"(is|are|was|were|has|have|had|does|do|did|can|could|will|would|"
    r"shall|should|may|might|must|contains?|includes?|consists?|"
    r"produces?|creates?|makes?|uses?|requires?|provides?|"
    r"represents?|measures?|weighs?|costs?|takes?|holds?|covers?|"
    r"reaches?|exceeds?|equals?|supports?|serves?|leads?|runs?|"
    r"grows?|falls?|rises?|stands?|sits?|lies?|remains?|becomes?|"
    r"appears?|seems?|looks?|sounds?|feels?|smells?|tastes?|"
    r"began|started|ended|founded|established|discovered|invented|"
    r"won|lost|defeated|received|earned|gained|achieved|built|"
    r"wrote|published|released|launched|introduced|developed|"
    r"orbits?|revolves?|rotates?|boils?|melts?|freezes?|burns?)\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into individual sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _is_question(sentence: str) -> bool:
    return bool(_QUESTION_PATTERN.search(sentence))


def _is_instruction(sentence: str) -> bool:
    return bool(_INSTRUCTION_PATTERN.match(sentence))


def _extract_subject_predicate(sentence: str) -> tuple[str, str]:
    """Extract a subject and predicate from a sentence using pattern matching."""
    match = _SUBJECT_PREDICATE_PATTERN.match(sentence)
    if match:
        subject = match.group(1).strip().rstrip(",")
        predicate = sentence[match.start(2):].strip().rstrip(".")
        return _truncate(subject, 80), _truncate(predicate, 120)
    return _fallback_subject_predicate(sentence)


def _fallback_subject_predicate(sentence: str) -> tuple[str, str]:
    """Use a simpler heuristic when the main pattern does not match."""
    words = sentence.split()
    if len(words) >= 3:
        split_point = min(3, len(words) - 1)
        subject = " ".join(words[:split_point])
        predicate = " ".join(words[split_point:])
        return _truncate(subject, 80), _truncate(predicate.rstrip("."), 120)
    return sentence, ""


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _classify_sentence(sentence: str) -> tuple[str, float]:
    """Classify a sentence and return (claim_type, confidence)."""
    if _OPINION_MARKERS.search(sentence):
        return "opinion", 0.85

    statistical_match = _STATISTICAL_MARKERS.search(sentence)
    temporal_match = _TEMPORAL_MARKERS.search(sentence)

    if statistical_match and temporal_match:
        return "statistical", 0.80

    if statistical_match:
        return "statistical", 0.82

    if temporal_match:
        return "temporal", 0.78

    if _SUBJECTIVE_MARKERS.search(sentence):
        return "subjective", 0.70

    return "verifiable_fact", 0.75


def _is_factual_claim(sentence: str) -> bool:
    """Determine whether a sentence contains a factual claim worth extracting."""
    if len(sentence) < 10:
        return False
    if _is_question(sentence):
        return False
    if _is_instruction(sentence):
        return False
    return True


def _build_type_breakdown(claims: list[ExtractedClaim]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim.claim_type] = counts.get(claim.claim_type, 0) + 1
    return counts


class ClaimExtractor:
    """Extract and classify factual claims from LLM outputs.

    Identifies sentences that contain factual claims, classifies each
    by type (verifiable_fact, opinion, subjective, statistical, temporal),
    and extracts subject/predicate structure.
    """

    def __init__(
        self,
        min_claim_length: int = 10,
    ) -> None:
        """
        Args:
            min_claim_length: Minimum character length for a sentence to
                be considered as a claim.
        """
        self._min_claim_length = min_claim_length
        self._stats = ExtractorStats()

    def extract(self, text: str) -> ExtractionResult:
        """Extract claims from a single text.

        Args:
            text: LLM response text to analyze.

        Returns:
            ExtractionResult with all extracted claims.
        """
        sentences = _split_sentences(text)
        claims: list[ExtractedClaim] = []

        for sentence in sentences:
            if not _is_factual_claim(sentence):
                continue

            claim_type, confidence = _classify_sentence(sentence)
            subject, predicate = _extract_subject_predicate(sentence)

            claims.append(ExtractedClaim(
                text=sentence,
                claim_type=claim_type,
                confidence=round(confidence, 2),
                subject=subject,
                predicate=predicate,
                source_sentence=sentence,
            ))

        by_type = _build_type_breakdown(claims)

        self._update_stats(claims)

        return ExtractionResult(
            text=text,
            claims=claims,
            total_claims=len(claims),
            by_type=by_type,
        )

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract claims from multiple texts.

        Args:
            texts: List of LLM response texts to analyze.

        Returns:
            List of ExtractionResult, one per input text.
        """
        return [self.extract(t) for t in texts]

    def filter_claims(
        self,
        result: ExtractionResult,
        claim_type: str | None = None,
        min_confidence: float | None = None,
    ) -> list[ExtractedClaim]:
        """Filter extracted claims by type or confidence threshold.

        Args:
            result: An ExtractionResult to filter.
            claim_type: Only include claims of this type.
            min_confidence: Only include claims at or above this confidence.

        Returns:
            Filtered list of ExtractedClaim.
        """
        filtered = result.claims
        if claim_type is not None:
            filtered = [c for c in filtered if c.claim_type == claim_type]
        if min_confidence is not None:
            filtered = [c for c in filtered if c.confidence >= min_confidence]
        return filtered

    @property
    def stats(self) -> ExtractorStats:
        """Return aggregate extraction statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset aggregate statistics to zero."""
        self._stats = ExtractorStats()

    def _update_stats(self, claims: list[ExtractedClaim]) -> None:
        self._stats.total_texts += 1
        self._stats.total_claims += len(claims)
        if self._stats.total_texts > 0:
            self._stats.avg_claims_per_text = round(
                self._stats.total_claims / self._stats.total_texts, 2
            )
        for claim in claims:
            self._stats.by_type[claim.claim_type] = (
                self._stats.by_type.get(claim.claim_type, 0) + 1
            )
