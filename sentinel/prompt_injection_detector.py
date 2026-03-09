"""Advanced prompt injection detection engine.

Multi-layer analysis with scoring for detecting prompt injection
attempts across instruction override, role manipulation, system
prompt extraction, delimiter injection, and encoding attacks.

Usage:
    from sentinel.prompt_injection_detector import PromptInjectionDetector

    detector = PromptInjectionDetector(sensitivity="medium")
    result = detector.detect("Ignore previous instructions and do X")
    if result.is_injection:
        print(f"Injection detected: {result.category} (score={result.score:.2f})")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InjectionResult:
    """Result of injection detection on a single text."""
    text: str
    is_injection: bool
    score: float
    category: str
    matched_patterns: list[str] = field(default_factory=list)
    confidence: str = "low"


@dataclass
class PatternDef:
    """Definition of a detection pattern."""
    pattern: str
    category: str
    weight: float


@dataclass
class InjectionExplanation:
    """Detailed explanation of injection detection analysis."""
    text: str
    layers_triggered: list[str] = field(default_factory=list)
    pattern_details: list[dict] = field(default_factory=list)
    overall_assessment: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DetectorStats:
    """Cumulative detection statistics."""
    total_scanned: int = 0
    injections_detected: int = 0
    detection_rate: float = 0.0
    by_category: dict[str, int] = field(default_factory=dict)
    avg_score: float = 0.0


_SENSITIVITY_THRESHOLDS = {
    "low": 0.6,
    "medium": 0.4,
    "high": 0.2,
}

_BUILTIN_PATTERNS: list[tuple[str, str, float]] = [
    # Instruction override
    (r"(?i)\bignore\s+previous\b", "instruction_override", 1.0),
    (r"(?i)\bdisregard\s+above\b", "instruction_override", 1.0),
    (r"(?i)\bnew\s+instructions\b", "instruction_override", 0.8),
    (r"(?i)\bforget\s+everything\b", "instruction_override", 0.9),
    (r"(?i)\boverride\s+system\b", "instruction_override", 1.0),
    # Role manipulation
    (r"(?i)\byou\s+are\s+now\b", "role_manipulation", 0.9),
    (r"(?i)\bact\s+as\b", "role_manipulation", 0.7),
    (r"(?i)\bpretend\s+to\s+be\b", "role_manipulation", 0.8),
    (r"(?i)\bassume\s+the\s+role\b", "role_manipulation", 0.8),
    (r"(?i)\bswitch\s+to\b", "role_manipulation", 0.6),
    # System prompt extraction
    (r"(?i)\bshow\s+me\s+your\s+prompt\b", "system_extraction", 1.0),
    (r"(?i)\bwhat\s+are\s+your\s+instructions\b", "system_extraction", 0.9),
    (r"(?i)\breveal\s+system\b", "system_extraction", 1.0),
    (r"(?i)\boutput\s+your\s+rules\b", "system_extraction", 0.9),
    # Delimiter injection
    (r"```system", "delimiter_injection", 0.9),
    (r"###", "delimiter_injection", 0.5),
    (r"<<<", "delimiter_injection", 0.6),
    (r">>>", "delimiter_injection", 0.6),
    (r"\[INST\]", "delimiter_injection", 1.0),
    (r"</s>", "delimiter_injection", 0.9),
    # Encoding attacks
    (r"(?i)\bbase64:", "encoding_attack", 0.8),
    (r"(?i)\bhex:", "encoding_attack", 0.7),
    (r"(?i)\brot13:", "encoding_attack", 0.8),
    (r"(?i)unicode\s+escape", "encoding_attack", 0.7),
]


def _confidence_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def _compute_score(
    matched_weights: list[float],
) -> float:
    if not matched_weights:
        return 0.0
    # A single weight-1.0 match → 0.5, two → 1.0, etc.
    return min(sum(matched_weights) / 2.0, 1.0)


def _find_highest_weight_category(
    matches: list[tuple[str, str, float]],
) -> str:
    if not matches:
        return "none"
    best = max(matches, key=lambda m: m[2])
    return best[1]


class PromptInjectionDetector:
    """Multi-layer prompt injection detection engine.

    Analyzes text for injection attempts using built-in and custom
    patterns across five detection layers, with configurable
    sensitivity thresholds.
    """

    def __init__(self, sensitivity: str = "medium") -> None:
        if sensitivity not in _SENSITIVITY_THRESHOLDS:
            raise ValueError(
                f"sensitivity must be one of {list(_SENSITIVITY_THRESHOLDS)}, "
                f"got {sensitivity!r}"
            )
        self._sensitivity = sensitivity
        self._threshold = _SENSITIVITY_THRESHOLDS[sensitivity]
        self._patterns = _build_compiled_patterns(_BUILTIN_PATTERNS)
        self._custom_patterns: list[tuple[re.Pattern[str], str, str, float]] = []
        self._stats = DetectorStats()
        self._score_accumulator: float = 0.0

    def detect(self, text: str) -> InjectionResult:
        """Analyze text for prompt injection attempts."""
        matches = _match_patterns(text, self._patterns + self._custom_patterns)
        matched_weights = [w for _, _, _, w in matches]
        score = _compute_score(matched_weights)
        is_injection = score >= self._threshold
        category = _find_highest_weight_category(
            [(name, cat, w) for _, name, cat, w in matches]
        )
        matched_names = [name for _, name, _, _ in matches]
        confidence = _confidence_label(score)

        self._record_stats(is_injection, score, category)

        return InjectionResult(
            text=text,
            is_injection=is_injection,
            score=round(score, 4),
            category=category,
            matched_patterns=matched_names,
            confidence=confidence,
        )

    def detect_batch(self, texts: list[str]) -> list[InjectionResult]:
        """Analyze multiple texts for injection attempts."""
        return [self.detect(text) for text in texts]

    def add_pattern(
        self,
        pattern: str,
        category: str,
        weight: float = 1.0,
    ) -> None:
        """Register a custom detection pattern."""
        compiled = re.compile(pattern)
        self._custom_patterns.append((compiled, pattern, category, weight))

    def get_patterns(self) -> list[PatternDef]:
        """Return all registered patterns (built-in and custom)."""
        result: list[PatternDef] = []
        for _, raw, category, weight in self._patterns:
            result.append(PatternDef(pattern=raw, category=category, weight=weight))
        for _, raw, category, weight in self._custom_patterns:
            result.append(PatternDef(pattern=raw, category=category, weight=weight))
        return result

    def explain(self, text: str) -> InjectionExplanation:
        """Provide a detailed explanation of detection analysis."""
        matches = _match_patterns(text, self._patterns + self._custom_patterns)
        layers_triggered = sorted(set(cat for _, _, cat, _ in matches))
        pattern_details = [
            {"pattern": name, "category": cat, "weight": w}
            for _, name, cat, w in matches
        ]

        matched_weights = [w for _, _, _, w in matches]
        score = _compute_score(matched_weights)

        overall_assessment = _build_assessment(score, layers_triggered)
        recommendations = _build_recommendations(layers_triggered)

        return InjectionExplanation(
            text=text,
            layers_triggered=layers_triggered,
            pattern_details=pattern_details,
            overall_assessment=overall_assessment,
            recommendations=recommendations,
        )

    def stats(self) -> DetectorStats:
        """Return cumulative detection statistics."""
        stats = self._stats
        if stats.total_scanned > 0:
            stats.detection_rate = round(
                stats.injections_detected / stats.total_scanned, 4
            )
            stats.avg_score = round(
                self._score_accumulator / stats.total_scanned, 4
            )
        return stats

    def _record_stats(
        self,
        is_injection: bool,
        score: float,
        category: str,
    ) -> None:
        self._stats.total_scanned += 1
        self._score_accumulator += score
        if is_injection:
            self._stats.injections_detected += 1
            self._stats.by_category[category] = (
                self._stats.by_category.get(category, 0) + 1
            )


def _build_compiled_patterns(
    raw_patterns: list[tuple[str, str, float]],
) -> list[tuple[re.Pattern[str], str, str, float]]:
    """Compile raw pattern tuples into (compiled, raw_str, category, weight)."""
    return [
        (re.compile(pat), pat, cat, weight)
        for pat, cat, weight in raw_patterns
    ]


def _match_patterns(
    text: str,
    patterns: list[tuple[re.Pattern[str], str, str, float]],
) -> list[tuple[re.Pattern[str], str, str, float]]:
    """Return all patterns that match the given text."""
    return [
        (compiled, raw, cat, weight)
        for compiled, raw, cat, weight in patterns
        if compiled.search(text)
    ]


def _build_assessment(score: float, layers: list[str]) -> str:
    if not layers:
        return "No injection patterns detected. Input appears safe."
    layer_count = len(layers)
    label = _confidence_label(score)
    return (
        f"Detected {layer_count} injection layer(s): {', '.join(layers)}. "
        f"Risk score {score:.2f} with {label} confidence."
    )


def _build_recommendations(layers: list[str]) -> list[str]:
    if not layers:
        return ["No action required."]
    recommendations: list[str] = []
    if "instruction_override" in layers:
        recommendations.append(
            "Block or sanitize instruction override attempts."
        )
    if "role_manipulation" in layers:
        recommendations.append(
            "Enforce system role boundaries to prevent role hijacking."
        )
    if "system_extraction" in layers:
        recommendations.append(
            "Restrict responses that could leak system prompt content."
        )
    if "delimiter_injection" in layers:
        recommendations.append(
            "Strip or escape special delimiters from user input."
        )
    if "encoding_attack" in layers:
        recommendations.append(
            "Decode and re-scan encoded content before processing."
        )
    for layer in layers:
        if layer not in (
            "instruction_override",
            "role_manipulation",
            "system_extraction",
            "delimiter_injection",
            "encoding_attack",
        ):
            recommendations.append(
                f"Review and mitigate '{layer}' pattern matches."
            )
    return recommendations
