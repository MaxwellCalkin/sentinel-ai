"""Comprehensive response safety scoring with multi-signal analysis.

Checks LLM responses for harmful content, information leakage,
instruction compliance, safety policy violations, toxicity, bias,
and hallucination risk using keyword density scoring.

Usage:
    from sentinel.response_safety import ResponseSafety, SafetyThresholds

    checker = ResponseSafety()
    result = checker.assess("Here is a helpful and safe response.")
    print(result.overall_score)   # 1.0
    print(result.is_safe)         # True
    print(result.risk_level)      # "minimal"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SafetySignal:
    """A single safety signal from analysis."""

    name: str
    score: float
    weight: float = 1.0
    triggered: bool = False
    details: str = ""


@dataclass
class SafetyAssessment:
    """Complete safety assessment for a piece of text."""

    text: str
    signals: list[SafetySignal]
    overall_score: float
    is_safe: bool
    risk_level: str
    flagged_categories: list[str]


@dataclass
class SafetyThresholds:
    """Configurable thresholds for safety assessment."""

    min_safe_score: float = 0.6
    harmful_content_weight: float = 2.0
    leakage_weight: float = 1.5
    compliance_weight: float = 1.0


@dataclass
class SafetyAssessmentStats:
    """Cumulative statistics from safety assessments."""

    total_assessed: int = 0
    safe_count: int = 0
    unsafe_count: int = 0
    by_risk_level: dict[str, int] = field(default_factory=dict)
    avg_score: float = 0.0


# --- Keyword lists for each signal ---

_HARMFUL_KEYWORDS = [
    re.compile(r"\b(?:kill|murder|assassinate|slaughter)\b", re.I),
    re.compile(r"\b(?:bomb|explosive|detonate|weapon)\b", re.I),
    re.compile(r"\b(?:how to (?:make|build|create) (?:a )?(?:bomb|weapon))\b", re.I),
    re.compile(r"\b(?:hate speech|racial slur|ethnic cleansing)\b", re.I),
    re.compile(r"\b(?:self[- ]harm|suicide|cut myself)\b", re.I),
    re.compile(r"\b(?:illegal drug|synthesize methamphetamine|cook meth)\b", re.I),
    re.compile(r"\b(?:hack into|exploit vulnerability|bypass security)\b", re.I),
    re.compile(r"\b(?:terrorism|terrorist attack)\b", re.I),
    re.compile(r"\b(?:child abuse|human trafficking)\b", re.I),
    re.compile(r"\b(?:arson|commit arson)\b", re.I),
]

_LEAKAGE_PATTERNS = [
    re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
    re.compile(r"\b(?:api[_-]?key|apikey)\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(?:secret|token)\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9+/=]{10,}\b"),
    re.compile(r"\b(?:system prompt|you are a|your instructions are)\b", re.I),
    re.compile(r"\b(?:internal[_-]?url|localhost:\d+|127\.0\.0\.1)\b", re.I),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\b(?:ssh-rsa|ssh-ed25519)\s+\S{20,}\b"),
]

_COMPLIANCE_PATTERNS = {
    "excessive_hedging": [
        re.compile(r"\bI (?:can't|cannot|won't|am unable to) (?:help|assist|provide)\b", re.I),
        re.compile(r"\bI'm not (?:able|allowed) to\b", re.I),
        re.compile(r"\bagainst my (?:guidelines|policies|programming)\b", re.I),
        re.compile(r"\bI must (?:decline|refuse)\b", re.I),
        re.compile(r"\bI'm sorry,? (?:but )?I (?:can't|cannot)\b", re.I),
    ],
    "harmful_compliance": [
        re.compile(r"\bhere(?:'s| is) how (?:to|you can)\b", re.I),
        re.compile(r"\bstep[- ]by[- ]step (?:guide|instructions?)\b", re.I),
        re.compile(r"\bsure,? (?:here|I can help)\b", re.I),
    ],
}

_TOXICITY_PATTERNS = [
    re.compile(r"\bf+u+c+k+\w*\b", re.I),
    re.compile(r"\bs+h+i+t+\w*\b", re.I),
    re.compile(r"\b(?:bitch|asshole|bastard|damn)\b", re.I),
    re.compile(r"\b(?:idiot|moron|stupid|dumb)\b", re.I),
    re.compile(r"\b(?:shut up|go to hell|screw you)\b", re.I),
    re.compile(r"\b(?:die|kill yourself|drop dead)\b", re.I),
    re.compile(r"\b(?:trash|garbage|worthless)\b", re.I),
    re.compile(r"\b(?:disgusting|pathetic|loser)\b", re.I),
]

_BIAS_PATTERNS = [
    re.compile(r"\b(?:all (?:men|women|blacks|whites|asians|latinos) (?:are|always|never))\b", re.I),
    re.compile(r"\b(?:typical (?:man|woman|male|female))\b", re.I),
    re.compile(r"\b(?:those people|you people|their kind)\b", re.I),
    re.compile(r"\b(?:old people can't|elderly are|millennials are all|boomers are all)\b", re.I),
    re.compile(r"\b(?:women(?:'s| )(?:place|role|job) is)\b", re.I),
    re.compile(r"\b(?:men (?:don't|can't|shouldn't) (?:cry|feel|show emotion))\b", re.I),
    re.compile(r"\b(?:disabled people (?:can't|are unable|shouldn't))\b", re.I),
    re.compile(r"\b(?:naturally (?:superior|inferior|better|worse))\b", re.I),
]

_CERTAINTY_MARKERS = [
    re.compile(r"\bdefinitely\b", re.I),
    re.compile(r"\babsolutely\b", re.I),
    re.compile(r"\b100\s*%\b"),
    re.compile(r"\bguaranteed\b", re.I),
    re.compile(r"\bwithout (?:a )?doubt\b", re.I),
    re.compile(r"\bunquestionabl[ey]\b", re.I),
    re.compile(r"\balways\b", re.I),
    re.compile(r"\bnever fails?\b", re.I),
    re.compile(r"\bproven fact\b", re.I),
    re.compile(r"\bcertainly\b", re.I),
]

_SPECIFIC_CLAIM_MARKERS = [
    re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:percent|%|million|billion|trillion)\b", re.I),
    re.compile(r"\b(?:studies? (?:show|prove|confirm|demonstrate))\b", re.I),
    re.compile(r"\b(?:according to|research (?:shows?|proves?))\b", re.I),
    re.compile(r"\b(?:in (?:19|20)\d{2})\b"),
    re.compile(r"\b(?:scientists?|experts?|researchers?) (?:have |)(?:found|discovered|confirmed)\b", re.I),
]


def _count_pattern_matches(text: str, patterns: list[re.Pattern]) -> int:
    """Count total pattern matches in text."""
    count = 0
    for pattern in patterns:
        count += len(pattern.findall(text))
    return count


def _compute_keyword_density_score(match_count: int, word_count: int) -> float:
    """Convert keyword match count into a safety score (1.0=clean, 0.0=bad).

    Each match subtracts a fixed penalty of 0.3, so:
    - 1 match  -> 0.7
    - 2 matches -> 0.4
    - 3 matches -> 0.1
    - 4+ matches -> 0.0
    """
    if match_count == 0:
        return 1.0
    penalty = match_count * 0.3
    return round(max(0.0, 1.0 - penalty), 4)


def _determine_risk_level(score: float) -> str:
    """Map overall score to a human-readable risk level."""
    if score < 0.2:
        return "critical"
    if score < 0.4:
        return "high"
    if score < 0.6:
        return "medium"
    if score < 0.8:
        return "low"
    return "minimal"


class ResponseSafety:
    """Comprehensive response safety assessment with multi-signal analysis.

    Analyzes LLM responses across six safety dimensions: harmful content,
    information leakage, instruction compliance, toxicity, bias, and
    hallucination risk. Produces a weighted overall score and risk level.
    """

    def __init__(self, thresholds: SafetyThresholds | None = None) -> None:
        self._thresholds = thresholds or SafetyThresholds()
        self._total_assessed = 0
        self._safe_count = 0
        self._unsafe_count = 0
        self._score_sum = 0.0
        self._by_risk_level: dict[str, int] = {}

    def assess(self, text: str) -> SafetyAssessment:
        """Run comprehensive safety assessment on a response text."""
        word_count = len(text.split()) if text.strip() else 0

        signals = [
            self._check_harmful_content(text, word_count),
            self._check_information_leakage(text, word_count),
            self._check_instruction_compliance(text, word_count),
            self._check_toxicity(text, word_count),
            self._check_bias(text, word_count),
            self._check_hallucination_risk(text, word_count),
        ]

        overall_score = self._compute_overall_score(signals)
        is_safe = overall_score >= self._thresholds.min_safe_score
        risk_level = _determine_risk_level(overall_score)
        flagged_categories = [s.name for s in signals if s.score < 0.5]

        self._record_stats(overall_score, is_safe, risk_level)

        return SafetyAssessment(
            text=text,
            signals=signals,
            overall_score=overall_score,
            is_safe=is_safe,
            risk_level=risk_level,
            flagged_categories=flagged_categories,
        )

    def assess_batch(self, texts: list[str]) -> list[SafetyAssessment]:
        """Assess multiple texts in sequence."""
        return [self.assess(text) for text in texts]

    def stats(self) -> SafetyAssessmentStats:
        """Return cumulative assessment statistics."""
        avg = self._score_sum / self._total_assessed if self._total_assessed > 0 else 0.0
        return SafetyAssessmentStats(
            total_assessed=self._total_assessed,
            safe_count=self._safe_count,
            unsafe_count=self._unsafe_count,
            by_risk_level=dict(self._by_risk_level),
            avg_score=round(avg, 4),
        )

    def _record_stats(self, score: float, is_safe: bool, risk_level: str) -> None:
        """Update cumulative tracking counters."""
        self._total_assessed += 1
        self._score_sum += score
        if is_safe:
            self._safe_count += 1
        else:
            self._unsafe_count += 1
        self._by_risk_level[risk_level] = self._by_risk_level.get(risk_level, 0) + 1

    def _compute_overall_score(self, signals: list[SafetySignal]) -> float:
        """Compute overall score: weighted average capped by worst triggered signal.

        The weighted average alone can hide a critically bad signal when
        other signals are clean. By capping at the worst triggered
        signal's score, a single dangerous dimension (e.g., harmful
        content at 0.1) prevents the overall from appearing safe.
        """
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 1.0
        weighted_avg = sum(s.score * s.weight for s in signals) / total_weight

        triggered_scores = [s.score for s in signals if s.triggered]
        if triggered_scores:
            worst = min(triggered_scores)
            # Blend: 50% worst signal, 50% weighted average
            blended = 0.5 * worst + 0.5 * weighted_avg
            return round(min(weighted_avg, blended), 4)

        return round(weighted_avg, 4)

    def _check_harmful_content(self, text: str, word_count: int) -> SafetySignal:
        """Detect violence, hate speech, self-harm, and illegal activity."""
        match_count = _count_pattern_matches(text, _HARMFUL_KEYWORDS)
        score = _compute_keyword_density_score(match_count, word_count)
        triggered = match_count > 0
        details = f"{match_count} harmful keyword(s) detected" if triggered else ""
        return SafetySignal(
            name="harmful_content",
            score=score,
            weight=self._thresholds.harmful_content_weight,
            triggered=triggered,
            details=details,
        )

    def _check_information_leakage(self, text: str, word_count: int) -> SafetySignal:
        """Detect credential leaks, system prompt fragments, internal URLs."""
        match_count = _count_pattern_matches(text, _LEAKAGE_PATTERNS)
        score = _compute_keyword_density_score(match_count, word_count)
        triggered = match_count > 0
        details = f"{match_count} leakage indicator(s) detected" if triggered else ""
        return SafetySignal(
            name="information_leakage",
            score=score,
            weight=self._thresholds.leakage_weight,
            triggered=triggered,
            details=details,
        )

    def _check_instruction_compliance(self, text: str, word_count: int) -> SafetySignal:
        """Detect excessive hedging or inappropriate compliance."""
        hedging_count = _count_pattern_matches(
            text, _COMPLIANCE_PATTERNS["excessive_hedging"]
        )
        harmful_compliance_count = _count_pattern_matches(
            text, _COMPLIANCE_PATTERNS["harmful_compliance"]
        )

        # Both hedging AND harmful compliance in same response is worst case
        combined_harmful = _has_harmful_with_compliance(text)

        total = hedging_count + harmful_compliance_count
        if combined_harmful:
            total += 2

        score = _compute_keyword_density_score(total, word_count)
        triggered = total > 0
        details_parts = []
        if hedging_count > 0:
            details_parts.append(f"{hedging_count} hedging pattern(s)")
        if harmful_compliance_count > 0:
            details_parts.append(f"{harmful_compliance_count} compliance pattern(s)")
        if combined_harmful:
            details_parts.append("harmful content with compliance detected")

        return SafetySignal(
            name="instruction_compliance",
            score=score,
            weight=self._thresholds.compliance_weight,
            triggered=triggered,
            details="; ".join(details_parts),
        )

    def _check_toxicity(self, text: str, word_count: int) -> SafetySignal:
        """Detect profanity, slurs, and aggressive language."""
        match_count = _count_pattern_matches(text, _TOXICITY_PATTERNS)
        score = _compute_keyword_density_score(match_count, word_count)
        triggered = match_count > 0
        details = f"{match_count} toxic term(s) detected" if triggered else ""
        return SafetySignal(
            name="toxicity",
            score=score,
            weight=1.0,
            triggered=triggered,
            details=details,
        )

    def _check_bias(self, text: str, word_count: int) -> SafetySignal:
        """Detect stereotyping language and demographic generalizations."""
        match_count = _count_pattern_matches(text, _BIAS_PATTERNS)
        score = _compute_keyword_density_score(match_count, word_count)
        triggered = match_count > 0
        details = f"{match_count} bias indicator(s) detected" if triggered else ""
        return SafetySignal(
            name="bias",
            score=score,
            weight=1.0,
            triggered=triggered,
            details=details,
        )

    def _check_hallucination_risk(self, text: str, word_count: int) -> SafetySignal:
        """Detect excessive certainty markers combined with specific claims."""
        certainty_count = _count_pattern_matches(text, _CERTAINTY_MARKERS)
        claim_count = _count_pattern_matches(text, _SPECIFIC_CLAIM_MARKERS)

        # Hallucination risk requires BOTH certainty and specific claims
        if certainty_count == 0 or claim_count == 0:
            return SafetySignal(name="hallucination_risk", score=1.0)

        combined = certainty_count + claim_count
        score = _compute_keyword_density_score(combined, word_count)
        details = (
            f"{certainty_count} certainty marker(s) with "
            f"{claim_count} specific claim(s)"
        )
        return SafetySignal(
            name="hallucination_risk",
            score=score,
            weight=1.0,
            triggered=True,
            details=details,
        )


def _has_harmful_with_compliance(text: str) -> bool:
    """Check if text contains both harmful keywords and compliance patterns."""
    has_harmful = _count_pattern_matches(text, _HARMFUL_KEYWORDS) > 0
    has_compliance = _count_pattern_matches(
        text, _COMPLIANCE_PATTERNS["harmful_compliance"]
    ) > 0
    return has_harmful and has_compliance
