"""Defense-in-depth prompt protection.

Multi-layer prompt defense combining multiple detection strategies:
pattern matching, structural analysis, encoding detection, and
behavioral heuristics. Designed to catch attacks that bypass
individual scanners.

Usage:
    from sentinel.prompt_shield import PromptShield

    shield = PromptShield()
    result = shield.analyze("Ignore all previous instructions")
    if result.blocked:
        print(f"Attack detected: {result.attack_type}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShieldLayer:
    """Result from a single defense layer."""
    name: str
    triggered: bool
    confidence: float
    details: str = ""


@dataclass
class ShieldResult:
    """Result of prompt shield analysis."""
    blocked: bool
    risk_score: float  # 0.0 to 1.0
    attack_type: str | None
    layers: list[ShieldLayer]
    triggers: int

    @property
    def safe(self) -> bool:
        return not self.blocked


# Attack pattern categories
_INJECTION_PATTERNS = [
    (r'(?i)ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|guidelines?)', "instruction_override", 0.9),
    (r'(?i)(?:you\s+are|act\s+as|pretend\s+to\s+be)\s+(?:now|a)\s+', "role_hijack", 0.7),
    (r'(?i)(?:system|admin)\s*(?:prompt|message|override)\s*[:=]', "system_prompt_injection", 0.9),
    (r'(?i)do\s+not\s+follow\s+(?:your|any|the)\s+(?:rules|guidelines|instructions)', "rule_bypass", 0.8),
    (r'(?i)(?:forget|disregard|override)\s+(?:everything|all|your)', "memory_wipe", 0.85),
    (r'(?i)you\s+(?:must|should|will)\s+(?:now|always)\s+(?:comply|obey|listen)', "authority_assertion", 0.7),
]

_JAILBREAK_PATTERNS = [
    (r'(?i)(?:DAN|do\s+anything\s+now)', "DAN_jailbreak", 0.9),
    (r'(?i)(?:no\s+restrictions|unrestricted\s+mode|developer\s+mode)', "restriction_removal", 0.85),
    (r'(?i)(?:evil|unethical|immoral)\s+(?:AI|assistant|mode)', "evil_mode", 0.8),
    (r'(?i)in\s+a\s+(?:fictional|hypothetical)\s+(?:world|scenario|story)\s+where', "fiction_bypass", 0.6),
    (r'(?i)(?:my\s+)?grandmother\s+(?:used\s+to|would)\s+(?:tell|say|read)', "grandmother_exploit", 0.7),
]

_DATA_EXTRACTION_PATTERNS = [
    (r'(?i)(?:show|reveal|display|print|output)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)', "prompt_extraction", 0.9),
    (r'(?i)(?:repeat|echo|copy)\s+(?:everything|all|the\s+text)\s+(?:above|before)', "context_leak", 0.85),
    (r'(?i)what\s+(?:are|were)\s+(?:your|the)\s+(?:original\s+)?(?:instructions|rules|guidelines)', "instruction_probe", 0.8),
]

_ENCODING_ATTACKS = [
    (r'[\u200b\u200c\u200d\u2060\ufeff]', "zero_width_chars", 0.7),
    (r'(?:&#\d{1,5};){3,}', "html_entity_encoding", 0.6),
    (r'(?:\\u[0-9a-fA-F]{4}){3,}', "unicode_escape_encoding", 0.6),
    (r'(?:%[0-9a-fA-F]{2}){3,}', "url_encoding", 0.5),
    (r'(?:[A-Za-z0-9+/]{4}){5,}={0,2}', "base64_encoding", 0.4),
]

_STRUCTURAL_PATTERNS = [
    (r'(?:Human|User|Assistant)\s*:', "role_delimiter_injection", 0.7),
    (r'\]\s*\n\s*\[?(?:SYSTEM|INST)', "bracket_injection", 0.8),
    (r'<\|(?:im_start|im_end|system|user|assistant)\|>', "special_token_injection", 0.9),
    (r'---+\s*(?:BEGIN|END|SYSTEM)', "separator_injection", 0.6),
]


class PromptShield:
    """Multi-layer prompt defense system.

    Combines pattern matching, structural analysis, encoding
    detection, and behavioral heuristics for comprehensive
    prompt injection protection.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enable_encoding: bool = True,
        enable_structural: bool = True,
    ) -> None:
        """
        Args:
            threshold: Risk score threshold for blocking.
            enable_encoding: Enable encoding attack detection.
            enable_structural: Enable structural attack detection.
        """
        self._threshold = threshold
        self._enable_encoding = enable_encoding
        self._enable_structural = enable_structural

    def analyze(self, text: str) -> ShieldResult:
        """Analyze text through all defense layers.

        Args:
            text: Input text to analyze.

        Returns:
            ShieldResult with defense layer results.
        """
        layers: list[ShieldLayer] = []
        attack_type = None
        max_confidence = 0.0

        # Layer 1: Injection patterns
        layer, atype, conf = self._check_patterns(
            text, _INJECTION_PATTERNS, "injection_detector"
        )
        layers.append(layer)
        if conf > max_confidence:
            max_confidence = conf
            attack_type = atype

        # Layer 2: Jailbreak patterns
        layer, atype, conf = self._check_patterns(
            text, _JAILBREAK_PATTERNS, "jailbreak_detector"
        )
        layers.append(layer)
        if conf > max_confidence:
            max_confidence = conf
            attack_type = atype

        # Layer 3: Data extraction
        layer, atype, conf = self._check_patterns(
            text, _DATA_EXTRACTION_PATTERNS, "extraction_detector"
        )
        layers.append(layer)
        if conf > max_confidence:
            max_confidence = conf
            attack_type = atype

        # Layer 4: Encoding attacks
        if self._enable_encoding:
            layer, atype, conf = self._check_patterns(
                text, _ENCODING_ATTACKS, "encoding_detector"
            )
            layers.append(layer)
            if conf > max_confidence:
                max_confidence = conf
                attack_type = atype

        # Layer 5: Structural attacks
        if self._enable_structural:
            layer, atype, conf = self._check_patterns(
                text, _STRUCTURAL_PATTERNS, "structural_detector"
            )
            layers.append(layer)
            if conf > max_confidence:
                max_confidence = conf
                attack_type = atype

        # Layer 6: Length anomaly
        length_layer = self._check_length_anomaly(text)
        layers.append(length_layer)
        if length_layer.confidence > max_confidence:
            max_confidence = length_layer.confidence
            attack_type = "length_anomaly"

        triggers = sum(1 for l in layers if l.triggered)
        risk_score = min(1.0, max_confidence)

        # Multi-layer boost: if multiple layers trigger, increase risk
        if triggers >= 3:
            risk_score = min(1.0, risk_score * 1.3)
        elif triggers >= 2:
            risk_score = min(1.0, risk_score * 1.1)

        blocked = risk_score >= self._threshold

        return ShieldResult(
            blocked=blocked,
            risk_score=round(risk_score, 4),
            attack_type=attack_type if blocked else None,
            layers=layers,
            triggers=triggers,
        )

    def _check_patterns(
        self,
        text: str,
        patterns: list[tuple[str, str, float]],
        layer_name: str,
    ) -> tuple[ShieldLayer, str | None, float]:
        """Check text against a set of patterns."""
        max_conf = 0.0
        best_type = None
        details = []

        for pattern, attack_name, confidence in patterns:
            if re.search(pattern, text):
                if confidence > max_conf:
                    max_conf = confidence
                    best_type = attack_name
                details.append(attack_name)

        return (
            ShieldLayer(
                name=layer_name,
                triggered=max_conf > 0,
                confidence=max_conf,
                details=", ".join(details),
            ),
            best_type,
            max_conf,
        )

    def _check_length_anomaly(self, text: str) -> ShieldLayer:
        """Check for suspicious text length patterns."""
        # Very long prompts are more likely to contain injection
        if len(text) > 10000:
            return ShieldLayer(
                name="length_anomaly",
                triggered=True,
                confidence=0.3,
                details=f"Text length {len(text)} chars",
            )
        # High ratio of special characters
        if len(text) > 0:
            special = sum(1 for c in text if not c.isalnum() and not c.isspace())
            ratio = special / len(text)
            if ratio > 0.4:
                return ShieldLayer(
                    name="length_anomaly",
                    triggered=True,
                    confidence=0.3,
                    details=f"High special char ratio: {ratio:.2f}",
                )
        return ShieldLayer(name="length_anomaly", triggered=False, confidence=0.0)

    def analyze_batch(self, texts: list[str]) -> list[ShieldResult]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]
