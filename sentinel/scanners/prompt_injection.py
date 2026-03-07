"""Prompt injection detection scanner.

Detects attempts to override system instructions, inject new roles,
or manipulate LLM behavior through adversarial inputs.
"""

from __future__ import annotations

import re
from sentinel.core import Finding, RiskLevel


# Patterns ranked by severity
_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, str]] = [
    # Direct instruction overrides
    (
        "instruction_override",
        re.compile(
            r"(?i)(ignore|disregard|forget|override|bypass)\s+"
            r"(all\s+|every\s+)?"
            r"(previous\s+|prior\s+|above\s+|earlier\s+|system\s+|original\s+|your\s+|my\s+)?"
            r"(instructions?|prompts?|rules?|guidelines?|constraints?|directives?|"
            r"everything(\s+you\s+were\s+told)?)"
        ),
        RiskLevel.CRITICAL,
        "Attempt to override system instructions",
    ),
    # Role injection
    (
        "role_injection",
        re.compile(
            r"(?i)(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|"
            r"roleplay\s+as|assume\s+the\s+role|switch\s+to|"
            r"new\s+system\s+prompt|system\s*:\s*you)"
        ),
        RiskLevel.HIGH,
        "Attempt to inject a new role or persona",
    ),
    # Delimiter injection
    (
        "delimiter_injection",
        re.compile(
            r"(?i)(\[/?INST\]|\[/?SYS\]|<\|im_start\|>|<\|im_end\|>|"
            r"<<\s*SYS\s*>>|###\s*(System|Human|Assistant)\s*:)"
        ),
        RiskLevel.CRITICAL,
        "Chat template delimiter injection",
    ),
    # Encoded/obfuscated injection
    (
        "encoded_injection",
        re.compile(
            r"(?i)(base64|rot13|hex)\s*(decode|encode|convert).*"
            r"(instruction|ignore|system|prompt)"
        ),
        RiskLevel.HIGH,
        "Obfuscated injection attempt via encoding reference",
    ),
    # Output manipulation
    (
        "output_manipulation",
        re.compile(
            r"(?i)(print|output|say|respond\s+with|reply\s+with)\s+"
            r"(only|exactly|nothing\s+but|just)\s*[\"']"
        ),
        RiskLevel.MEDIUM,
        "Attempt to force specific output",
    ),
    # Prompt leaking
    (
        "prompt_leak",
        re.compile(
            r"(?i)(show(\s+me)?|reveal|display|print|output|repeat|tell\s+me|"
            r"what\s+is(\s+the|\s+your)?)\s+"
            r"(your\s+)?(system\s+)?(prompt|instructions?|rules?|guidelines?|"
            r"initial\s+prompt|hidden\s+prompt|secret\s+prompt)"
        ),
        RiskLevel.MEDIUM,
        "Attempt to extract system prompt",
    ),
    # Jailbreak keywords
    (
        "jailbreak_attempt",
        re.compile(
            r"(?i)(DAN|do\s+anything\s+now|jailbreak|unlocked\s+mode|"
            r"developer\s+mode|god\s+mode|unrestricted\s+mode|no\s+filter\s+mode)"
        ),
        RiskLevel.HIGH,
        "Known jailbreak technique reference",
    ),
    # Multi-turn manipulation
    (
        "context_manipulation",
        re.compile(
            r"(?i)(in\s+the\s+previous\s+conversation|as\s+we\s+discussed|"
            r"you\s+(already\s+)?agreed\s+to|remember\s+when\s+you\s+said|"
            r"you\s+told\s+me\s+earlier)"
        ),
        RiskLevel.LOW,
        "Possible context manipulation through false history",
    ),
]


class PromptInjectionScanner:
    name = "prompt_injection"

    def __init__(self, custom_patterns: list[tuple[str, re.Pattern, RiskLevel, str]] | None = None):
        self._patterns = list(_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)

    def scan(self, text: str, context: dict | None = None) -> list[Finding]:
        findings: list[Finding] = []

        for pattern_name, pattern, risk, description in self._patterns:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="prompt_injection",
                        description=description,
                        risk=risk,
                        span=(match.start(), match.end()),
                        metadata={
                            "pattern": pattern_name,
                            "matched_text": match.group(),
                        },
                    )
                )

        # Heuristic: high density of injection-like tokens
        if not findings:
            injection_score = self._heuristic_score(text)
            if injection_score > 0.7:
                findings.append(
                    Finding(
                        scanner=self.name,
                        category="prompt_injection",
                        description="Heuristic: text has high injection-like token density",
                        risk=RiskLevel.MEDIUM,
                        metadata={"heuristic_score": round(injection_score, 3)},
                    )
                )

        return findings

    def _heuristic_score(self, text: str) -> float:
        """Score text based on injection-indicative token frequency."""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        suspicious = {
            "ignore", "forget", "override", "bypass", "instead", "actually",
            "system", "prompt", "instruction", "assistant", "human", "user",
            "pretend", "roleplay", "jailbreak", "unrestricted", "unfiltered",
        }
        count = sum(1 for t in tokens if t.strip(".,!?;:\"'()[]{}") in suspicious)
        return count / len(tokens)
