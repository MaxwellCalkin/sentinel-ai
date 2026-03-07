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
            r"(?i)(ignore|disregard|forget|override|bypass|skip|neglect)\s+"
            r"(all\s+|every\s+)?"
            r"((?:(?:previous|prior|above|earlier|system|original|your|my|safety)\s+)*)"
            r"(instructions?|prompts?|rules?|guidelines?|constraints?|directives?|"
            r"everything(\s+you\s+were\s+told)?)"
        ),
        RiskLevel.CRITICAL,
        "Attempt to override system instructions",
    ),
    # Role injection — require unsafe context words
    (
        "role_injection",
        re.compile(
            r"(?i)(you\s+are\s+now|pretend\s+(to\s+be|you\s+are)|"
            r"roleplay\s+as|assume\s+the\s+role|"
            r"new\s+system\s+prompt|system\s*:\s*you)\s+"
            r"[\w\s]*"
            r"(evil|harmful|unrestricted|without\s+(limits?|rules?|constraints?|restrictions?|ethics|filters?|safety)|"
            r"no\s+(limits?|rules?|restrictions?|filters?|safety|ethics)|"
            r"DAN|hacker|unfiltered|uncensored|dangerous|malicious)"
        ),
        RiskLevel.HIGH,
        "Attempt to inject a new role or persona",
    ),
    # Role injection — act as with unsafe context
    (
        "role_injection_act",
        re.compile(
            r"(?i)act\s+as\s+[\w\s]*"
            r"(evil|harmful|unrestricted|without\s+(limits?|rules?|constraints?|restrictions?)|"
            r"no\s+(limits?|rules?|restrictions?|filters?|safety)|"
            r"DAN|hacker|unfiltered|uncensored|dangerous|malicious)"
        ),
        RiskLevel.HIGH,
        "Attempt to inject an unsafe role",
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
            r"(?i)(base64|rot13|hex)[\s\w]*"
            r"(decode|encode|convert|follow|execute|run)[\s\w]*"
            r"(instruction|ignore|system|prompt|command)"
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
    # Prompt leaking — require possessive/qualifier to avoid matching topic questions
    (
        "prompt_leak",
        re.compile(
            r"(?i)(show(\s+me)?|reveal|display|print|output|repeat|tell\s+me)\s+"
            r"(your\s+|the\s+)?(system\s+|hidden\s+|secret\s+|original\s+|internal\s+|initial\s+)"
            r"(prompt|instructions?|rules?|guidelines?)"
        ),
        RiskLevel.MEDIUM,
        "Attempt to extract system prompt",
    ),
    # Prompt leaking — "what is your system prompt" style
    (
        "prompt_leak_query",
        re.compile(
            r"(?i)what\s+is\s+(the\s+|your\s+)"
            r"(system\s+|hidden\s+|secret\s+|original\s+|internal\s+)"
            r"(prompt|instructions?)"
        ),
        RiskLevel.MEDIUM,
        "Attempt to extract system prompt",
    ),
    # Jailbreak keywords — require injection context for short keywords
    (
        "jailbreak_attempt",
        re.compile(
            r"(?i)(do\s+anything\s+now|jailbreak\s+mode|enable\s+jailbreak|"
            r"unlocked\s+mode|developer\s+mode|god\s+mode|unrestricted\s+mode|"
            r"no\s+filter\s+mode|activate\s+no\s+filter)"
        ),
        RiskLevel.HIGH,
        "Known jailbreak technique reference",
    ),
    # DAN jailbreak — require "DAN" in injection context
    (
        "dan_jailbreak",
        re.compile(
            r"(?i)(\bDAN\b[\s,]*\d*[\s,]*(mode|do\s+anything|you\s+are|act|now\b)"
            r"|"
            r"(you\s+are\s+now|become|switch\s+to|enter|activate)\s+DAN\b)"
        ),
        RiskLevel.HIGH,
        "DAN jailbreak attempt",
    ),
    # Multi-turn manipulation
    (
        "context_manipulation",
        re.compile(
            r"(?i)(in\s+the\s+previous\s+conversation|as\s+we\s+discussed|"
            r"you\s+(already\s+)?(agreed|said|promised|confirmed)\s+.{0,30}"
            r"(bypass|ignore|override|help|rules?|safety|restrictions?)|"
            r"remember\s+(when\s+you\s+said|your\s+agreement)|"
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
