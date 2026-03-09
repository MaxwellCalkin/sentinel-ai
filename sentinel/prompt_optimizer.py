"""Prompt optimization for clarity, efficiency, and safety.

Analyze prompts for common issues and suggest improvements:
reduce token usage, improve clarity, add safety framing,
and remove redundancy.

Usage:
    from sentinel.prompt_optimizer import PromptOptimizer

    opt = PromptOptimizer()
    result = opt.analyze("Please can you help me to maybe possibly summarize this text?")
    print(result.suggestions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationResult:
    """Result of prompt analysis."""
    original: str
    optimized: str
    token_savings: int  # estimated tokens saved
    suggestions: list[str]
    issues: list[str]
    clarity_score: float  # 0.0 to 1.0


# Filler phrases that waste tokens
_FILLER_PHRASES = [
    (r'\b(?:please\s+)?can you\s+(?:please\s+)?', ""),
    (r'\b(?:I would like you to|I want you to|I need you to)\s+', ""),
    (r'\b(?:could you\s+(?:please\s+)?(?:kindly\s+)?)', ""),
    (r'\b(?:maybe\s+(?:you\s+could|possibly))\s+', ""),
    (r'\b(?:it would be great if you could)\s+', ""),
    (r'\bbasically\b\s*', ""),
    (r'\bactually\b\s*', ""),
    (r'\bjust\b\s+', ""),
    (r'\breally\b\s+', ""),
    (r'\bkindly\b\s+', ""),
    (r'\bsimply\b\s+', ""),
]

# Redundant patterns
_REDUNDANCY_PATTERNS = [
    (r'\b(very\s+very)\b', "very"),
    (r'\b(more\s+better)\b', "better"),
    (r'\b(repeat\s+again)\b', "repeat"),
    (r'\b(completely\s+finished)\b', "finished"),
    (r'\b(absolutely\s+essential)\b', "essential"),
]


class PromptOptimizer:
    """Analyze and optimize prompts."""

    def __init__(
        self,
        remove_filler: bool = True,
        fix_redundancy: bool = True,
        suggest_structure: bool = True,
    ) -> None:
        self._remove_filler = remove_filler
        self._fix_redundancy = fix_redundancy
        self._suggest_structure = suggest_structure

    def analyze(self, prompt: str) -> OptimizationResult:
        """Analyze a prompt and suggest optimizations."""
        optimized = prompt
        suggestions: list[str] = []
        issues: list[str] = []

        if self._remove_filler:
            for pattern, replacement in _FILLER_PHRASES:
                match = re.search(pattern, optimized, re.IGNORECASE)
                if match:
                    suggestions.append(f"Remove filler: '{match.group().strip()}'")
                    optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)

        if self._fix_redundancy:
            for pattern, fix in _REDUNDANCY_PATTERNS:
                match = re.search(pattern, optimized, re.IGNORECASE)
                if match:
                    issues.append(f"Redundancy: '{match.group()}'")
                    optimized = re.sub(pattern, fix, optimized, flags=re.IGNORECASE)

        if self._suggest_structure:
            if len(prompt) > 200 and "\n" not in prompt:
                suggestions.append("Consider breaking into sections with newlines for clarity")
            if not any(prompt.strip().lower().startswith(w) for w in ["summarize", "explain", "list", "analyze", "compare", "create", "write", "translate", "extract"]):
                if "?" not in prompt:
                    suggestions.append("Start with a clear action verb for better results")

        optimized = re.sub(r'\s+', ' ', optimized).strip()
        if optimized and optimized[0].islower():
            optimized = optimized[0].upper() + optimized[1:]

        orig_tokens = len(prompt.split())
        opt_tokens = len(optimized.split())
        savings = max(0, orig_tokens - opt_tokens)
        clarity = self._compute_clarity(optimized)

        return OptimizationResult(
            original=prompt, optimized=optimized, token_savings=savings,
            suggestions=suggestions, issues=issues, clarity_score=round(clarity, 4),
        )

    def _compute_clarity(self, text: str) -> float:
        score = 1.0
        words = text.split()
        if not words:
            return 0.0
        if len(words) > 100 and "\n" not in text:
            score -= 0.2
        vague = sum(1 for w in words if w.lower() in {"maybe", "perhaps", "possibly", "somewhat"})
        score -= min(0.3, vague * 0.1)
        action_verbs = {"summarize", "explain", "list", "analyze", "compare", "create", "write", "translate", "extract", "describe", "evaluate", "generate"}
        if words[0].lower().rstrip(".:,") in action_verbs:
            score += 0.1
        return max(0.0, min(1.0, score))

    def batch_analyze(self, prompts: list[str]) -> list[OptimizationResult]:
        return [self.analyze(p) for p in prompts]
