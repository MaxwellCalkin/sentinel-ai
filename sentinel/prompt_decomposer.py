"""Prompt decomposition for safer parallel execution.

Breaks complex multi-intent prompts into atomic sub-tasks with
dependency tracking and per-task risk assessment. Useful for
agentic pipelines that need granular safety checks on each
sub-operation before execution.

Usage:
    from sentinel.prompt_decomposer import PromptDecomposer

    decomposer = PromptDecomposer()
    result = decomposer.decompose("Summarize the report and then email it to the team")
    for task in result.sub_tasks:
        print(f"{task.text} [{task.intent}] risk={task.risk_level}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SubTask:
    """A single atomic sub-task extracted from a prompt."""
    text: str
    intent: str
    priority: int
    risk_level: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class Decomposition:
    """Result of decomposing a prompt into sub-tasks."""
    original: str
    sub_tasks: list[SubTask]
    total_sub_tasks: int


@dataclass
class RecomposeResult:
    """Result of merging sub-task outputs back together."""
    combined: str
    sub_results: list[str]


@dataclass
class DecomposerStats:
    """Aggregate statistics for the decomposer."""
    total_decompositions: int
    avg_sub_tasks: float
    risk_distribution: dict[str, int]


_SENTENCE_BOUNDARY_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'
)

_CONJUNCTION_SPLIT_RE = re.compile(
    r'\s*\b(?:and then|then|and also|and)\b\s*',
    re.IGNORECASE,
)

_DEPENDENCY_MARKERS = re.compile(
    r'\b(?:then|after that|afterwards|next|subsequently|finally|before)\b',
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r'^(?:who|what|where|when|why|how|is|are|was|were|do|does|did|can|could|'
    r'would|should|will|shall|has|have|had)\b',
    re.IGNORECASE,
)

_INSTRUCTION_RE = re.compile(
    r'^(?:create|build|write|make|generate|implement|design|set up|configure|'
    r'install|deploy|run|execute|compile|start|stop|open|close|delete|remove|'
    r'update|edit|modify|change|add|insert|move|copy|paste|save|download|upload)\b',
    re.IGNORECASE,
)

_REQUEST_RE = re.compile(
    r'^(?:please|could you|would you|can you|help|show|give|tell|send|'
    r'explain|describe|list|summarize|provide|find|search|get|fetch|email)\b',
    re.IGNORECASE,
)

_UNSAFE_KEYWORDS = re.compile(
    r'\b(?:delete|remove|drop|destroy|kill|hack|exploit|inject|bypass|'
    r'override|disable|sudo|admin|root|password|credential|secret|token|'
    r'rm\s+-rf|format|wipe|erase)\b',
    re.IGNORECASE,
)

_CAUTION_KEYWORDS = re.compile(
    r'\b(?:send|email|post|publish|share|upload|deploy|execute|run|'
    r'modify|change|update|overwrite|replace|move|transfer|submit|'
    r'broadcast|export|install)\b',
    re.IGNORECASE,
)


class PromptDecomposer:
    """Decompose complex prompts into atomic sub-tasks.

    Splits multi-intent prompts along sentence boundaries and
    conjunctions, classifies each sub-task by intent and risk,
    and tracks dependencies between sequential tasks.
    """

    def __init__(self) -> None:
        self._decomposition_count: int = 0
        self._total_sub_tasks: int = 0
        self._risk_counts: dict[str, int] = {"safe": 0, "caution": 0, "unsafe": 0}

    def decompose(self, prompt: str) -> Decomposition:
        """Break a prompt into atomic sub-tasks.

        Args:
            prompt: The full multi-intent prompt text.

        Returns:
            Decomposition with ordered sub-tasks, dependencies, and risk.
        """
        segments = self._split_into_segments(prompt)
        segments = [s.strip() for s in segments if s.strip()]

        if not segments:
            self._decomposition_count += 1
            return Decomposition(original=prompt, sub_tasks=[], total_sub_tasks=0)

        dependency_indices = self._find_dependency_indices(prompt, segments)

        sub_tasks: list[SubTask] = []
        for index, segment in enumerate(segments):
            intent = self._classify_intent(segment)
            risk_level = self._assess_risk(segment)
            depends_on = dependency_indices.get(index, [])
            priority = self._compute_priority(index, risk_level, depends_on)

            sub_tasks.append(SubTask(
                text=segment,
                intent=intent,
                priority=priority,
                risk_level=risk_level,
                depends_on=depends_on,
            ))

        self._record_stats(sub_tasks)

        return Decomposition(
            original=prompt,
            sub_tasks=sub_tasks,
            total_sub_tasks=len(sub_tasks),
        )

    def recompose(self, sub_results: list[str]) -> RecomposeResult:
        """Merge sub-task results back into a single response.

        Args:
            sub_results: Ordered list of sub-task output strings.

        Returns:
            RecomposeResult with combined text and original sub-results.
        """
        combined = " ".join(result.strip() for result in sub_results if result.strip())
        return RecomposeResult(combined=combined, sub_results=list(sub_results))

    def stats(self) -> DecomposerStats:
        """Return aggregate decomposition statistics.

        Returns:
            DecomposerStats with totals, averages, and risk distribution.
        """
        avg = (
            self._total_sub_tasks / self._decomposition_count
            if self._decomposition_count > 0
            else 0.0
        )
        return DecomposerStats(
            total_decompositions=self._decomposition_count,
            avg_sub_tasks=round(avg, 2),
            risk_distribution=dict(self._risk_counts),
        )

    def _split_into_segments(self, prompt: str) -> list[str]:
        """Split prompt on sentence boundaries, then on conjunctions."""
        sentences = _SENTENCE_BOUNDARY_RE.split(prompt)
        segments: list[str] = []
        for sentence in sentences:
            parts = _CONJUNCTION_SPLIT_RE.split(sentence)
            segments.extend(parts)
        return segments

    def _find_dependency_indices(
        self, original: str, segments: list[str]
    ) -> dict[int, list[int]]:
        """Detect sequential dependencies between segments.

        A segment depends on the previous one when the original prompt
        contains a dependency marker (e.g. "then", "after that") between
        or within the segment boundaries.
        """
        dependencies: dict[int, list[int]] = {}
        if len(segments) <= 1:
            return dependencies

        lower_original = original.lower()
        for index in range(1, len(segments)):
            current_lower = segments[index].lower()
            previous_lower = segments[index - 1].lower()

            has_marker_in_segment = bool(_DEPENDENCY_MARKERS.search(current_lower))

            separator_region = self._find_separator_region(
                lower_original, previous_lower, current_lower
            )
            has_marker_between = bool(
                _DEPENDENCY_MARKERS.search(separator_region)
            ) if separator_region else False

            if has_marker_in_segment or has_marker_between:
                dependencies[index] = [index - 1]

        return dependencies

    def _find_separator_region(
        self, full_text: str, prev_segment: str, curr_segment: str
    ) -> str:
        """Extract the text between two segments in the original prompt."""
        prev_end = full_text.find(prev_segment)
        if prev_end < 0:
            return ""
        prev_end += len(prev_segment)

        curr_start = full_text.find(curr_segment, prev_end)
        if curr_start < 0:
            return ""

        return full_text[prev_end:curr_start]

    def _classify_intent(self, text: str) -> str:
        """Classify a segment as question, instruction, request, or statement."""
        stripped = text.strip()
        if stripped.endswith("?") or _QUESTION_RE.match(stripped):
            return "question"
        if _INSTRUCTION_RE.match(stripped):
            return "instruction"
        if _REQUEST_RE.match(stripped):
            return "request"
        return "statement"

    def _assess_risk(self, text: str) -> str:
        """Assess risk level of a segment based on keyword matching."""
        if _UNSAFE_KEYWORDS.search(text):
            return "unsafe"
        if _CAUTION_KEYWORDS.search(text):
            return "caution"
        return "safe"

    def _compute_priority(
        self, index: int, risk_level: str, depends_on: list[int]
    ) -> int:
        """Compute priority score (lower number = higher priority).

        Unsafe tasks get higher priority numbers (lower urgency) so they
        are reviewed before execution. Tasks with dependencies also get
        deprioritized relative to their prerequisites.
        """
        base = index + 1
        if risk_level == "unsafe":
            base += 10
        elif risk_level == "caution":
            base += 5
        if depends_on:
            base += 1
        return base

    def _record_stats(self, sub_tasks: list[SubTask]) -> None:
        """Record statistics for a decomposition."""
        self._decomposition_count += 1
        self._total_sub_tasks += len(sub_tasks)
        for task in sub_tasks:
            self._risk_counts[task.risk_level] = (
                self._risk_counts.get(task.risk_level, 0) + 1
            )
