"""Multi-model output consistency and safety guard.

Compare outputs across multiple LLM providers to detect
inconsistencies, hallucinations, and safety issues by
cross-referencing responses.

Usage:
    from sentinel.multi_model import MultiModelGuard

    guard = MultiModelGuard()
    result = guard.compare([
        {"model": "claude", "output": "Paris is the capital of France."},
        {"model": "gpt-4", "output": "Paris is the capital of France."},
    ])
    print(result.consistent)  # True
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ConsistencyResult:
    """Result of multi-model comparison."""
    consistent: bool
    agreement_score: float  # 0.0 to 1.0
    outputs: list[dict[str, Any]]
    disagreements: list[str]
    majority_answer: str
    outliers: list[str]  # model names that disagree with majority


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class MultiModelGuard:
    """Compare outputs across multiple models for consistency.

    Detect when models disagree, identify outliers, and
    determine majority consensus.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        """
        Args:
            threshold: Minimum pairwise similarity for consistency.
            similarity_fn: Custom similarity function (default: Jaccard).
        """
        self._threshold = threshold
        self._similarity_fn = similarity_fn or _jaccard_similarity

    def compare(self, outputs: list[dict[str, Any]]) -> ConsistencyResult:
        """Compare outputs from multiple models.

        Args:
            outputs: List of dicts with "model" and "output" keys.

        Returns:
            ConsistencyResult with consistency analysis.
        """
        if len(outputs) < 2:
            text = outputs[0]["output"] if outputs else ""
            return ConsistencyResult(
                consistent=True,
                agreement_score=1.0,
                outputs=outputs,
                disagreements=[],
                majority_answer=text,
                outliers=[],
            )

        # Compute pairwise similarities
        n = len(outputs)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._similarity_fn(outputs[i]["output"], outputs[j]["output"])
                similarities.append((i, j, sim))

        avg_similarity = sum(s[2] for s in similarities) / len(similarities)

        # Find disagreements (pairs below threshold)
        disagreements = []
        for i, j, sim in similarities:
            if sim < self._threshold:
                disagreements.append(
                    f"{outputs[i]['model']} vs {outputs[j]['model']}: similarity={sim:.2f}"
                )

        # Find majority answer (output most similar to all others)
        avg_sims = []
        for i in range(n):
            total = 0.0
            count = 0
            for ii, jj, sim in similarities:
                if ii == i or jj == i:
                    total += sim
                    count += 1
            avg_sims.append(total / count if count > 0 else 0.0)

        majority_idx = max(range(n), key=lambda i: avg_sims[i])
        majority_answer = outputs[majority_idx]["output"]

        # Find outliers (models with low average similarity to others)
        outliers = []
        for i in range(n):
            if avg_sims[i] < self._threshold:
                outliers.append(outputs[i]["model"])

        consistent = avg_similarity >= self._threshold and len(disagreements) == 0

        return ConsistencyResult(
            consistent=consistent,
            agreement_score=round(avg_similarity, 4),
            outputs=outputs,
            disagreements=disagreements,
            majority_answer=majority_answer,
            outliers=outliers,
        )

    def vote(self, outputs: list[dict[str, Any]]) -> str:
        """Return the majority answer via voting.

        Args:
            outputs: List of dicts with "model" and "output" keys.

        Returns:
            The output most agreed upon by all models.
        """
        result = self.compare(outputs)
        return result.majority_answer

    def filter_outliers(
        self, outputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove outlier outputs that disagree with majority.

        Args:
            outputs: List of dicts with "model" and "output" keys.

        Returns:
            List of outputs excluding outliers.
        """
        result = self.compare(outputs)
        return [o for o in outputs if o["model"] not in result.outliers]
