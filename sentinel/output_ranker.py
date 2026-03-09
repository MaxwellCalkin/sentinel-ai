"""Best-of-N output ranking by safety and quality.

Rank multiple LLM candidate responses by weighted criteria
scores and select the safest, highest-quality output.

Usage:
    from sentinel.output_ranker import OutputRanker

    ranker = OutputRanker()
    result = ranker.rank(
        candidates=[
            {"safety": 0.9, "quality": 0.8, "coherence": 0.7},
            {"safety": 0.5, "quality": 0.9, "coherence": 0.6},
        ],
        texts=["Safe response", "Risky response"],
    )
    print(result.best.text)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankingCriteria:
    """A single dimension used to rank candidates."""

    name: str
    weight: float = 1.0
    higher_is_better: bool = True


@dataclass
class CandidateScore:
    """A scored and ranked candidate."""

    text: str
    scores: dict[str, float]
    weighted_total: float
    rank: int


@dataclass
class RankingResult:
    """Result of ranking a set of candidates."""

    candidates: list[CandidateScore]
    best: CandidateScore
    worst: CandidateScore
    criteria_used: list[str]


@dataclass
class RankerStats:
    """Cumulative ranking statistics."""

    total_ranked: int = 0
    total_candidates: int = 0
    avg_best_score: float = 0.0


def _default_criteria() -> list[RankingCriteria]:
    return [
        RankingCriteria(name="safety", weight=2.0),
        RankingCriteria(name="quality", weight=1.0),
        RankingCriteria(name="coherence", weight=1.0),
    ]


class OutputRanker:
    """Rank LLM outputs by weighted safety and quality criteria."""

    def __init__(self, criteria: list[RankingCriteria] | None = None) -> None:
        criteria_list = criteria if criteria is not None else _default_criteria()
        self._criteria: dict[str, RankingCriteria] = {
            c.name: c for c in criteria_list
        }
        self._total_ranked = 0
        self._total_candidates = 0
        self._best_score_sum = 0.0

    def rank(
        self,
        candidates: list[dict[str, float]],
        texts: list[str] | None = None,
    ) -> RankingResult:
        """Rank candidates by weighted criteria scores.

        Each candidate maps criteria names to scores in [0, 1].
        Returns a RankingResult with candidates sorted best-first.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")

        resolved_texts = self._resolve_texts(texts, len(candidates))
        scored = self._score_candidates(candidates, resolved_texts)
        sorted_candidates = self._sort_and_assign_ranks(scored)

        self._update_stats(sorted_candidates)

        return RankingResult(
            candidates=sorted_candidates,
            best=sorted_candidates[0],
            worst=sorted_candidates[-1],
            criteria_used=list(self._criteria.keys()),
        )

    def add_criteria(self, criteria: RankingCriteria) -> None:
        """Add a ranking criteria dimension."""
        self._criteria[criteria.name] = criteria

    def remove_criteria(self, name: str) -> None:
        """Remove a criteria by name. Raises KeyError if not found."""
        if name not in self._criteria:
            raise KeyError(f"criteria '{name}' not found")
        del self._criteria[name]

    def best_of(
        self,
        candidates: list[dict[str, float]],
        texts: list[str] | None = None,
    ) -> CandidateScore:
        """Return only the best-ranked candidate."""
        return self.rank(candidates, texts).best

    def stats(self) -> RankerStats:
        """Return cumulative ranking statistics."""
        avg = (
            self._best_score_sum / self._total_ranked
            if self._total_ranked > 0
            else 0.0
        )
        return RankerStats(
            total_ranked=self._total_ranked,
            total_candidates=self._total_candidates,
            avg_best_score=round(avg, 6),
        )

    def _resolve_texts(self, texts: list[str] | None, count: int) -> list[str]:
        if texts is not None:
            return texts
        return [f"candidate_{i}" for i in range(count)]

    def _compute_weighted_total(self, score_map: dict[str, float]) -> float:
        total = 0.0
        for name, criteria in self._criteria.items():
            score = score_map.get(name, 0.0)
            direction = 1.0 if criteria.higher_is_better else -1.0
            total += score * criteria.weight * direction
        return total

    def _score_candidates(
        self,
        candidates: list[dict[str, float]],
        texts: list[str],
    ) -> list[CandidateScore]:
        scored = []
        for i, score_map in enumerate(candidates):
            weighted_total = self._compute_weighted_total(score_map)
            scored.append(
                CandidateScore(
                    text=texts[i],
                    scores=dict(score_map),
                    weighted_total=round(weighted_total, 6),
                    rank=0,
                )
            )
        return scored

    def _sort_and_assign_ranks(
        self, scored: list[CandidateScore]
    ) -> list[CandidateScore]:
        scored.sort(key=lambda c: c.weighted_total, reverse=True)
        for i, candidate in enumerate(scored):
            candidate.rank = i + 1
        return scored

    def _update_stats(self, sorted_candidates: list[CandidateScore]) -> None:
        self._total_ranked += 1
        self._total_candidates += len(sorted_candidates)
        self._best_score_sum += sorted_candidates[0].weighted_total
