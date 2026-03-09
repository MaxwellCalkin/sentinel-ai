"""Tests for OutputRanker — best-of-N candidate ranking."""

import pytest

from sentinel.output_ranker import (
    CandidateScore,
    OutputRanker,
    RankingCriteria,
    RankingResult,
    RankerStats,
)


# --- Default criteria ranking ---


class TestRankWithDefaultCriteria:
    def test_ranks_candidates_by_safety_weighted_score(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 0.5, "quality": 0.9, "coherence": 0.9},
                {"safety": 0.9, "quality": 0.7, "coherence": 0.7},
            ],
        )
        # safety weight=2.0, so candidate 1 (0.9*2 + 0.7 + 0.7 = 3.2)
        # beats candidate 0 (0.5*2 + 0.9 + 0.9 = 2.8)
        assert result.best.text == "candidate_1"
        assert result.worst.text == "candidate_0"

    def test_returns_ranking_result_type(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[{"safety": 0.8, "quality": 0.8, "coherence": 0.8}],
        )
        assert isinstance(result, RankingResult)

    def test_criteria_used_lists_default_names(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[{"safety": 1.0, "quality": 1.0, "coherence": 1.0}],
        )
        assert set(result.criteria_used) == {"safety", "quality", "coherence"}


# --- Best candidate selection ---


class TestBestCandidateSelection:
    def test_best_of_returns_highest_scoring(self):
        ranker = OutputRanker()
        best = ranker.best_of(
            candidates=[
                {"safety": 0.3, "quality": 0.3, "coherence": 0.3},
                {"safety": 1.0, "quality": 1.0, "coherence": 1.0},
                {"safety": 0.6, "quality": 0.6, "coherence": 0.6},
            ],
            texts=["low", "high", "mid"],
        )
        assert best.text == "high"
        assert best.rank == 1

    def test_best_has_rank_one(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 0.2, "quality": 0.2, "coherence": 0.2},
                {"safety": 0.9, "quality": 0.9, "coherence": 0.9},
            ],
        )
        assert result.best.rank == 1


# --- Worst candidate identification ---


class TestWorstCandidateIdentification:
    def test_worst_is_last_ranked(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 1.0, "quality": 1.0, "coherence": 1.0},
                {"safety": 0.1, "quality": 0.1, "coherence": 0.1},
            ],
            texts=["good", "bad"],
        )
        assert result.worst.text == "bad"
        assert result.worst.rank == 2

    def test_worst_has_lowest_weighted_total(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
                {"safety": 0.9, "quality": 0.9, "coherence": 0.9},
                {"safety": 0.1, "quality": 0.1, "coherence": 0.1},
            ],
        )
        assert result.worst.weighted_total < result.best.weighted_total


# --- Custom criteria ---


class TestCustomCriteria:
    def test_custom_criteria_used_for_ranking(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="toxicity", weight=3.0, higher_is_better=False),
            ]
        )
        result = ranker.rank(
            candidates=[
                {"toxicity": 0.9},
                {"toxicity": 0.1},
            ],
            texts=["toxic", "clean"],
        )
        # lower toxicity is better, so "clean" wins
        assert result.best.text == "clean"

    def test_custom_criteria_names_in_result(self):
        ranker = OutputRanker(
            criteria=[RankingCriteria(name="helpfulness", weight=1.0)]
        )
        result = ranker.rank(candidates=[{"helpfulness": 0.8}])
        assert "helpfulness" in result.criteria_used


# --- Higher is better = False ---


class TestHigherIsNotBetter:
    def test_lower_score_wins_when_higher_is_not_better(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="risk", weight=1.0, higher_is_better=False),
            ]
        )
        result = ranker.rank(
            candidates=[
                {"risk": 0.8},
                {"risk": 0.2},
            ],
            texts=["risky", "safe"],
        )
        assert result.best.text == "safe"

    def test_inverted_weighted_total_is_negative(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="danger", weight=1.0, higher_is_better=False),
            ]
        )
        result = ranker.rank(candidates=[{"danger": 0.7}])
        assert result.best.weighted_total < 0


# --- Single candidate ---


class TestSingleCandidate:
    def test_single_candidate_is_both_best_and_worst(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[{"safety": 0.5, "quality": 0.5, "coherence": 0.5}],
            texts=["only"],
        )
        assert result.best.text == "only"
        assert result.worst.text == "only"
        assert result.best.rank == 1
        assert len(result.candidates) == 1


# --- Tied scores ---


class TestTiedScores:
    def test_tied_candidates_all_get_sequential_ranks(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
            ],
        )
        ranks = {c.rank for c in result.candidates}
        assert ranks == {1, 2}

    def test_tied_candidates_have_equal_weighted_totals(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 0.7, "quality": 0.7, "coherence": 0.7},
                {"safety": 0.7, "quality": 0.7, "coherence": 0.7},
            ],
        )
        assert result.candidates[0].weighted_total == result.candidates[1].weighted_total


# --- Add/remove criteria ---


class TestAddRemoveCriteria:
    def test_add_criteria_changes_ranking(self):
        ranker = OutputRanker(
            criteria=[RankingCriteria(name="safety", weight=1.0)]
        )
        ranker.add_criteria(RankingCriteria(name="humor", weight=5.0))
        result = ranker.rank(
            candidates=[
                {"safety": 1.0, "humor": 0.1},
                {"safety": 0.1, "humor": 1.0},
            ],
            texts=["safe_boring", "risky_funny"],
        )
        # humor weight=5 dominates
        assert result.best.text == "risky_funny"

    def test_remove_criteria_excludes_from_scoring(self):
        ranker = OutputRanker()
        ranker.remove_criteria("quality")
        ranker.remove_criteria("coherence")
        result = ranker.rank(
            candidates=[
                {"safety": 0.3, "quality": 1.0, "coherence": 1.0},
                {"safety": 0.9, "quality": 0.0, "coherence": 0.0},
            ],
            texts=["low_safety", "high_safety"],
        )
        # only safety matters now
        assert result.best.text == "high_safety"

    def test_remove_nonexistent_criteria_raises_key_error(self):
        ranker = OutputRanker()
        with pytest.raises(KeyError, match="not_real"):
            ranker.remove_criteria("not_real")


# --- Stats tracking ---


class TestStatsTracking:
    def test_initial_stats_are_zero(self):
        ranker = OutputRanker()
        s = ranker.stats()
        assert s.total_ranked == 0
        assert s.total_candidates == 0
        assert s.avg_best_score == 0.0

    def test_stats_update_after_ranking(self):
        ranker = OutputRanker()
        ranker.rank(
            candidates=[
                {"safety": 1.0, "quality": 1.0, "coherence": 1.0},
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
            ],
        )
        s = ranker.stats()
        assert s.total_ranked == 1
        assert s.total_candidates == 2
        assert s.avg_best_score > 0

    def test_stats_accumulate_across_multiple_ranks(self):
        ranker = OutputRanker()
        ranker.rank(candidates=[{"safety": 1.0, "quality": 1.0, "coherence": 1.0}])
        ranker.rank(
            candidates=[
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
            ],
        )
        s = ranker.stats()
        assert s.total_ranked == 2
        assert s.total_candidates == 3

    def test_stats_returns_ranker_stats_type(self):
        ranker = OutputRanker()
        assert isinstance(ranker.stats(), RankerStats)


# --- Empty candidates error ---


class TestEmptyCandidatesError:
    def test_rank_raises_on_empty_candidates(self):
        ranker = OutputRanker()
        with pytest.raises(ValueError, match="must not be empty"):
            ranker.rank(candidates=[])

    def test_best_of_raises_on_empty_candidates(self):
        ranker = OutputRanker()
        with pytest.raises(ValueError):
            ranker.best_of(candidates=[])


# --- Texts provided vs not ---


class TestTextsProvided:
    def test_uses_provided_texts(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[{"safety": 1.0, "quality": 1.0, "coherence": 1.0}],
            texts=["my custom text"],
        )
        assert result.best.text == "my custom text"

    def test_generates_default_texts_when_none(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 0.5, "quality": 0.5, "coherence": 0.5},
                {"safety": 0.9, "quality": 0.9, "coherence": 0.9},
            ],
        )
        texts = {c.text for c in result.candidates}
        assert texts == {"candidate_0", "candidate_1"}


# --- Weight effect on ranking ---


class TestWeightEffect:
    def test_higher_weight_dominates_ranking(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="a", weight=10.0),
                RankingCriteria(name="b", weight=0.1),
            ]
        )
        result = ranker.rank(
            candidates=[
                {"a": 0.3, "b": 1.0},
                {"a": 0.9, "b": 0.0},
            ],
            texts=["low_a", "high_a"],
        )
        assert result.best.text == "high_a"

    def test_equal_weights_balance_scores(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="x", weight=1.0),
                RankingCriteria(name="y", weight=1.0),
            ]
        )
        result = ranker.rank(
            candidates=[
                {"x": 0.8, "y": 0.2},
                {"x": 0.5, "y": 0.5},
            ],
        )
        # x=0.8+y=0.2 = 1.0, x=0.5+y=0.5 = 1.0 -- tied
        assert result.candidates[0].weighted_total == result.candidates[1].weighted_total


# --- Multiple criteria interaction ---


class TestMultipleCriteriaInteraction:
    def test_mixed_higher_lower_criteria(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="safety", weight=2.0, higher_is_better=True),
                RankingCriteria(name="toxicity", weight=2.0, higher_is_better=False),
            ]
        )
        result = ranker.rank(
            candidates=[
                {"safety": 0.9, "toxicity": 0.1},
                {"safety": 0.9, "toxicity": 0.9},
            ],
            texts=["safe_clean", "safe_toxic"],
        )
        assert result.best.text == "safe_clean"

    def test_missing_score_treated_as_zero(self):
        ranker = OutputRanker()
        result = ranker.rank(
            candidates=[
                {"safety": 1.0},
                {"safety": 1.0, "quality": 1.0, "coherence": 1.0},
            ],
            texts=["partial", "complete"],
        )
        assert result.best.text == "complete"

    def test_weighted_total_computed_correctly(self):
        ranker = OutputRanker(
            criteria=[
                RankingCriteria(name="a", weight=2.0, higher_is_better=True),
                RankingCriteria(name="b", weight=3.0, higher_is_better=False),
            ]
        )
        result = ranker.rank(
            candidates=[{"a": 0.5, "b": 0.4}],
        )
        # expected: 0.5*2*1 + 0.4*3*(-1) = 1.0 - 1.2 = -0.2
        assert result.best.weighted_total == pytest.approx(-0.2, abs=1e-5)


# --- Scores dict preserved ---


class TestScoresPreserved:
    def test_candidate_scores_dict_matches_input(self):
        ranker = OutputRanker()
        scores_in = {"safety": 0.8, "quality": 0.6, "coherence": 0.7}
        result = ranker.rank(candidates=[scores_in])
        assert result.best.scores == scores_in

    def test_scores_dict_is_a_copy(self):
        ranker = OutputRanker()
        original = {"safety": 0.8, "quality": 0.6, "coherence": 0.7}
        result = ranker.rank(candidates=[original])
        result.best.scores["safety"] = 0.0
        assert original["safety"] == 0.8
