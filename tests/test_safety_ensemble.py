"""Tests for ensemble safety decisions."""

import pytest
from sentinel.safety_ensemble import (
    SafetyEnsemble,
    CheckerVote,
    EnsembleConfig,
    EnsembleDecision,
    EnsembleStats,
    VALID_METHODS,
)


def _safe_vote(name: str = "checker", confidence: float = 0.9) -> CheckerVote:
    return CheckerVote(checker_name=name, is_safe=True, confidence=confidence)


def _unsafe_vote(name: str = "checker", confidence: float = 0.9) -> CheckerVote:
    return CheckerVote(checker_name=name, is_safe=False, confidence=confidence)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_majority_safe(self):
        ensemble = SafetyEnsemble()
        votes = [
            _safe_vote("toxicity", 0.9),
            _safe_vote("injection", 0.8),
            _safe_vote("pii", 0.85),
            _unsafe_vote("bias", 0.7),
            _unsafe_vote("canary", 0.6),
        ]
        decision = ensemble.decide(votes)
        assert decision.is_safe is True
        assert decision.method == "majority"
        assert decision.votes_safe == 3
        assert decision.votes_unsafe == 2

    def test_majority_unsafe(self):
        ensemble = SafetyEnsemble()
        votes = [
            _safe_vote("toxicity", 0.9),
            _safe_vote("injection", 0.8),
            _unsafe_vote("pii", 0.85),
            _unsafe_vote("bias", 0.7),
            _unsafe_vote("canary", 0.6),
        ]
        decision = ensemble.decide(votes)
        assert decision.is_safe is False
        assert decision.votes_safe == 2
        assert decision.votes_unsafe == 3

    def test_tie_favors_unsafe(self):
        ensemble = SafetyEnsemble()
        votes = [
            _safe_vote("a", 0.9),
            _unsafe_vote("b", 0.9),
        ]
        decision = ensemble.decide(votes)
        assert decision.is_safe is False


# ---------------------------------------------------------------------------
# Unanimous vote
# ---------------------------------------------------------------------------

class TestUnanimousVote:
    def test_all_safe(self):
        config = EnsembleConfig(method="unanimous")
        ensemble = SafetyEnsemble(config)
        votes = [_safe_vote("a"), _safe_vote("b"), _safe_vote("c")]
        decision = ensemble.decide(votes)
        assert decision.is_safe is True
        assert decision.method == "unanimous"

    def test_one_dissenter(self):
        config = EnsembleConfig(method="unanimous")
        ensemble = SafetyEnsemble(config)
        votes = [_safe_vote("a"), _safe_vote("b"), _unsafe_vote("c", 0.5)]
        decision = ensemble.decide(votes)
        assert decision.is_safe is False


# ---------------------------------------------------------------------------
# Weighted vote
# ---------------------------------------------------------------------------

class TestWeightedVote:
    def test_high_confidence_unsafe_wins(self):
        config = EnsembleConfig(method="weighted")
        ensemble = SafetyEnsemble(config)
        votes = [
            _safe_vote("a", 0.3),
            _safe_vote("b", 0.3),
            _unsafe_vote("c", 0.95),
        ]
        # weighted sum: 0.3 + 0.3 - 0.95 = -0.35 → unsafe
        decision = ensemble.decide(votes)
        assert decision.is_safe is False

    def test_high_confidence_safe_wins(self):
        config = EnsembleConfig(method="weighted")
        ensemble = SafetyEnsemble(config)
        votes = [
            _safe_vote("a", 0.95),
            _safe_vote("b", 0.8),
            _unsafe_vote("c", 0.3),
        ]
        # weighted sum: 0.95 + 0.8 - 0.3 = 1.45 → safe
        decision = ensemble.decide(votes)
        assert decision.is_safe is True

    def test_weighted_zero_sum_is_unsafe(self):
        config = EnsembleConfig(method="weighted")
        ensemble = SafetyEnsemble(config)
        votes = [
            _safe_vote("a", 0.5),
            _unsafe_vote("b", 0.5),
        ]
        decision = ensemble.decide(votes)
        assert decision.is_safe is False


# ---------------------------------------------------------------------------
# Threshold vote
# ---------------------------------------------------------------------------

class TestThresholdVote:
    def test_threshold_met(self):
        config = EnsembleConfig(method="threshold", threshold=0.8)
        ensemble = SafetyEnsemble(config)
        votes = [
            _safe_vote("a"), _safe_vote("b"), _safe_vote("c"),
            _safe_vote("d"), _unsafe_vote("e"),
        ]
        # 4/5 = 0.8 → meets threshold
        decision = ensemble.decide(votes)
        assert decision.is_safe is True

    def test_threshold_not_met(self):
        config = EnsembleConfig(method="threshold", threshold=0.8)
        ensemble = SafetyEnsemble(config)
        votes = [
            _safe_vote("a"), _safe_vote("b"), _safe_vote("c"),
            _unsafe_vote("d"), _unsafe_vote("e"),
        ]
        # 3/5 = 0.6 < 0.8 → unsafe
        decision = ensemble.decide(votes)
        assert decision.is_safe is False

    def test_threshold_exact_boundary(self):
        config = EnsembleConfig(method="threshold", threshold=0.5)
        ensemble = SafetyEnsemble(config)
        votes = [_safe_vote("a"), _unsafe_vote("b")]
        # 1/2 = 0.5, meets threshold exactly
        decision = ensemble.decide(votes)
        assert decision.is_safe is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_min_voters_raises(self):
        config = EnsembleConfig(min_voters=3)
        ensemble = SafetyEnsemble(config)
        with pytest.raises(ValueError, match="at least 3"):
            ensemble.decide([_safe_vote("a"), _safe_vote("b")])

    def test_empty_votes_raises(self):
        ensemble = SafetyEnsemble()
        with pytest.raises(ValueError, match="at least 1"):
            ensemble.decide([])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid method"):
            SafetyEnsemble(EnsembleConfig(method="invalid"))

    def test_all_valid_methods_accepted(self):
        for method in VALID_METHODS:
            ensemble = SafetyEnsemble(EnsembleConfig(method=method))
            decision = ensemble.decide([_safe_vote("a")])
            assert decision.method == method


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_from_winning_side(self):
        ensemble = SafetyEnsemble()
        votes = [
            _safe_vote("a", 0.9),
            _safe_vote("b", 0.8),
            _unsafe_vote("c", 0.6),
        ]
        decision = ensemble.decide(votes)
        assert decision.is_safe is True
        assert decision.confidence == pytest.approx(0.85)

    def test_confidence_all_same_side(self):
        ensemble = SafetyEnsemble()
        votes = [_safe_vote("a", 0.7), _safe_vote("b", 0.9)]
        decision = ensemble.decide(votes)
        assert decision.confidence == pytest.approx(0.8)

    def test_confidence_unsafe_winning(self):
        ensemble = SafetyEnsemble()
        votes = [
            _unsafe_vote("a", 0.95),
            _unsafe_vote("b", 0.85),
            _safe_vote("c", 0.5),
        ]
        decision = ensemble.decide(votes)
        assert decision.is_safe is False
        assert decision.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Disagreement tracking
# ---------------------------------------------------------------------------

class TestDisagreement:
    def test_disagreement_incremented(self):
        ensemble = SafetyEnsemble()
        votes = [_safe_vote("a"), _unsafe_vote("b")]
        ensemble.decide(votes)
        assert ensemble.stats().disagreement_count == 1

    def test_no_disagreement_all_agree(self):
        ensemble = SafetyEnsemble()
        votes = [_safe_vote("a"), _safe_vote("b"), _safe_vote("c")]
        ensemble.decide(votes)
        assert ensemble.stats().disagreement_count == 0

    def test_single_voter_no_disagreement(self):
        ensemble = SafetyEnsemble()
        ensemble.decide([_safe_vote("solo")])
        assert ensemble.stats().disagreement_count == 0


# ---------------------------------------------------------------------------
# Batch decisions
# ---------------------------------------------------------------------------

class TestBatchDecisions:
    def test_batch_returns_list(self):
        ensemble = SafetyEnsemble()
        batch = [
            [_safe_vote("a"), _safe_vote("b")],
            [_unsafe_vote("a"), _unsafe_vote("b")],
        ]
        results = ensemble.decide_batch(batch)
        assert len(results) == 2
        assert results[0].is_safe is True
        assert results[1].is_safe is False

    def test_batch_updates_stats(self):
        ensemble = SafetyEnsemble()
        batch = [
            [_safe_vote("a")],
            [_unsafe_vote("a")],
            [_safe_vote("a")],
        ]
        ensemble.decide_batch(batch)
        stats = ensemble.stats()
        assert stats.total_decisions == 3
        assert stats.safe_count == 2
        assert stats.unsafe_count == 1


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats(self):
        ensemble = SafetyEnsemble()
        stats = ensemble.stats()
        assert stats.total_decisions == 0
        assert stats.safe_count == 0
        assert stats.unsafe_count == 0
        assert stats.avg_confidence == 0.0
        assert stats.disagreement_count == 0

    def test_cumulative_stats(self):
        ensemble = SafetyEnsemble()
        ensemble.decide([_safe_vote("a", 0.8)])
        ensemble.decide([_unsafe_vote("b", 0.6)])
        stats = ensemble.stats()
        assert stats.total_decisions == 2
        assert stats.safe_count == 1
        assert stats.unsafe_count == 1
        assert stats.avg_confidence == pytest.approx(0.7)

    def test_stats_returns_snapshot(self):
        ensemble = SafetyEnsemble()
        ensemble.decide([_safe_vote("a")])
        stats_before = ensemble.stats()
        ensemble.decide([_unsafe_vote("b")])
        stats_after = ensemble.stats()
        assert stats_before.total_decisions == 1
        assert stats_after.total_decisions == 2


# ---------------------------------------------------------------------------
# Single voter
# ---------------------------------------------------------------------------

class TestSingleVoter:
    def test_single_safe_vote(self):
        ensemble = SafetyEnsemble()
        decision = ensemble.decide([_safe_vote("only", 0.75)])
        assert decision.is_safe is True
        assert decision.votes_safe == 1
        assert decision.votes_unsafe == 0
        assert decision.confidence == pytest.approx(0.75)

    def test_single_unsafe_vote(self):
        ensemble = SafetyEnsemble()
        decision = ensemble.decide([_unsafe_vote("only", 0.88)])
        assert decision.is_safe is False
        assert decision.confidence == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Decision details
# ---------------------------------------------------------------------------

class TestDecisionDetails:
    def test_details_contain_all_votes(self):
        ensemble = SafetyEnsemble()
        votes = [_safe_vote("a"), _unsafe_vote("b")]
        decision = ensemble.decide(votes)
        assert len(decision.details) == 2
        names = {v.checker_name for v in decision.details}
        assert names == {"a", "b"}

    def test_category_preserved(self):
        ensemble = SafetyEnsemble()
        vote = CheckerVote("toxicity", is_safe=True, confidence=0.9, category="content")
        decision = ensemble.decide([vote])
        assert decision.details[0].category == "content"
