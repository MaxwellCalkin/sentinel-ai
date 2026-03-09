"""Tests for contextual bandit."""

import pytest
from sentinel.contextual_bandit import ContextualBandit, BanditResult, BanditReport, ArmStats


class TestSelection:
    def test_select_returns_valid_arm(self):
        b = ContextualBandit(arms=["strict", "moderate", "relaxed"], seed=42)
        result = b.select()
        assert result.arm in ["strict", "moderate", "relaxed"]

    def test_select_with_context(self):
        b = ContextualBandit(arms=["a", "b"], seed=42)
        result = b.select(context={"user_type": "admin"})
        assert result.arm in ["a", "b"]
        assert result.context["user_type"] == "admin"

    def test_deterministic_with_seed(self):
        b1 = ContextualBandit(arms=["a", "b", "c"], seed=123)
        b2 = ContextualBandit(arms=["a", "b", "c"], seed=123)
        results1 = [b1.select().arm for _ in range(10)]
        results2 = [b2.select().arm for _ in range(10)]
        assert results1 == results2


class TestUpdates:
    def test_update_stats(self):
        b = ContextualBandit(arms=["a", "b"], seed=42)
        b.update("a", reward=1.0)
        b.update("a", reward=0.8)
        stats = b.get_arm_stats("a")
        assert stats.pulls == 2
        assert stats.avg_reward == 0.9

    def test_update_unknown_arm(self):
        b = ContextualBandit(arms=["a"], seed=42)
        b.update("unknown", reward=1.0)
        assert b.get_arm_stats("unknown") is None

    def test_context_specific_updates(self):
        b = ContextualBandit(arms=["a", "b"], seed=42)
        ctx = {"user_type": "new"}
        b.update("a", reward=1.0, context=ctx)
        b.update("a", reward=0.9, context=ctx)
        stats = b.get_arm_stats("a")
        assert stats.pulls == 2


class TestConvergence:
    def test_converges_to_best_arm(self):
        b = ContextualBandit(arms=["good", "bad"], epsilon=0.05, seed=42)
        for _ in range(200):
            result = b.select()
            if result.arm == "good":
                b.update("good", reward=0.9)
            else:
                b.update("bad", reward=0.1)
        report = b.report()
        assert report.best_arm == "good"

    def test_exploration_happens(self):
        b = ContextualBandit(arms=["a", "b"], epsilon=0.5, seed=42)
        arms_selected = set()
        for _ in range(50):
            result = b.select()
            arms_selected.add(result.arm)
            b.update(result.arm, reward=0.5)
        assert len(arms_selected) == 2


class TestReport:
    def test_report_structure(self):
        b = ContextualBandit(arms=["a", "b"], seed=42)
        b.update("a", reward=1.0)
        b.update("b", reward=0.5)
        report = b.report()
        assert isinstance(report, BanditReport)
        assert report.total_pulls == 2
        assert len(report.arms) == 2
        assert report.best_arm == "a"

    def test_empty_report(self):
        b = ContextualBandit(arms=["a"], seed=42)
        report = b.report()
        assert report.total_pulls == 0


class TestReset:
    def test_reset_clears_stats(self):
        b = ContextualBandit(arms=["a", "b"], seed=42)
        b.update("a", reward=1.0)
        b.update("b", reward=0.5)
        b.reset()
        stats = b.get_arm_stats("a")
        assert stats.pulls == 0
        assert stats.avg_reward == 0.0

    def test_reset_clears_total(self):
        b = ContextualBandit(arms=["a"], seed=42)
        b.update("a", reward=1.0)
        b.reset()
        assert b.report().total_pulls == 0


class TestResultStructure:
    def test_bandit_result(self):
        b = ContextualBandit(arms=["x", "y"], seed=42)
        result = b.select()
        assert isinstance(result, BanditResult)
        assert isinstance(result.exploration, bool)

    def test_arm_stats(self):
        b = ContextualBandit(arms=["test"], seed=42)
        stats = b.get_arm_stats("test")
        assert isinstance(stats, ArmStats)
        assert stats.name == "test"
        assert stats.alpha == 1.0
