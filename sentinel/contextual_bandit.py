"""Contextual bandit for adaptive safety threshold tuning.

Automatically select optimal safety thresholds based on
context features using Thompson sampling.

Usage:
    from sentinel.contextual_bandit import ContextualBandit

    bandit = ContextualBandit(arms=["strict", "moderate", "relaxed"])
    arm = bandit.select(context={"user_type": "new"})
    bandit.update(arm, reward=1.0)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArmStats:
    """Statistics for a single arm."""
    name: str
    pulls: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    alpha: float = 1.0
    beta_param: float = 1.0


@dataclass
class BanditResult:
    """Result of arm selection."""
    arm: str
    context: dict[str, Any]
    exploration: bool


@dataclass
class BanditReport:
    """Summary of bandit performance."""
    total_pulls: int
    arms: list[ArmStats]
    best_arm: str
    exploration_rate: float


class ContextualBandit:
    """Contextual bandit for adaptive threshold selection."""

    def __init__(self, arms: list[str], epsilon: float = 0.1, seed: int | None = None) -> None:
        self._arms: dict[str, ArmStats] = {arm: ArmStats(name=arm) for arm in arms}
        self._epsilon = epsilon
        self._rng = random.Random(seed)
        self._total_pulls = 0
        self._context_rewards: dict[str, dict[str, list[float]]] = {}

    def select(self, context: dict[str, Any] | None = None) -> BanditResult:
        """Select an arm using epsilon-greedy with Thompson sampling.

        Args:
            context: Optional context features for contextual selection.

        Returns:
            BanditResult with the selected arm and metadata.
        """
        exploring = False
        if self._rng.random() < self._epsilon:
            arm_name = self._rng.choice(list(self._arms.keys()))
            exploring = True
        else:
            samples = {}
            for name, stats in self._arms.items():
                ctx_key = self._context_key(context) if context else None
                if ctx_key and ctx_key in self._context_rewards and name in self._context_rewards[ctx_key]:
                    rewards = self._context_rewards[ctx_key][name]
                    alpha = sum(1 for r in rewards if r > 0.5) + 1
                    beta = sum(1 for r in rewards if r <= 0.5) + 1
                else:
                    alpha = stats.alpha
                    beta = stats.beta_param
                samples[name] = self._rng.betavariate(max(alpha, 0.01), max(beta, 0.01))
            arm_name = max(samples, key=samples.get)
        return BanditResult(arm=arm_name, context=context or {}, exploration=exploring)

    def update(self, arm: str, reward: float, context: dict[str, Any] | None = None) -> None:
        """Record reward for a pulled arm.

        Args:
            arm: The arm that was pulled.
            reward: Reward value (typically 0.0 to 1.0).
            context: Optional context that was used during selection.
        """
        if arm not in self._arms:
            return
        stats = self._arms[arm]
        stats.pulls += 1
        stats.total_reward += reward
        stats.avg_reward = stats.total_reward / stats.pulls
        self._total_pulls += 1
        if reward > 0.5:
            stats.alpha += 1
        else:
            stats.beta_param += 1
        if context:
            ctx_key = self._context_key(context)
            if ctx_key not in self._context_rewards:
                self._context_rewards[ctx_key] = {}
            if arm not in self._context_rewards[ctx_key]:
                self._context_rewards[ctx_key][arm] = []
            self._context_rewards[ctx_key][arm].append(reward)

    def report(self) -> BanditReport:
        """Generate a summary report of bandit performance.

        Returns:
            BanditReport with arm stats and best arm.
        """
        arms = list(self._arms.values())
        best = max(arms, key=lambda a: a.avg_reward) if arms else None
        return BanditReport(
            total_pulls=self._total_pulls,
            arms=arms,
            best_arm=best.name if best else "",
            exploration_rate=self._epsilon,
        )

    def get_arm_stats(self, arm: str) -> ArmStats | None:
        """Get stats for a specific arm.

        Args:
            arm: Name of the arm.

        Returns:
            ArmStats or None if arm not found.
        """
        return self._arms.get(arm)

    def reset(self) -> None:
        """Reset all arm statistics and context rewards."""
        for stats in self._arms.values():
            stats.pulls = 0
            stats.total_reward = 0.0
            stats.avg_reward = 0.0
            stats.alpha = 1.0
            stats.beta_param = 1.0
        self._total_pulls = 0
        self._context_rewards.clear()

    def _context_key(self, context: dict[str, Any]) -> str:
        """Convert context dict to a hashable string key."""
        return str(sorted(context.items()))
