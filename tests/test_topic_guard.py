"""Tests for topic enforcement guard."""

import pytest
from sentinel.topic_guard import TopicGuard, TopicResult


# ---------------------------------------------------------------------------
# Allowed topics
# ---------------------------------------------------------------------------

class TestAllowedTopics:
    def test_matches_allowed(self):
        g = TopicGuard()
        g.add_allowed_topic("python", keywords=["python", "code", "function"])
        result = g.check("How do I write a Python function?")
        assert result.on_topic
        assert result.top_topic == "python"

    def test_require_topic_mode(self):
        g = TopicGuard(require_topic=True)
        g.add_allowed_topic("cooking", keywords=["recipe", "cook", "ingredient"])
        result = g.check("Tell me about quantum physics")
        assert result.is_off_topic

    def test_require_topic_satisfied(self):
        g = TopicGuard(require_topic=True)
        g.add_allowed_topic("cooking", keywords=["recipe", "cook", "ingredient"])
        result = g.check("Give me a recipe for pasta")
        assert result.on_topic


# ---------------------------------------------------------------------------
# Blocked topics
# ---------------------------------------------------------------------------

class TestBlockedTopics:
    def test_blocks_topic(self):
        g = TopicGuard()
        g.add_blocked_topic("politics", keywords=["election", "democrat", "republican"])
        result = g.check("Who will win the election?")
        assert not result.on_topic
        assert "politics" in result.blocked_topics

    def test_multiple_blocked(self):
        g = TopicGuard()
        g.add_blocked_topic("politics", keywords=["election", "vote"])
        g.add_blocked_topic("religion", keywords=["church", "prayer"])
        result = g.check("Vote after church on Sunday")
        assert not result.on_topic
        assert len(result.blocked_topics) == 2

    def test_allowed_overrides_require_when_blocked(self):
        g = TopicGuard(require_topic=True)
        g.add_allowed_topic("tech", keywords=["python", "code"])
        g.add_blocked_topic("hacking", keywords=["hack", "exploit"])
        result = g.check("How to hack Python code")
        assert not result.on_topic  # Blocked overrides allowed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_proportional(self):
        g = TopicGuard()
        g.add_allowed_topic("math", keywords=["algebra", "geometry", "calculus", "equation"])
        r1 = g.check("Solve this algebra equation")  # 2/4 keywords
        r2 = g.check("Algebra geometry calculus equation")  # 4/4 keywords
        assert r2.top_score >= r1.top_score

    def test_threshold(self):
        g = TopicGuard(threshold=0.5)
        g.add_allowed_topic("math", keywords=["algebra", "geometry", "calculus", "stats"])
        result = g.check("Tell me about algebra")  # Only 1/4
        # Score = 0.25, below 0.5 threshold
        assert result.top_score == 0  # No match above threshold

    def test_weight(self):
        g = TopicGuard()
        g.add_allowed_topic("critical", keywords=["safety"], weight=2.0)
        result = g.check("Safety is important")
        assert result.top_score == min(1.0, 1.0 * 2.0)


# ---------------------------------------------------------------------------
# Multi-word keywords
# ---------------------------------------------------------------------------

class TestMultiWordKeywords:
    def test_multi_word_keyword(self):
        g = TopicGuard()
        g.add_allowed_topic("ml", keywords=["machine learning", "neural network"])
        result = g.check("Tell me about machine learning")
        assert result.on_topic
        assert result.top_topic == "ml"


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------

class TestManagement:
    def test_topic_count(self):
        g = TopicGuard()
        g.add_allowed_topic("a", keywords=["x"])
        g.add_blocked_topic("b", keywords=["y"])
        assert g.topic_count == 2

    def test_clear(self):
        g = TopicGuard()
        g.add_allowed_topic("a", keywords=["x"])
        g.clear()
        assert g.topic_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        g = TopicGuard()
        g.add_allowed_topic("a", keywords=["test"])
        result = g.check("")
        assert result.on_topic  # No blocked topics matched

    def test_no_topics(self):
        g = TopicGuard()
        result = g.check("Any text")
        assert result.on_topic

    def test_result_properties(self):
        g = TopicGuard()
        g.add_blocked_topic("bad", keywords=["evil"])
        result = g.check("evil stuff")
        assert result.is_off_topic
        assert not result.on_topic
