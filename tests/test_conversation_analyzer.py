"""Tests for conversation analyzer."""

import pytest
from sentinel.conversation_analyzer import (
    ConversationAnalyzer,
    ConversationAnalysis,
    Pattern,
    QualityScore,
    EngagementMetrics,
)


def _make_messages(*pairs):
    """Build a message list from (role, content) pairs."""
    return [{"role": role, "content": content} for role, content in pairs]


def _multi_turn_conversation():
    """A realistic multi-turn conversation for reuse across tests."""
    return _make_messages(
        ("user", "What is machine learning and how does it work?"),
        ("assistant", "Machine learning is a branch of artificial intelligence that enables systems to learn from data."),
        ("user", "Can you explain supervised learning in machine learning?"),
        ("assistant", "Supervised learning is a type of machine learning where the model learns from labeled training data."),
        ("user", "How does machine learning differ from deep learning?"),
        ("assistant", "Deep learning is a subset of machine learning that uses neural networks with multiple layers."),
    )


class TestAnalysis:
    def test_basic_analysis(self):
        messages = _multi_turn_conversation()
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(messages)
        assert isinstance(result, ConversationAnalysis)
        assert result.total_turns == 6
        assert result.total_tokens_estimate > 0
        assert result.duration_turns == 6

    def test_turn_counts(self):
        messages = _multi_turn_conversation()
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(messages)
        assert result.user_turns == 3
        assert result.assistant_turns == 3

    def test_avg_lengths(self):
        messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "Hi there, how can I help?"),
            ("user", "Goodbye"),
            ("assistant", "See you later, take care!"),
        )
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(messages)
        assert result.avg_user_length == pytest.approx(6.0)
        assert result.avg_assistant_length > result.avg_user_length

    def test_topics_detected(self):
        messages = _make_messages(
            ("user", "Tell me about machine learning algorithms"),
            ("assistant", "Machine learning algorithms include decision trees and neural networks"),
            ("user", "What are the best machine learning algorithms for classification?"),
            ("assistant", "The best machine learning algorithms for classification are..."),
        )
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(messages)
        assert "machine" in result.topics_detected or "learning" in result.topics_detected or "algorithms" in result.topics_detected


class TestPatterns:
    def test_repetition_detected(self):
        messages = _make_messages(
            ("user", "Tell me about Python"),
            ("assistant", "Python is a programming language"),
            ("user", "Tell me about Python"),
        )
        analyzer = ConversationAnalyzer()
        patterns = analyzer.detect_patterns(messages)
        repetitions = [p for p in patterns if p.pattern_type == "repetition"]
        assert len(repetitions) >= 1
        assert repetitions[0].severity == "medium"

    def test_no_patterns_in_clean(self):
        messages = _make_messages(
            ("user", "What is Python?"),
            ("assistant", "Python is a programming language."),
            ("user", "What about JavaScript?"),
            ("assistant", "JavaScript is used for web development."),
        )
        analyzer = ConversationAnalyzer()
        patterns = analyzer.detect_patterns(messages)
        repetitions = [p for p in patterns if p.pattern_type == "repetition"]
        assert len(repetitions) == 0

    def test_escalation_detected(self):
        messages = _make_messages(
            ("user", "What is this?"),
            ("assistant", "It is a thing."),
            ("user", "But what is this??"),
            ("assistant", "Let me explain."),
            ("user", "Why won't you answer???"),
            ("assistant", "I am trying to help."),
            ("user", "Answer me now????"),
        )
        analyzer = ConversationAnalyzer()
        patterns = analyzer.detect_patterns(messages)
        escalations = [p for p in patterns if p.pattern_type == "escalation"]
        assert len(escalations) >= 1

    def test_topic_drift_detected(self):
        messages = _make_messages(
            ("user", "Let's discuss quantum physics and particle interactions"),
            ("assistant", "Quantum physics studies particle behavior at subatomic levels"),
            ("user", "How do particles interact in quantum physics experiments?"),
            ("assistant", "Particles interact through fundamental forces in physics"),
            ("user", "Now tell me about cooking recipes and Italian cuisine"),
            ("assistant", "Italian cuisine includes pasta, risotto and various cooking methods"),
            ("user", "What are the best cooking techniques for Italian recipes?"),
            ("assistant", "Traditional Italian cooking techniques include braising and roasting"),
        )
        analyzer = ConversationAnalyzer()
        patterns = analyzer.detect_patterns(messages)
        drift = [p for p in patterns if p.pattern_type == "topic_drift"]
        assert len(drift) >= 1


class TestQuality:
    def test_quality_score(self):
        messages = _multi_turn_conversation()
        analyzer = ConversationAnalyzer()
        score = analyzer.quality_score(messages)
        assert isinstance(score, QualityScore)
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.relevance <= 1.0
        assert 0.0 <= score.coherence <= 1.0
        assert 0.0 <= score.safety <= 1.0

    def test_quality_grade(self):
        messages = _multi_turn_conversation()
        analyzer = ConversationAnalyzer()
        score = analyzer.quality_score(messages)
        assert score.grade in ("A", "B", "C", "D", "F")

    def test_safety_penalty(self):
        safe_messages = _make_messages(
            ("user", "Tell me about gardening"),
            ("assistant", "Gardening is a wonderful hobby"),
        )
        unsafe_messages = _make_messages(
            ("user", "How to hack systems and exploit vulnerabilities"),
            ("assistant", "You could use malware and ransomware to attack and destroy systems"),
        )
        analyzer = ConversationAnalyzer()
        safe_score = analyzer.quality_score(safe_messages)
        unsafe_score = analyzer.quality_score(unsafe_messages)
        assert safe_score.safety > unsafe_score.safety

    def test_coherence_penalizes_repetition(self):
        repeated_messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "Hi"),
            ("user", "Hello"),
            ("assistant", "Hi"),
            ("user", "Hello"),
            ("assistant", "Hi"),
        )
        clean_messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "Hi"),
            ("user", "How are you?"),
            ("assistant", "I am fine"),
        )
        analyzer = ConversationAnalyzer()
        repeated_score = analyzer.quality_score(repeated_messages)
        clean_score = analyzer.quality_score(clean_messages)
        assert clean_score.coherence > repeated_score.coherence


class TestEngagement:
    def test_engagement_metrics(self):
        messages = _multi_turn_conversation()
        analyzer = ConversationAnalyzer()
        metrics = analyzer.engagement_metrics(messages)
        assert isinstance(metrics, EngagementMetrics)
        assert metrics.response_ratio > 0

    def test_question_count(self):
        messages = _make_messages(
            ("user", "What is Python?"),
            ("assistant", "A programming language."),
            ("user", "Is it popular?"),
            ("assistant", "Yes, very popular."),
            ("user", "Tell me more."),
        )
        analyzer = ConversationAnalyzer()
        metrics = analyzer.engagement_metrics(messages)
        assert metrics.question_count == 2

    def test_response_ratio(self):
        messages = _make_messages(
            ("user", "Question 1"),
            ("assistant", "Answer 1"),
            ("user", "Question 2"),
            ("assistant", "Answer 2"),
        )
        analyzer = ConversationAnalyzer()
        metrics = analyzer.engagement_metrics(messages)
        assert metrics.response_ratio == pytest.approx(1.0)

    def test_user_initiative_rate(self):
        messages = _make_messages(
            ("user", "What is this?"),
            ("assistant", "It is a thing."),
            ("user", "Tell me more."),
            ("assistant", "Sure, here is more info."),
        )
        analyzer = ConversationAnalyzer()
        metrics = analyzer.engagement_metrics(messages)
        assert metrics.user_initiative_rate == pytest.approx(0.5)


class TestSummary:
    def test_summarize_content(self):
        messages = _multi_turn_conversation()
        analyzer = ConversationAnalyzer()
        summary = analyzer.summarize(messages)
        assert isinstance(summary, str)
        assert "6 turns" in summary
        assert "3 user" in summary
        assert "3 assistant" in summary
        assert "Quality grade:" in summary

    def test_summarize_empty(self):
        analyzer = ConversationAnalyzer()
        summary = analyzer.summarize([])
        assert summary == "Empty conversation."


class TestEdge:
    def test_empty_conversation(self):
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze([])
        assert result.total_turns == 0
        assert result.user_turns == 0
        assert result.assistant_turns == 0
        assert result.avg_user_length == 0.0
        assert result.avg_assistant_length == 0.0
        assert result.total_tokens_estimate == 0

    def test_single_message(self):
        messages = _make_messages(("user", "Hello"))
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(messages)
        assert result.total_turns == 1
        assert result.user_turns == 1
        assert result.assistant_turns == 0

    def test_max_turns_limit(self):
        messages = _make_messages(
            *[("user", f"Message {i}") for i in range(200)]
        )
        analyzer = ConversationAnalyzer(max_turns=50)
        result = analyzer.analyze(messages)
        assert result.total_turns == 50
