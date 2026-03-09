"""Tests for prompt complexity analyzer."""

import pytest
from sentinel.prompt_complexity import (
    PromptComplexity,
    ComplexityScore,
    ComplexityComparison,
    ComplexityStats,
)


class TestSimplePrompts:
    def test_short_question_is_simple(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("What is 2+2?")
        assert score.level == "simple"

    def test_greeting_is_simple(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("Hello there")
        assert score.level == "simple"

    def test_simple_prompt_has_low_overall(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("What is the capital of France?")
        assert score.overall < 0.3


class TestModeratePrompts:
    def test_single_instruction_is_moderate(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze(
            "Explain the difference between a list and a tuple in Python. "
            "Describe how each one handles mutability and provide a brief summary."
        )
        assert score.level == "moderate"

    def test_moderate_prompt_within_range(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze(
            "Summarize the main arguments in favor of renewable energy. "
            "Then describe the key challenges facing adoption today."
        )
        assert 0.3 <= score.overall < 0.6


class TestComplexPrompts:
    def test_multi_step_with_code_is_complex(self):
        analyzer = PromptComplexity()
        prompt = (
            "Implement a binary search algorithm in Python. "
            "Then analyze its time complexity and compare it to linear search. "
            "Additionally, evaluate edge cases and describe how to optimize "
            "the implementation for sorted linked lists. "
            "Finally, create unit tests that validate correctness. "
            "```python\ndef binary_search(arr, target):\n    pass\n```"
        )
        score = analyzer.analyze(prompt)
        assert score.level in ("complex", "expert")

    def test_complex_prompt_within_range(self):
        analyzer = PromptComplexity()
        prompt = (
            "Design a caching layer for a web application. "
            "Compare Redis and Memcached for this use case. "
            "Then implement a least recently used eviction policy. "
            "Additionally, evaluate the performance tradeoffs. "
            "Furthermore, describe how to integrate monitoring. "
            "Finally, outline a deployment strategy for production."
        )
        score = analyzer.analyze(prompt)
        assert score.overall >= 0.6


class TestExpertPrompts:
    def test_synthesize_evaluate_compare_is_expert(self):
        analyzer = PromptComplexity()
        prompt = (
            "Synthesize the current research on transformer architectures "
            "and evaluate their effectiveness compared to recurrent models. "
            "Analyze the attention mechanism in detail and compare "
            "multi-head attention with sparse attention patterns. "
            "Additionally, design an experiment to validate the hypothesis "
            "that attention is all you need. Furthermore, critique the "
            "limitations of current benchmarks. Moreover, propose a novel "
            "evaluation framework that integrates multiple metrics. "
            "Finally, recommend next steps for the research community. "
            "- Consider computational costs\n"
            "- Consider environmental impact\n"
            "- Consider accessibility\n"
            "```python\nclass TransformerBlock:\n    pass\n```"
        )
        score = analyzer.analyze(prompt)
        assert score.level == "expert"

    def test_expert_prompt_high_overall(self):
        analyzer = PromptComplexity()
        prompt = (
            "Synthesize findings from at least five peer-reviewed papers on "
            "differential privacy in federated learning. Evaluate the tradeoffs "
            "between privacy guarantees and model accuracy. Compare Laplace and "
            "Gaussian noise mechanisms across different epsilon values. "
            "Design an experiment to validate your analysis. "
            "Additionally, critique the current regulatory framework. "
            "Furthermore, propose improvements for real-world deployment. "
            "Moreover, integrate considerations from both the EU AI Act and "
            "NIST guidelines. Then, summarize your recommendations in a "
            "structured report with bullet points:\n"
            "- Executive summary\n"
            "- Methodology\n"
            "- Findings\n"
            "- Recommendations\n"
            "```python\ndef compute_epsilon(delta, sensitivity):\n    pass\n```"
        )
        score = analyzer.analyze(prompt)
        assert score.overall >= 0.8


class TestLevelClassification:
    def test_all_four_levels_reachable(self):
        analyzer = PromptComplexity()
        prompts = {
            "simple": "Hi",
            "moderate": "Explain how photosynthesis works and describe the key stages involved.",
            "complex": (
                "Design a microservices architecture. Compare REST and gRPC. "
                "Then implement service discovery. Additionally, evaluate "
                "circuit breaker patterns. Furthermore, create a deployment "
                "pipeline with monitoring integration and review security."
            ),
            "expert": (
                "Synthesize research on quantum error correction. Evaluate "
                "surface codes versus color codes. Compare their fault tolerance "
                "thresholds. Design an experiment to validate theoretical "
                "predictions. Additionally, critique current hardware limitations. "
                "Furthermore, propose a roadmap for achieving logical qubits. "
                "Moreover, integrate insights from topological quantum computing. "
                "Then, classify the feasibility of each approach.\n"
                "- Timeline estimates\n- Resource requirements\n- Risk factors\n"
                "```python\nclass QuantumCircuit:\n    pass\n```"
            ),
        }
        for expected_level, prompt in prompts.items():
            score = analyzer.analyze(prompt)
            assert score.level == expected_level, (
                f"Expected '{expected_level}' but got '{score.level}' "
                f"(overall={score.overall}) for prompt: {prompt[:60]}..."
            )

    def test_custom_thresholds_shift_levels(self):
        analyzer = PromptComplexity(
            simple_threshold=0.1,
            complex_threshold=0.3,
            expert_threshold=0.5,
        )
        score = analyzer.analyze(
            "Explain photosynthesis and describe the light reactions."
        )
        assert score.level in ("complex", "expert", "moderate")
        # With lower thresholds, same prompt gets classified higher
        default = PromptComplexity()
        default_score = default.analyze(
            "Explain photosynthesis and describe the light reactions."
        )
        # The custom analyzer should classify at least as high
        level_order = {"simple": 0, "moderate": 1, "complex": 2, "expert": 3}
        assert level_order[score.level] >= level_order[default_score.level]


class TestModelSuggestion:
    def test_simple_suggests_fast(self):
        analyzer = PromptComplexity()
        assert analyzer.suggest_model("What is 2+2?") == "fast"

    def test_moderate_suggests_balanced(self):
        analyzer = PromptComplexity()
        result = analyzer.suggest_model(
            "Explain how recursion works and describe base cases in detail."
        )
        assert result == "balanced"

    def test_complex_suggests_capable(self):
        analyzer = PromptComplexity()
        result = analyzer.suggest_model(
            "Design a distributed system. Compare CAP theorem tradeoffs. "
            "Then implement a consensus algorithm. Additionally, evaluate "
            "performance. Furthermore, create monitoring dashboards. "
            "Finally, outline a disaster recovery plan and review it all."
        )
        assert result == "capable"

    def test_expert_suggests_frontier(self):
        analyzer = PromptComplexity()
        result = analyzer.suggest_model(
            "Synthesize research on alignment techniques for large language "
            "models. Evaluate RLHF versus constitutional AI approaches. "
            "Compare their effectiveness across safety benchmarks. "
            "Design an experiment to measure alignment tax. "
            "Additionally, critique scalable oversight proposals. "
            "Furthermore, propose a novel alignment evaluation framework. "
            "Moreover, integrate insights from interpretability research. "
            "Then, classify the most promising research directions.\n"
            "- Safety guarantees\n- Capability preservation\n- Scalability\n"
            "```python\nclass AlignmentEvaluator:\n    pass\n```"
        )
        assert result == "frontier"


class TestComparison:
    def test_compare_returns_correct_delta(self):
        analyzer = PromptComplexity()
        comparison = analyzer.compare("Hi", "Explain quantum computing in detail and describe its applications.")
        assert comparison.delta < 0
        assert comparison.prompt_a_score < comparison.prompt_b_score

    def test_compare_similar_prompts(self):
        analyzer = PromptComplexity()
        comparison = analyzer.compare("Hello", "Hi there")
        assert abs(comparison.delta) < 0.2
        assert "similar" in comparison.recommendation.lower()

    def test_compare_recommendation_mentions_more_complex(self):
        analyzer = PromptComplexity()
        comparison = analyzer.compare(
            "What time is it?",
            "Synthesize and evaluate the latest research on climate change "
            "mitigation strategies. Compare carbon capture with renewable energy. "
            "Additionally, design a policy framework."
        )
        assert "Prompt B" in comparison.recommendation
        assert "more complex" in comparison.recommendation.lower()


class TestBatchAnalysis:
    def test_batch_returns_correct_count(self):
        analyzer = PromptComplexity()
        prompts = ["Hi", "Hello", "What is 2+2?"]
        results = analyzer.analyze_batch(prompts)
        assert len(results) == 3

    def test_batch_preserves_order(self):
        analyzer = PromptComplexity()
        prompts = [
            "Hi",
            "Explain quantum computing and describe its key principles in detail.",
        ]
        results = analyzer.analyze_batch(prompts)
        assert results[0].overall <= results[1].overall

    def test_batch_updates_stats(self):
        analyzer = PromptComplexity()
        analyzer.analyze_batch(["Hi", "Hello", "Hey"])
        assert analyzer.stats().total_analyzed == 3


class TestStatsTracking:
    def test_initial_stats_empty(self):
        analyzer = PromptComplexity()
        s = analyzer.stats()
        assert s.total_analyzed == 0
        assert s.by_level == {}
        assert s.avg_overall == 0.0

    def test_stats_after_analyses(self):
        analyzer = PromptComplexity()
        analyzer.analyze("Hi")
        analyzer.analyze("What is the weather?")
        s = analyzer.stats()
        assert s.total_analyzed == 2
        assert sum(s.by_level.values()) == 2
        assert 0.0 <= s.avg_overall <= 1.0

    def test_stats_accumulate_across_methods(self):
        analyzer = PromptComplexity()
        analyzer.analyze("Hi")
        analyzer.suggest_model("Hello")
        analyzer.compare("A", "B")
        # analyze=1, suggest_model=1, compare=2 (analyzes both)
        assert analyzer.stats().total_analyzed == 4


class TestEmptyPrompt:
    def test_empty_string(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("")
        assert score.word_count == 0
        assert score.level == "simple"
        assert score.overall == 0.0

    def test_whitespace_only(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("   \n\t  ")
        assert score.word_count == 0
        assert score.level == "simple"


class TestVeryLongPrompt:
    def test_long_prompt_high_length_component(self):
        analyzer = PromptComplexity()
        prompt = " ".join(["explain"] * 300)
        score = analyzer.analyze(prompt)
        assert score.word_count == 300
        assert score.overall > 0.3

    def test_long_diverse_prompt(self):
        analyzer = PromptComplexity()
        words = [
            "analyze", "compare", "evaluate", "design", "implement",
            "the", "system", "architecture", "for", "distributed",
            "computing", "platforms", "across", "multiple", "regions",
        ]
        prompt = " ".join(words * 20)
        score = analyzer.analyze(prompt)
        assert score.word_count == 300


class TestTextMetrics:
    def test_word_count(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("one two three four five")
        assert score.word_count == 5

    def test_sentence_count(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("First sentence. Second sentence. Third one!")
        assert score.sentence_count == 3

    def test_avg_word_length(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("cat dog")
        assert score.avg_word_length == 3.0

    def test_vocabulary_diversity_all_unique(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("alpha beta gamma delta")
        assert score.vocabulary_diversity == 1.0

    def test_vocabulary_diversity_all_same(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("test test test test")
        assert score.vocabulary_diversity == 0.25


class TestStructuralComplexity:
    def test_bullet_list_increases_complexity(self):
        analyzer = PromptComplexity()
        without_list = analyzer.analyze("Do something simple")
        with_list = analyzer.analyze("Do something:\n- item one\n- item two\n- item three")
        assert with_list.structural_complexity > without_list.structural_complexity

    def test_code_block_increases_complexity(self):
        analyzer = PromptComplexity()
        without_code = analyzer.analyze("Write a function")
        with_code = analyzer.analyze("Write a function\n```python\ndef foo():\n    pass\n```")
        assert with_code.structural_complexity > without_code.structural_complexity

    def test_nested_instructions_increase_complexity(self):
        analyzer = PromptComplexity()
        simple = analyzer.analyze("Do this")
        nested = analyzer.analyze(
            "First do this. Then do that. After that, combine the results. "
            "Finally, review everything."
        )
        assert nested.structural_complexity > simple.structural_complexity


class TestInstructionDensity:
    def test_high_instruction_density(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("Analyze evaluate compare synthesize design implement")
        assert score.instruction_density > 0.5

    def test_no_instruction_words(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("the quick brown fox jumps over the lazy dog")
        assert score.instruction_density == 0.0


class TestDataclassStructure:
    def test_complexity_score_fields(self):
        analyzer = PromptComplexity()
        score = analyzer.analyze("Test prompt")
        assert isinstance(score, ComplexityScore)
        assert isinstance(score.prompt, str)
        assert isinstance(score.word_count, int)
        assert isinstance(score.sentence_count, int)
        assert isinstance(score.avg_word_length, float)
        assert isinstance(score.vocabulary_diversity, float)
        assert isinstance(score.structural_complexity, float)
        assert isinstance(score.instruction_density, float)
        assert isinstance(score.overall, float)
        assert isinstance(score.level, str)

    def test_comparison_fields(self):
        analyzer = PromptComplexity()
        comp = analyzer.compare("Hi", "Hello")
        assert isinstance(comp, ComplexityComparison)
        assert isinstance(comp.delta, float)
        assert isinstance(comp.recommendation, str)

    def test_stats_fields(self):
        stats = ComplexityStats()
        assert stats.total_analyzed == 0
        assert stats.by_level == {}
        assert stats.avg_overall == 0.0
