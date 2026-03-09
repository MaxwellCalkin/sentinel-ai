"""Tests for prompt optimizer."""

import pytest
from sentinel.prompt_optimizer import PromptOptimizer, OptimizationResult


class TestFillerRemoval:
    def test_remove_please_can_you(self):
        opt = PromptOptimizer()
        result = opt.analyze("Please can you summarize this text?")
        assert "please can you" not in result.optimized.lower()
        assert result.token_savings > 0

    def test_remove_i_want_you_to(self):
        opt = PromptOptimizer()
        result = opt.analyze("I want you to explain quantum physics")
        assert "I want you to" not in result.optimized

    def test_remove_basically(self):
        opt = PromptOptimizer()
        result = opt.analyze("Basically summarize the main points")
        assert "basically" not in result.optimized.lower()

    def test_filler_disabled(self):
        opt = PromptOptimizer(remove_filler=False)
        result = opt.analyze("Please can you help me?")
        assert "can you" in result.optimized.lower()


class TestRedundancy:
    def test_very_very(self):
        opt = PromptOptimizer()
        result = opt.analyze("This is very very important")
        assert "very very" not in result.optimized
        assert len(result.issues) > 0

    def test_repeat_again(self):
        opt = PromptOptimizer()
        result = opt.analyze("Please repeat again the main idea")
        assert "repeat again" not in result.optimized

    def test_no_redundancy(self):
        opt = PromptOptimizer()
        result = opt.analyze("Summarize the key findings")
        redundancy_issues = [i for i in result.issues if "Redundancy" in i]
        assert len(redundancy_issues) == 0


class TestStructure:
    def test_action_verb_suggestion(self):
        opt = PromptOptimizer()
        result = opt.analyze("The data needs to be processed and filtered")
        has_verb_suggestion = any("action verb" in s.lower() for s in result.suggestions)
        assert has_verb_suggestion


class TestClarity:
    def test_clear_prompt_high_score(self):
        opt = PromptOptimizer()
        result = opt.analyze("Summarize the key points of this research paper")
        assert result.clarity_score >= 0.8

    def test_vague_prompt_lower_score(self):
        opt = PromptOptimizer()
        result = opt.analyze("Maybe perhaps possibly do something with this maybe")
        assert result.clarity_score < 0.8

    def test_score_range(self):
        opt = PromptOptimizer()
        result = opt.analyze("Write a haiku about spring")
        assert 0.0 <= result.clarity_score <= 1.0


class TestBatch:
    def test_batch_analyze(self):
        opt = PromptOptimizer()
        results = opt.batch_analyze([
            "Please can you summarize this?",
            "Explain the concept of gravity",
        ])
        assert len(results) == 2


class TestResult:
    def test_result_structure(self):
        opt = PromptOptimizer()
        result = opt.analyze("Test prompt")
        assert isinstance(result, OptimizationResult)
        assert result.original == "Test prompt"
        assert isinstance(result.suggestions, list)
        assert isinstance(result.issues, list)
        assert result.token_savings >= 0

    def test_empty_prompt(self):
        opt = PromptOptimizer()
        result = opt.analyze("")
        assert result.clarity_score == 0.0
