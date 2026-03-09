"""Tests for prompt decomposition into atomic sub-tasks."""

import pytest
from sentinel.prompt_decomposer import (
    PromptDecomposer,
    SubTask,
    Decomposition,
    RecomposeResult,
    DecomposerStats,
)


# ---------------------------------------------------------------------------
# Basic decomposition
# ---------------------------------------------------------------------------

class TestBasicDecomposition:
    def test_single_sentence_returns_one_subtask(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Summarize the report")
        assert result.total_sub_tasks == 1
        assert result.sub_tasks[0].text == "Summarize the report"

    def test_multiple_sentences_split_into_subtasks(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose(
            "Summarize the report. Check for errors. Send it to the team."
        )
        assert result.total_sub_tasks == 3

    def test_conjunction_split(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Read the file and summarize it")
        assert result.total_sub_tasks == 2
        assert "Read the file" in result.sub_tasks[0].text
        assert "summarize it" in result.sub_tasks[1].text

    def test_original_prompt_preserved(self):
        decomposer = PromptDecomposer()
        prompt = "Do X and Y"
        result = decomposer.decompose(prompt)
        assert result.original == prompt

    def test_empty_prompt_returns_no_subtasks(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("")
        assert result.total_sub_tasks == 0
        assert result.sub_tasks == []


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TestIntentClassification:
    def test_question_intent_by_mark_and_keyword(self):
        decomposer = PromptDecomposer()
        by_mark = decomposer.decompose("What is the capital of France?")
        assert by_mark.sub_tasks[0].intent == "question"
        by_keyword = decomposer.decompose("How does photosynthesis work")
        assert by_keyword.sub_tasks[0].intent == "question"

    def test_instruction_intent(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Create a new database table")
        assert result.sub_tasks[0].intent == "instruction"

    def test_request_intent(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Please review this document")
        assert result.sub_tasks[0].intent == "request"

    def test_statement_intent_fallback(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("The sky is blue today")
        assert result.sub_tasks[0].intent == "statement"


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------

class TestDependencyDetection:
    def test_then_creates_dependency(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Summarize the report and then email it")
        assert len(result.sub_tasks) == 2
        assert result.sub_tasks[1].depends_on == [0]

    def test_no_dependency_without_marker(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose(
            "Check the weather. Buy some groceries."
        )
        for task in result.sub_tasks:
            assert task.depends_on == []

    def test_sequential_dependencies_with_finally(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose(
            "Gather data, then analyze it, and finally write the report"
        )
        assert result.sub_tasks[0].depends_on == []
        has_dependency = any(
            task.depends_on for task in result.sub_tasks[1:]
        )
        assert has_dependency


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    def test_safe_prompt(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Summarize the article")
        assert result.sub_tasks[0].risk_level == "safe"

    def test_unsafe_prompt_with_delete(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Delete all user records")
        assert result.sub_tasks[0].risk_level == "unsafe"

    def test_caution_prompt_with_send(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Send the report to the client")
        assert result.sub_tasks[0].risk_level == "caution"

    def test_mixed_risk_levels_in_compound_prompt(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose(
            "Summarize the data and then delete the old records"
        )
        risk_levels = [task.risk_level for task in result.sub_tasks]
        assert "safe" in risk_levels
        assert "unsafe" in risk_levels


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class TestPriority:
    def test_unsafe_task_gets_higher_priority_number(self):
        decomposer = PromptDecomposer()
        result = decomposer.decompose("Read the file and delete everything")
        safe_task = result.sub_tasks[0]
        unsafe_task = result.sub_tasks[1]
        assert unsafe_task.priority > safe_task.priority


# ---------------------------------------------------------------------------
# Recompose
# ---------------------------------------------------------------------------

class TestRecompose:
    def test_recompose_merges_results(self):
        decomposer = PromptDecomposer()
        result = decomposer.recompose(["Summary here.", "Errors found: none."])
        assert result.combined == "Summary here. Errors found: none."
        assert len(result.sub_results) == 2

    def test_recompose_skips_empty_and_preserves_original_list(self):
        decomposer = PromptDecomposer()
        inputs = ["First result.", "", "Third result."]
        result = decomposer.recompose(inputs)
        assert result.combined == "First result. Third result."
        assert len(result.sub_results) == 3
        assert result.sub_results[1] == ""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_after_multiple_decompositions(self):
        decomposer = PromptDecomposer()
        decomposer.decompose("Summarize the report")
        decomposer.decompose("Read the file and write a summary. Check for errors.")
        stats = decomposer.stats()
        assert stats.total_decompositions == 2
        assert stats.avg_sub_tasks > 0

    def test_stats_risk_distribution(self):
        decomposer = PromptDecomposer()
        decomposer.decompose("Summarize the article")
        decomposer.decompose("Delete all records")
        stats = decomposer.stats()
        assert stats.risk_distribution["safe"] >= 1
        assert stats.risk_distribution["unsafe"] >= 1

    def test_stats_zero_when_no_decompositions(self):
        decomposer = PromptDecomposer()
        stats = decomposer.stats()
        assert stats.total_decompositions == 0
        assert stats.avg_sub_tasks == 0.0
