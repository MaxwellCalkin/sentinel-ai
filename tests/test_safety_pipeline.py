"""Tests for sentinel.safety_pipeline — declarative safety pipeline builder."""

import pytest

from sentinel.safety_pipeline import (
    PipelineConfig,
    PipelineOutput,
    PipelineStage,
    PipelineStats,
    SafetyPipeline,
    StageResult,
)


# ---------------------------------------------------------------------------
# Helpers — small factory functions to reduce test boilerplate
# ---------------------------------------------------------------------------

def _pass_through_fn(stage_name: str):
    """Return a stage function that passes text through unchanged."""
    def fn(text: str) -> StageResult:
        return StageResult(stage_name=stage_name, passed=True, output=text)
    return fn


def _blocking_fn(stage_name: str):
    """Return a stage function that blocks the pipeline."""
    def fn(text: str) -> StageResult:
        return StageResult(
            stage_name=stage_name, passed=False, output=text, action_taken="blocked"
        )
    return fn


def _transform_fn(stage_name: str, transform):
    """Return a stage function that transforms text."""
    def fn(text: str) -> StageResult:
        transformed = transform(text)
        return StageResult(
            stage_name=stage_name,
            passed=True,
            output=transformed,
            action_taken="transformed",
        )
    return fn


def _logging_fn(stage_name: str):
    """Return a stage function that logs without modifying text."""
    def fn(text: str) -> StageResult:
        return StageResult(
            stage_name=stage_name, passed=True, output=text, action_taken="logged"
        )
    return fn


# ---------------------------------------------------------------------------
# PipelineStage validation
# ---------------------------------------------------------------------------

class TestPipelineStageValidation:
    def test_valid_stage_types_accepted(self):
        for stage_type in ("validate", "transform", "filter", "log"):
            stage = PipelineStage(name="s", stage_type=stage_type)
            assert stage.stage_type == stage_type

    def test_invalid_stage_type_rejected(self):
        with pytest.raises(ValueError, match="stage_type"):
            PipelineStage(name="s", stage_type="unknown")


# ---------------------------------------------------------------------------
# StageResult validation
# ---------------------------------------------------------------------------

class TestStageResultValidation:
    def test_valid_actions_accepted(self):
        for action in ("none", "blocked", "transformed", "logged"):
            result = StageResult(stage_name="s", passed=True, output="", action_taken=action)
            assert result.action_taken == action

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError, match="action_taken"):
            StageResult(stage_name="s", passed=True, output="", action_taken="bad")


# ---------------------------------------------------------------------------
# Empty pipeline
# ---------------------------------------------------------------------------

class TestEmptyPipeline:
    def test_empty_pipeline_passes_text_through(self):
        pipeline = SafetyPipeline()
        result = pipeline.run("hello")

        assert result.input_text == "hello"
        assert result.output_text == "hello"
        assert result.blocked is False
        assert result.stage_results == []
        assert result.total_stages == 0
        assert result.stages_passed == 0


# ---------------------------------------------------------------------------
# Single stage behaviors
# ---------------------------------------------------------------------------

class TestSingleValidationStage:
    def test_passing_validation_stage(self):
        pipeline = SafetyPipeline()
        stage = PipelineStage(name="check", stage_type="validate", order=1)
        pipeline.add_stage(stage, _pass_through_fn("check"))

        result = pipeline.run("safe input")

        assert result.blocked is False
        assert result.output_text == "safe input"
        assert result.stages_passed == 1
        assert len(result.stage_results) == 1
        assert result.stage_results[0].stage_name == "check"

    def test_blocking_validation_stage(self):
        pipeline = SafetyPipeline()
        stage = PipelineStage(name="blocker", stage_type="validate", order=1)
        pipeline.add_stage(stage, _blocking_fn("blocker"))

        result = pipeline.run("dangerous input")

        assert result.blocked is True
        assert result.output_text == ""
        assert result.stages_passed == 0


class TestSingleTransformStage:
    def test_transform_modifies_output_text(self):
        pipeline = SafetyPipeline()
        stage = PipelineStage(name="upper", stage_type="transform", order=1)
        pipeline.add_stage(stage, _transform_fn("upper", str.upper))

        result = pipeline.run("hello")

        assert result.output_text == "HELLO"
        assert result.blocked is False
        assert result.stages_passed == 1


# ---------------------------------------------------------------------------
# Fail-fast vs. non-fail-fast blocking
# ---------------------------------------------------------------------------

class TestBlockingBehavior:
    def test_fail_fast_stops_after_block(self):
        pipeline = SafetyPipeline(PipelineConfig(fail_fast=True))
        pipeline.add_stage(
            PipelineStage(name="blocker", stage_type="filter", order=1),
            _blocking_fn("blocker"),
        )
        pipeline.add_stage(
            PipelineStage(name="after", stage_type="validate", order=2),
            _pass_through_fn("after"),
        )

        result = pipeline.run("text")

        assert result.blocked is True
        assert len(result.stage_results) == 1
        assert result.stage_results[0].stage_name == "blocker"

    def test_no_fail_fast_continues_after_block(self):
        pipeline = SafetyPipeline(PipelineConfig(fail_fast=False))
        pipeline.add_stage(
            PipelineStage(name="blocker", stage_type="filter", order=1),
            _blocking_fn("blocker"),
        )
        pipeline.add_stage(
            PipelineStage(name="after", stage_type="validate", order=2),
            _pass_through_fn("after"),
        )

        result = pipeline.run("text")

        assert result.blocked is True
        assert len(result.stage_results) == 2
        assert result.output_text == "text"


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------

class TestStageOrdering:
    def test_stages_execute_in_order(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="second", stage_type="transform", order=20),
            _transform_fn("second", lambda t: t + "-B"),
        )
        pipeline.add_stage(
            PipelineStage(name="first", stage_type="transform", order=10),
            _transform_fn("first", lambda t: t + "-A"),
        )

        result = pipeline.run("start")

        assert result.output_text == "start-A-B"
        assert result.stage_results[0].stage_name == "first"
        assert result.stage_results[1].stage_name == "second"


# ---------------------------------------------------------------------------
# Enable / disable
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disabled_stage_is_skipped(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="skip_me", stage_type="validate", order=1),
            _blocking_fn("skip_me"),
        )
        pipeline.disable("skip_me")

        result = pipeline.run("text")

        assert result.blocked is False
        assert result.total_stages == 0

    def test_re_enabled_stage_runs(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="toggle", stage_type="validate", order=1),
            _pass_through_fn("toggle"),
        )
        pipeline.disable("toggle")
        pipeline.enable("toggle")

        result = pipeline.run("text")

        assert result.total_stages == 1

    def test_enable_missing_stage_raises(self):
        pipeline = SafetyPipeline()
        with pytest.raises(KeyError, match="not found"):
            pipeline.enable("missing")

    def test_disable_missing_stage_raises(self):
        pipeline = SafetyPipeline()
        with pytest.raises(KeyError, match="not found"):
            pipeline.disable("missing")


# ---------------------------------------------------------------------------
# Remove stage
# ---------------------------------------------------------------------------

class TestRemoveStage:
    def test_remove_existing_stage(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="temp", stage_type="validate", order=1),
            _pass_through_fn("temp"),
        )
        pipeline.remove_stage("temp")

        assert pipeline.list_stages() == []

    def test_remove_missing_stage_raises(self):
        pipeline = SafetyPipeline()
        with pytest.raises(KeyError, match="not found"):
            pipeline.remove_stage("ghost")


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

class TestBatchRun:
    def test_batch_returns_one_output_per_input(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="upper", stage_type="transform", order=1),
            _transform_fn("upper", str.upper),
        )

        results = pipeline.run_batch(["a", "b", "c"])

        assert len(results) == 3
        assert results[0].output_text == "A"
        assert results[1].output_text == "B"
        assert results[2].output_text == "C"

    def test_batch_empty_list_returns_empty(self):
        pipeline = SafetyPipeline()
        assert pipeline.run_batch([]) == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats_are_zero(self):
        pipeline = SafetyPipeline()
        s = pipeline.stats()

        assert s.total_runs == 0
        assert s.blocked_count == 0
        assert s.transformed_count == 0
        assert s.avg_stages_passed == 0.0

    def test_stats_increment_on_runs(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="v", stage_type="validate", order=1),
            _pass_through_fn("v"),
        )
        pipeline.run("a")
        pipeline.run("b")

        s = pipeline.stats()
        assert s.total_runs == 2
        assert s.blocked_count == 0
        assert s.avg_stages_passed == 1.0

    def test_stats_track_blocked(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="block", stage_type="filter", order=1),
            _blocking_fn("block"),
        )
        pipeline.run("x")

        s = pipeline.stats()
        assert s.blocked_count == 1

    def test_stats_track_transformed(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="t", stage_type="transform", order=1),
            _transform_fn("t", str.upper),
        )
        pipeline.run("x")

        s = pipeline.stats()
        assert s.transformed_count == 1

    def test_avg_stages_passed_computes_correctly(self):
        pipeline = SafetyPipeline(PipelineConfig(fail_fast=True))
        pipeline.add_stage(
            PipelineStage(name="pass", stage_type="validate", order=1),
            _pass_through_fn("pass"),
        )
        pipeline.add_stage(
            PipelineStage(name="block", stage_type="filter", order=2),
            _blocking_fn("block"),
        )
        # Run 1: pass=1 stage, then blocked => stages_passed=1
        pipeline.run("a")
        # Disable blocker for run 2 => stages_passed=1
        pipeline.disable("block")
        pipeline.run("b")

        s = pipeline.stats()
        assert s.avg_stages_passed == 1.0


# ---------------------------------------------------------------------------
# List stages
# ---------------------------------------------------------------------------

class TestListStages:
    def test_list_stages_sorted_by_order(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="c", stage_type="log", order=30),
            _logging_fn("c"),
        )
        pipeline.add_stage(
            PipelineStage(name="a", stage_type="validate", order=10),
            _pass_through_fn("a"),
        )
        pipeline.add_stage(
            PipelineStage(name="b", stage_type="transform", order=20),
            _transform_fn("b", str.upper),
        )

        names = [s.name for s in pipeline.list_stages()]
        assert names == ["a", "b", "c"]

    def test_list_stages_includes_disabled(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="off", stage_type="validate", order=1, enabled=False),
            _pass_through_fn("off"),
        )

        stages = pipeline.list_stages()
        assert len(stages) == 1
        assert stages[0].enabled is False


# ---------------------------------------------------------------------------
# Multiple stages chained
# ---------------------------------------------------------------------------

class TestMultipleStagesChained:
    def test_three_transforms_compose(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="strip", stage_type="transform", order=1),
            _transform_fn("strip", str.strip),
        )
        pipeline.add_stage(
            PipelineStage(name="upper", stage_type="transform", order=2),
            _transform_fn("upper", str.upper),
        )
        pipeline.add_stage(
            PipelineStage(name="exclaim", stage_type="transform", order=3),
            _transform_fn("exclaim", lambda t: t + "!"),
        )

        result = pipeline.run("  hello  ")

        assert result.output_text == "HELLO!"
        assert result.stages_passed == 3

    def test_validate_then_transform(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="check", stage_type="validate", order=1),
            _pass_through_fn("check"),
        )
        pipeline.add_stage(
            PipelineStage(name="clean", stage_type="transform", order=2),
            _transform_fn("clean", lambda t: t.replace("bad", "***")),
        )

        result = pipeline.run("some bad word")

        assert result.output_text == "some *** word"
        assert result.blocked is False


# ---------------------------------------------------------------------------
# Max stages enforcement
# ---------------------------------------------------------------------------

class TestMaxStages:
    def test_exceeding_max_stages_raises(self):
        pipeline = SafetyPipeline(PipelineConfig(max_stages=2))
        pipeline.add_stage(
            PipelineStage(name="a", stage_type="validate", order=1),
            _pass_through_fn("a"),
        )
        pipeline.add_stage(
            PipelineStage(name="b", stage_type="validate", order=2),
            _pass_through_fn("b"),
        )

        with pytest.raises(ValueError, match="max_stages"):
            pipeline.add_stage(
                PipelineStage(name="c", stage_type="validate", order=3),
                _pass_through_fn("c"),
            )


# ---------------------------------------------------------------------------
# Logging stage
# ---------------------------------------------------------------------------

class TestLoggingStage:
    def test_log_stage_does_not_alter_text(self):
        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="logger", stage_type="log", order=1),
            _logging_fn("logger"),
        )

        result = pipeline.run("original")

        assert result.output_text == "original"
        assert result.stage_results[0].action_taken == "logged"


# ---------------------------------------------------------------------------
# Metadata passthrough
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_stage_metadata_preserved_in_results(self):
        def fn_with_meta(text: str) -> StageResult:
            return StageResult(
                stage_name="meta",
                passed=True,
                output=text,
                metadata={"confidence": 0.95, "model": "v1"},
            )

        pipeline = SafetyPipeline()
        pipeline.add_stage(
            PipelineStage(name="meta", stage_type="validate", order=1),
            fn_with_meta,
        )

        result = pipeline.run("test")

        assert result.stage_results[0].metadata == {"confidence": 0.95, "model": "v1"}


# ---------------------------------------------------------------------------
# PipelineOutput dataclass
# ---------------------------------------------------------------------------

class TestPipelineOutput:
    def test_pipeline_output_fields(self):
        output = PipelineOutput(
            input_text="in",
            output_text="out",
            blocked=False,
            stage_results=[],
            total_stages=0,
            stages_passed=0,
        )
        assert output.input_text == "in"
        assert output.output_text == "out"


# ---------------------------------------------------------------------------
# PipelineStats dataclass
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_defaults(self):
        s = PipelineStats()
        assert s.total_runs == 0
        assert s.blocked_count == 0
        assert s.transformed_count == 0
        assert s.avg_stages_passed == 0.0
