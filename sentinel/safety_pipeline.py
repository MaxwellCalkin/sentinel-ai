"""Declarative pipeline builder for composing safety checks.

Build ordered processing stages that validate, transform, filter,
or log inputs and outputs. Stages execute in order and can block
or transform text as it flows through the pipeline.

Usage:
    from sentinel.safety_pipeline import SafetyPipeline, PipelineStage, StageResult

    pipeline = SafetyPipeline()
    pipeline.add_stage(
        PipelineStage(name="profanity", stage_type="validate", order=1),
        fn=lambda text: StageResult(
            stage_name="profanity",
            passed=True,
            output=text,
        ),
    )
    result = pipeline.run("Hello world")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


VALID_STAGE_TYPES = {"validate", "transform", "filter", "log"}
VALID_ACTIONS = {"none", "blocked", "transformed", "logged"}


@dataclass
class PipelineStage:
    """Definition of a single pipeline stage."""

    name: str
    stage_type: str
    order: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.stage_type not in VALID_STAGE_TYPES:
            raise ValueError(
                f"stage_type must be one of {VALID_STAGE_TYPES}, got '{self.stage_type}'"
            )


@dataclass
class StageResult:
    """Output produced by executing a single stage."""

    stage_name: str
    passed: bool
    output: str
    action_taken: str = "none"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_taken not in VALID_ACTIONS:
            raise ValueError(
                f"action_taken must be one of {VALID_ACTIONS}, got '{self.action_taken}'"
            )


@dataclass
class PipelineOutput:
    """Final result of running the full pipeline."""

    input_text: str
    output_text: str
    blocked: bool
    stage_results: list[StageResult]
    total_stages: int
    stages_passed: int


@dataclass
class PipelineConfig:
    """Configuration for pipeline behavior."""

    fail_fast: bool = True
    log_all: bool = False
    max_stages: int = 50


@dataclass
class PipelineStats:
    """Cumulative statistics across pipeline runs."""

    total_runs: int = 0
    blocked_count: int = 0
    transformed_count: int = 0
    avg_stages_passed: float = 0.0


class SafetyPipeline:
    """Declarative pipeline for composing ordered safety checks.

    Stages execute in ascending order. Each stage can pass text
    through, transform it, block the pipeline, or log observations.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._stages: dict[str, tuple[PipelineStage, Callable[[str], StageResult]]] = {}
        self._stats = PipelineStats()
        self._total_stages_passed_sum: int = 0

    def add_stage(
        self, stage: PipelineStage, fn: Callable[[str], StageResult]
    ) -> None:
        """Register a named stage with its processing function."""
        if len(self._stages) >= self._config.max_stages:
            raise ValueError(
                f"Cannot exceed max_stages ({self._config.max_stages})"
            )
        self._stages[stage.name] = (stage, fn)

    def remove_stage(self, name: str) -> None:
        """Remove a stage by name. Raises KeyError if not found."""
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not found")
        del self._stages[name]

    def enable(self, name: str) -> None:
        """Enable a stage by name. Raises KeyError if not found."""
        self._get_stage(name).enabled = True

    def disable(self, name: str) -> None:
        """Disable a stage by name. Raises KeyError if not found."""
        self._get_stage(name).enabled = False

    def run(self, text: str) -> PipelineOutput:
        """Execute all enabled stages in order against the input text."""
        sorted_stages = self._sorted_enabled_stages()
        current_text = text
        stage_results: list[StageResult] = []
        blocked = False
        stages_passed = 0

        for stage, fn in sorted_stages:
            result = fn(current_text)
            stage_results.append(result)

            if result.action_taken == "blocked":
                blocked = True
                if self._config.fail_fast:
                    break
                continue

            if result.action_taken == "transformed":
                current_text = result.output

            if result.passed:
                stages_passed += 1

        output_text = "" if blocked and self._config.fail_fast else current_text
        self._record_run(blocked, stage_results, stages_passed)

        return PipelineOutput(
            input_text=text,
            output_text=output_text,
            blocked=blocked,
            stage_results=stage_results,
            total_stages=len(stage_results),
            stages_passed=stages_passed,
        )

    def run_batch(self, texts: list[str]) -> list[PipelineOutput]:
        """Run the pipeline against multiple inputs."""
        return [self.run(text) for text in texts]

    def list_stages(self) -> list[PipelineStage]:
        """Return all stages sorted by order (ascending)."""
        stages = [stage for stage, _fn in self._stages.values()]
        return sorted(stages, key=lambda s: s.order)

    def stats(self) -> PipelineStats:
        """Return cumulative pipeline statistics."""
        return PipelineStats(
            total_runs=self._stats.total_runs,
            blocked_count=self._stats.blocked_count,
            transformed_count=self._stats.transformed_count,
            avg_stages_passed=self._stats.avg_stages_passed,
        )

    # -- Private helpers --

    def _get_stage(self, name: str) -> PipelineStage:
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not found")
        return self._stages[name][0]

    def _sorted_enabled_stages(
        self,
    ) -> list[tuple[PipelineStage, Callable[[str], StageResult]]]:
        all_stages = list(self._stages.values())
        enabled = [(s, fn) for s, fn in all_stages if s.enabled]
        return sorted(enabled, key=lambda pair: pair[0].order)

    def _record_run(
        self, blocked: bool, results: list[StageResult], stages_passed: int
    ) -> None:
        self._stats.total_runs += 1
        if blocked:
            self._stats.blocked_count += 1
        has_transform = any(r.action_taken == "transformed" for r in results)
        if has_transform:
            self._stats.transformed_count += 1
        self._total_stages_passed_sum += stages_passed
        self._stats.avg_stages_passed = (
            self._total_stages_passed_sum / self._stats.total_runs
        )
