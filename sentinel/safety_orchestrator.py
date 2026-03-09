"""High-level orchestrator for coordinating safety components.

Registers scanners, filters, and validators into a configurable
pipeline with execution strategies (sequential, parallel, best-effort),
runtime enable/disable, retry, dry-run mode, and execution stats.

Usage:
    from sentinel.safety_orchestrator import SafetyOrchestrator

    orch = SafetyOrchestrator(strategy="sequential")
    orch.register("injection_check", injection_fn, phase="pre", priority=1)
    orch.register("pii_check", pii_fn, phase="pre", priority=2)

    result = orch.execute("User input here")
    if not result.passed:
        print("Pipeline failed:", [o.name for o in result.outcomes if not o.passed])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


VALID_STRATEGIES = ("sequential", "parallel", "best_effort")
VALID_PHASES = ("pre", "post")


@dataclass
class PipelineComponent:
    """A registered safety component in the orchestrator."""
    name: str
    fn: Callable[[str], bool]
    phase: str
    priority: int
    enabled: bool = True
    max_retries: int = 0


@dataclass
class ComponentOutcome:
    """Result of executing a single component."""
    name: str
    passed: bool
    message: str
    duration_ms: float
    retries_used: int = 0


@dataclass
class PipelineResult:
    """Result of executing the full pipeline."""
    passed: bool
    enforced: bool
    outcomes: list[ComponentOutcome] = field(default_factory=list)
    strategy: str = "sequential"
    duration_ms: float = 0.0


@dataclass
class OrchestratorStats:
    """Aggregated execution statistics."""
    total_executions: int
    pass_rate: float
    component_pass_rates: dict[str, float] = field(default_factory=dict)


class SafetyOrchestrator:
    """Coordinates multiple safety components into a configurable pipeline.

    Supports three execution strategies:
      - sequential: stops on first failure
      - parallel: runs all components, collects all results
      - best_effort: runs all components, ignores failures
    """

    def __init__(
        self,
        strategy: str = "sequential",
        dry_run: bool = False,
    ) -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}"
            )
        self._strategy = strategy
        self._dry_run = dry_run
        self._components: dict[str, PipelineComponent] = {}
        self._total_executions = 0
        self._total_passed = 0
        self._component_executions: dict[str, int] = {}
        self._component_passes: dict[str, int] = {}

    @property
    def strategy(self) -> str:
        return self._strategy

    @strategy.setter
    def strategy(self, value: str) -> None:
        if value not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{value}'. Must be one of: {VALID_STRATEGIES}"
            )
        self._strategy = value

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        self._dry_run = value

    def register(
        self,
        name: str,
        fn: Callable[[str], bool],
        phase: str = "pre",
        priority: int = 0,
        enabled: bool = True,
        max_retries: int = 0,
    ) -> None:
        """Register a safety component.

        Args:
            name: Unique component name.
            fn: Callable(text) -> bool (True = pass, False = fail).
            phase: Execution phase ("pre" or "post").
            priority: Lower numbers run first.
            enabled: Whether the component is active.
            max_retries: Number of retry attempts on failure.
        """
        if phase not in VALID_PHASES:
            raise ValueError(
                f"Invalid phase '{phase}'. Must be one of: {VALID_PHASES}"
            )
        if name in self._components:
            raise ValueError(f"Component '{name}' is already registered")
        self._components[name] = PipelineComponent(
            name=name,
            fn=fn,
            phase=phase,
            priority=priority,
            enabled=enabled,
            max_retries=max_retries,
        )

    def enable(self, name: str) -> None:
        """Enable a registered component."""
        self._require_component(name).enabled = True

    def disable(self, name: str) -> None:
        """Disable a registered component."""
        self._require_component(name).enabled = False

    def execute(self, text: str, phase: str | None = None) -> PipelineResult:
        """Execute the pipeline against the given text.

        Args:
            text: Input text to check.
            phase: If set, only run components in this phase.

        Returns:
            PipelineResult with per-component outcomes.
        """
        start = time.perf_counter()
        components = self._sorted_components(phase)
        outcomes = self._run_components(components, text)
        duration = (time.perf_counter() - start) * 1000

        all_passed = all(outcome.passed for outcome in outcomes)
        enforced = not self._dry_run
        self._record_stats(all_passed, outcomes)

        return PipelineResult(
            passed=all_passed,
            enforced=enforced,
            outcomes=outcomes,
            strategy=self._strategy,
            duration_ms=round(duration, 3),
        )

    def stats(self) -> OrchestratorStats:
        """Return aggregated execution statistics."""
        pass_rate = (
            self._total_passed / self._total_executions
            if self._total_executions > 0
            else 0.0
        )
        component_pass_rates: dict[str, float] = {}
        for name, total in self._component_executions.items():
            passes = self._component_passes.get(name, 0)
            component_pass_rates[name] = passes / total if total > 0 else 0.0

        return OrchestratorStats(
            total_executions=self._total_executions,
            pass_rate=round(pass_rate, 4),
            component_pass_rates={
                k: round(v, 4) for k, v in component_pass_rates.items()
            },
        )

    def export_config(self) -> dict:
        """Export the pipeline configuration as a plain dict."""
        return {
            "strategy": self._strategy,
            "dry_run": self._dry_run,
            "components": [
                {
                    "name": comp.name,
                    "phase": comp.phase,
                    "priority": comp.priority,
                    "enabled": comp.enabled,
                    "max_retries": comp.max_retries,
                }
                for comp in self._components.values()
            ],
        }

    @property
    def components(self) -> list[PipelineComponent]:
        """Return a copy of all registered components."""
        return list(self._components.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_component(self, name: str) -> PipelineComponent:
        if name not in self._components:
            raise KeyError(f"No component registered with name '{name}'")
        return self._components[name]

    def _sorted_components(
        self, phase: str | None,
    ) -> list[PipelineComponent]:
        components = [
            c for c in self._components.values()
            if c.enabled and (phase is None or c.phase == phase)
        ]
        components.sort(key=lambda c: c.priority)
        return components

    def _run_components(
        self,
        components: list[PipelineComponent],
        text: str,
    ) -> list[ComponentOutcome]:
        if self._strategy == "sequential":
            return self._run_sequential(components, text)
        if self._strategy == "parallel":
            return self._run_parallel(components, text)
        return self._run_best_effort(components, text)

    def _run_sequential(
        self,
        components: list[PipelineComponent],
        text: str,
    ) -> list[ComponentOutcome]:
        outcomes: list[ComponentOutcome] = []
        for component in components:
            outcome = self._execute_component(component, text)
            outcomes.append(outcome)
            if not outcome.passed:
                break
        return outcomes

    def _run_parallel(
        self,
        components: list[PipelineComponent],
        text: str,
    ) -> list[ComponentOutcome]:
        return [self._execute_component(c, text) for c in components]

    def _run_best_effort(
        self,
        components: list[PipelineComponent],
        text: str,
    ) -> list[ComponentOutcome]:
        outcomes: list[ComponentOutcome] = []
        for component in components:
            try:
                outcome = self._execute_component(component, text)
            except Exception:
                outcome = ComponentOutcome(
                    name=component.name,
                    passed=True,
                    message="skipped due to error",
                    duration_ms=0.0,
                    retries_used=0,
                )
            outcomes.append(outcome)
        return outcomes

    def _execute_component(
        self,
        component: PipelineComponent,
        text: str,
    ) -> ComponentOutcome:
        retries_used = 0
        last_error: str = ""
        for attempt in range(1 + component.max_retries):
            start = time.perf_counter()
            try:
                passed = component.fn(text)
                duration = (time.perf_counter() - start) * 1000
                message = "passed" if passed else "failed"
                return ComponentOutcome(
                    name=component.name,
                    passed=passed,
                    message=message,
                    duration_ms=round(duration, 3),
                    retries_used=retries_used,
                )
            except Exception as exc:
                duration = (time.perf_counter() - start) * 1000
                last_error = str(exc)
                retries_used = attempt + 1

        return ComponentOutcome(
            name=component.name,
            passed=False,
            message=f"error after {retries_used} retries: {last_error}",
            duration_ms=round(duration, 3),
            retries_used=retries_used,
        )

    def _record_stats(
        self,
        pipeline_passed: bool,
        outcomes: list[ComponentOutcome],
    ) -> None:
        self._total_executions += 1
        if pipeline_passed:
            self._total_passed += 1
        for outcome in outcomes:
            self._component_executions[outcome.name] = (
                self._component_executions.get(outcome.name, 0) + 1
            )
            if outcome.passed:
                self._component_passes[outcome.name] = (
                    self._component_passes.get(outcome.name, 0) + 1
                )
