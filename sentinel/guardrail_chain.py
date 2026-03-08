"""Composable guardrail chains for building safety pipelines.

Chain multiple safety checks together with configurable behavior
on failure (block, warn, transform). Supports both sequential
and parallel execution modes.

Usage:
    from sentinel.guardrail_chain import GuardrailChain, Step, OnFail

    chain = GuardrailChain([
        Step("sanitize", sanitize_fn, on_fail=OnFail.TRANSFORM),
        Step("pii_check", pii_fn, on_fail=OnFail.BLOCK),
        Step("injection", injection_fn, on_fail=OnFail.BLOCK),
    ])

    result = chain.run("User input here")
    if result.blocked:
        print(f"Blocked: {result.block_reason}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class OnFail(Enum):
    """Action when a guardrail step fails."""
    BLOCK = "block"       # Stop processing, return blocked
    WARN = "warn"         # Log warning, continue
    TRANSFORM = "transform"  # Apply transformation, continue
    SKIP = "skip"         # Silently skip


@dataclass
class StepResult:
    """Result of a single guardrail step."""
    name: str
    passed: bool
    duration_ms: float = 0.0
    message: str = ""
    transformed_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainResult:
    """Result of running the full guardrail chain."""
    text: str
    original: str
    blocked: bool = False
    block_reason: str = ""
    steps: list[StepResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.blocked

    @property
    def modified(self) -> bool:
        return self.text != self.original

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocked": self.blocked,
            "block_reason": self.block_reason if self.blocked else None,
            "steps_run": self.step_count,
            "warnings": len(self.warnings),
            "duration_ms": round(self.total_duration_ms, 2),
            "modified": self.modified,
        }


@dataclass
class Step:
    """A single guardrail step in the chain.

    The check function receives text and returns a tuple of
    (passed: bool, message: str, transformed_text: str | None).
    """
    name: str
    check: Callable[[str], tuple[bool, str] | tuple[bool, str, str | None]]
    on_fail: OnFail = OnFail.BLOCK
    enabled: bool = True

    def run(self, text: str) -> StepResult:
        """Execute this step."""
        start = time.perf_counter()
        result = self.check(text)

        if len(result) == 2:
            passed, message = result
            transformed = None
        else:
            passed, message, transformed = result

        duration = (time.perf_counter() - start) * 1000

        return StepResult(
            name=self.name,
            passed=passed,
            duration_ms=duration,
            message=message,
            transformed_text=transformed,
        )


class GuardrailChain:
    """Composable chain of guardrail steps.

    Steps are executed sequentially. Each step receives the (potentially
    transformed) text from previous steps. A BLOCK step that fails stops
    the chain immediately.
    """

    def __init__(self, steps: list[Step] | None = None):
        self._steps = list(steps) if steps else []

    def add(self, step: Step) -> GuardrailChain:
        """Add a step to the chain. Returns self for chaining."""
        self._steps.append(step)
        return self

    def run(self, text: str) -> ChainResult:
        """Run the full guardrail chain."""
        original = text
        current_text = text
        step_results: list[StepResult] = []
        warnings: list[str] = []
        blocked = False
        block_reason = ""
        start = time.perf_counter()

        for step in self._steps:
            if not step.enabled:
                continue

            result = step.run(current_text)
            step_results.append(result)

            if result.passed:
                # Step passed — apply any transformation
                if result.transformed_text is not None:
                    current_text = result.transformed_text
                continue

            # Step failed
            if step.on_fail == OnFail.BLOCK:
                blocked = True
                block_reason = f"[{step.name}] {result.message}"
                break

            elif step.on_fail == OnFail.WARN:
                warnings.append(f"[{step.name}] {result.message}")

            elif step.on_fail == OnFail.TRANSFORM:
                if result.transformed_text is not None:
                    current_text = result.transformed_text
                    warnings.append(
                        f"[{step.name}] Transformed: {result.message}"
                    )

            # OnFail.SKIP — do nothing

        total_duration = (time.perf_counter() - start) * 1000

        return ChainResult(
            text=current_text,
            original=original,
            blocked=blocked,
            block_reason=block_reason,
            steps=step_results,
            warnings=warnings,
            total_duration_ms=total_duration,
        )

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)
