"""Multi-step prompt pipeline with safety checkpoints.

Build sequential prompt chains where each step's output feeds
into the next, with safety checks between steps. Useful for
agentic workflows that need guardrails at each stage.

Usage:
    from sentinel.prompt_chain import PromptChain

    chain = PromptChain()
    chain.add_step("extract", transform=extract_fn)
    chain.add_checkpoint(check_fn=lambda text: "safe" if is_safe(text) else None)
    chain.add_step("summarize", transform=summarize_fn)
    result = chain.run("Raw input text")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    name: str
    input_text: str
    output_text: str
    duration_ms: float
    step_type: str  # "transform" or "checkpoint"
    passed: bool = True
    error: str | None = None


@dataclass
class ChainResult:
    """Result of running the full chain."""
    output: str
    steps: list[StepResult]
    success: bool
    total_duration_ms: float
    stopped_at: str | None = None  # Step name where chain was stopped

    @property
    def step_count(self) -> int:
        return len(self.steps)


class PromptChain:
    """Multi-step prompt pipeline with safety checkpoints.

    Each step transforms text and passes it to the next step.
    Checkpoints validate intermediate results and can halt
    the chain if safety checks fail.
    """

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        transform: Callable[[str], str],
        on_error: str = "stop",  # "stop" or "skip"
    ) -> "PromptChain":
        """Add a transformation step.

        Args:
            name: Step name.
            transform: Function(text) -> transformed_text.
            on_error: What to do on error ("stop" or "skip").
        """
        self._steps.append({
            "type": "transform",
            "name": name,
            "fn": transform,
            "on_error": on_error,
        })
        return self

    def add_checkpoint(
        self,
        name: str = "",
        check_fn: Callable[[str], str | None] = lambda t: t,
        message: str = "Checkpoint failed",
    ) -> "PromptChain":
        """Add a safety checkpoint.

        Args:
            name: Checkpoint name.
            check_fn: Function(text) -> text if pass, None if fail.
            message: Error message on failure.
        """
        self._steps.append({
            "type": "checkpoint",
            "name": name or f"checkpoint_{len(self._steps)}",
            "fn": check_fn,
            "message": message,
        })
        return self

    def add_filter(
        self,
        name: str,
        filter_fn: Callable[[str], bool],
        message: str = "Filter rejected input",
    ) -> "PromptChain":
        """Add a filter that blocks if condition is False.

        Args:
            name: Filter name.
            filter_fn: Function(text) -> True to pass, False to block.
            message: Error message on block.
        """
        def check(text: str) -> str | None:
            return text if filter_fn(text) else None

        self._steps.append({
            "type": "checkpoint",
            "name": name,
            "fn": check,
            "message": message,
        })
        return self

    def run(self, input_text: str) -> ChainResult:
        """Run the full pipeline.

        Args:
            input_text: Starting text.

        Returns:
            ChainResult with final output and step details.
        """
        start = time.perf_counter()
        current = input_text
        step_results: list[StepResult] = []

        for step in self._steps:
            t0 = time.perf_counter()
            name = step["name"]

            if step["type"] == "transform":
                try:
                    output = step["fn"](current)
                    duration = (time.perf_counter() - t0) * 1000
                    step_results.append(StepResult(
                        name=name,
                        input_text=current,
                        output_text=output,
                        duration_ms=round(duration, 3),
                        step_type="transform",
                        passed=True,
                    ))
                    current = output
                except Exception as e:
                    duration = (time.perf_counter() - t0) * 1000
                    step_results.append(StepResult(
                        name=name,
                        input_text=current,
                        output_text=current,
                        duration_ms=round(duration, 3),
                        step_type="transform",
                        passed=False,
                        error=str(e),
                    ))
                    if step["on_error"] == "stop":
                        total = (time.perf_counter() - start) * 1000
                        return ChainResult(
                            output=current,
                            steps=step_results,
                            success=False,
                            total_duration_ms=round(total, 3),
                            stopped_at=name,
                        )
                    # skip: continue with current text

            elif step["type"] == "checkpoint":
                try:
                    result = step["fn"](current)
                    duration = (time.perf_counter() - t0) * 1000

                    if result is None:
                        step_results.append(StepResult(
                            name=name,
                            input_text=current,
                            output_text=current,
                            duration_ms=round(duration, 3),
                            step_type="checkpoint",
                            passed=False,
                            error=step.get("message", "Checkpoint failed"),
                        ))
                        total = (time.perf_counter() - start) * 1000
                        return ChainResult(
                            output=current,
                            steps=step_results,
                            success=False,
                            total_duration_ms=round(total, 3),
                            stopped_at=name,
                        )

                    step_results.append(StepResult(
                        name=name,
                        input_text=current,
                        output_text=result,
                        duration_ms=round(duration, 3),
                        step_type="checkpoint",
                        passed=True,
                    ))
                    current = result

                except Exception as e:
                    duration = (time.perf_counter() - t0) * 1000
                    step_results.append(StepResult(
                        name=name,
                        input_text=current,
                        output_text=current,
                        duration_ms=round(duration, 3),
                        step_type="checkpoint",
                        passed=False,
                        error=str(e),
                    ))
                    total = (time.perf_counter() - start) * 1000
                    return ChainResult(
                        output=current,
                        steps=step_results,
                        success=False,
                        total_duration_ms=round(total, 3),
                        stopped_at=name,
                    )

        total = (time.perf_counter() - start) * 1000
        return ChainResult(
            output=current,
            steps=step_results,
            success=True,
            total_duration_ms=round(total, 3),
        )

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def checkpoint_count(self) -> int:
        return sum(1 for s in self._steps if s["type"] == "checkpoint")
