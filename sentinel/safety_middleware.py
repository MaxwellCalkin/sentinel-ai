"""Middleware pattern for wrapping LLM calls with pre/post safety checks.

Compose a chain of middleware hooks that run before and after LLM calls,
with the ability to block, modify, or log requests and responses.

Usage:
    from sentinel.safety_middleware import SafetyMiddleware, MiddlewareHook, HookResult

    mw = SafetyMiddleware()
    mw.register(
        MiddlewareHook(name="profanity", stage="pre", priority=0),
        lambda text: HookResult(hook_name="profanity", action="block", message="blocked")
            if "badword" in text
            else HookResult(hook_name="profanity", action="pass"),
    )
    result = mw.process_input("hello")
    assert not result.blocked
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MiddlewareHook:
    """Descriptor for a registered middleware hook."""

    name: str
    stage: str  # "pre" | "post" | "both"
    priority: int = 0  # lower runs first
    enabled: bool = True


@dataclass
class HookResult:
    """Result returned by an individual hook function."""

    hook_name: str
    action: str  # "pass" | "block" | "modify" | "warn"
    message: str = ""
    modified_text: str | None = None


@dataclass
class MiddlewareResult:
    """Aggregate result from running all applicable hooks on a text."""

    original_text: str
    final_text: str
    blocked: bool
    hook_results: list[HookResult]
    elapsed_ms: float = 0.0


@dataclass
class MiddlewareStats:
    """Cumulative statistics across all middleware invocations."""

    total_processed: int = 0
    blocked_count: int = 0
    modified_count: int = 0
    pass_count: int = 0
    hooks_triggered: dict[str, int] = field(default_factory=dict)


class SafetyMiddleware:
    """Middleware pipeline for pre/post safety checks on LLM I/O."""

    def __init__(self) -> None:
        self._hooks: dict[str, tuple[MiddlewareHook, Callable[[str], HookResult]]] = {}
        self._stats = MiddlewareStats()

    # -- Registration ----------------------------------------------------------

    def register(self, hook: MiddlewareHook, fn: Callable[[str], HookResult]) -> None:
        """Register a hook function under *hook.name*."""
        self._hooks[hook.name] = (hook, fn)

    def remove(self, hook_name: str) -> None:
        """Unregister a hook. Raises KeyError if not found."""
        if hook_name not in self._hooks:
            raise KeyError(hook_name)
        del self._hooks[hook_name]

    # -- Enable / Disable ------------------------------------------------------

    def enable(self, hook_name: str) -> None:
        """Enable a previously disabled hook. Raises KeyError if not found."""
        if hook_name not in self._hooks:
            raise KeyError(hook_name)
        self._hooks[hook_name][0].enabled = True

    def disable(self, hook_name: str) -> None:
        """Disable a hook so it is skipped during processing. Raises KeyError if not found."""
        if hook_name not in self._hooks:
            raise KeyError(hook_name)
        self._hooks[hook_name][0].enabled = False

    # -- Introspection ---------------------------------------------------------

    def list_hooks(self) -> list[MiddlewareHook]:
        """Return all registered hooks sorted by priority (ascending)."""
        return sorted(
            (hook for hook, _fn in self._hooks.values()),
            key=lambda h: h.priority,
        )

    def stats(self) -> MiddlewareStats:
        """Return cumulative processing statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        self._stats = MiddlewareStats()

    # -- Processing ------------------------------------------------------------

    def process_input(self, text: str) -> MiddlewareResult:
        """Run all *pre* and *both* hooks against *text* in priority order."""
        return self._run_hooks(text, applicable_stages=("pre", "both"))

    def process_output(self, text: str) -> MiddlewareResult:
        """Run all *post* and *both* hooks against *text* in priority order."""
        return self._run_hooks(text, applicable_stages=("post", "both"))

    # -- Internal --------------------------------------------------------------

    def _run_hooks(self, text: str, applicable_stages: tuple[str, ...]) -> MiddlewareResult:
        start = time.perf_counter()
        original_text = text
        hook_results: list[HookResult] = []
        blocked = False

        ordered_hooks = self._hooks_for_stages(applicable_stages)

        for hook, fn in ordered_hooks:
            result = fn(text)
            hook_results.append(result)
            self._record_trigger(hook.name)

            if result.action == "block":
                blocked = True
                break

            if result.action == "modify" and result.modified_text is not None:
                text = result.modified_text

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._update_stats(blocked, original_text != text)

        return MiddlewareResult(
            original_text=original_text,
            final_text=text,
            blocked=blocked,
            hook_results=hook_results,
            elapsed_ms=elapsed_ms,
        )

    def _hooks_for_stages(
        self, stages: tuple[str, ...]
    ) -> list[tuple[MiddlewareHook, Callable[[str], HookResult]]]:
        """Return enabled hooks matching *stages*, sorted by priority."""
        matching = [
            (hook, fn)
            for hook, fn in self._hooks.values()
            if hook.enabled and hook.stage in stages
        ]
        matching.sort(key=lambda pair: pair[0].priority)
        return matching

    def _record_trigger(self, hook_name: str) -> None:
        counts = self._stats.hooks_triggered
        counts[hook_name] = counts.get(hook_name, 0) + 1

    def _update_stats(self, blocked: bool, modified: bool) -> None:
        self._stats.total_processed += 1
        if blocked:
            self._stats.blocked_count += 1
        elif modified:
            self._stats.modified_count += 1
        else:
            self._stats.pass_count += 1
