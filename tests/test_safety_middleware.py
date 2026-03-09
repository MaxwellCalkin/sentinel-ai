"""Tests for sentinel.safety_middleware — SafetyMiddleware pipeline."""

from __future__ import annotations

import pytest

from sentinel.safety_middleware import (
    HookResult,
    MiddlewareHook,
    MiddlewareResult,
    MiddlewareStats,
    SafetyMiddleware,
)


# ---------------------------------------------------------------------------
# Helpers — reusable hook factories
# ---------------------------------------------------------------------------


def _pass_hook(name: str):
    """Return a hook function that always passes."""
    return lambda text: HookResult(hook_name=name, action="pass")


def _block_hook(name: str, *, keyword: str = "bad"):
    """Return a hook function that blocks when *keyword* is present."""
    def hook_fn(text: str) -> HookResult:
        if keyword in text:
            return HookResult(hook_name=name, action="block", message=f"contains '{keyword}'")
        return HookResult(hook_name=name, action="pass")
    return hook_fn


def _modify_hook(name: str, *, old: str, new: str):
    """Return a hook function that replaces *old* with *new* in the text."""
    def hook_fn(text: str) -> HookResult:
        if old in text:
            return HookResult(
                hook_name=name, action="modify", modified_text=text.replace(old, new),
            )
        return HookResult(hook_name=name, action="pass")
    return hook_fn


def _warn_hook(name: str, *, keyword: str = "risky"):
    """Return a hook function that warns when *keyword* is present."""
    def hook_fn(text: str) -> HookResult:
        if keyword in text:
            return HookResult(hook_name=name, action="warn", message=f"contains '{keyword}'")
        return HookResult(hook_name=name, action="pass")
    return hook_fn


# ---------------------------------------------------------------------------
# Tests — pass-through / basic lifecycle
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_process_input_no_hooks(self):
        mw = SafetyMiddleware()
        result = mw.process_input("hello world")

        assert result.original_text == "hello world"
        assert result.final_text == "hello world"
        assert result.blocked is False
        assert result.hook_results == []

    def test_process_output_no_hooks(self):
        mw = SafetyMiddleware()
        result = mw.process_output("response text")

        assert result.blocked is False
        assert result.final_text == "response text"

    def test_empty_text(self):
        mw = SafetyMiddleware()
        result = mw.process_input("")
        assert result.final_text == ""
        assert result.blocked is False


# ---------------------------------------------------------------------------
# Tests — pre / post / both hooks
# ---------------------------------------------------------------------------


class TestHookStages:
    def test_pre_hook_runs_on_input(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="pre1", stage="pre"),
            _pass_hook("pre1"),
        )
        result = mw.process_input("hello")
        assert len(result.hook_results) == 1
        assert result.hook_results[0].hook_name == "pre1"

    def test_pre_hook_skipped_on_output(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="pre1", stage="pre"),
            _pass_hook("pre1"),
        )
        result = mw.process_output("hello")
        assert result.hook_results == []

    def test_post_hook_runs_on_output(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="post1", stage="post"),
            _pass_hook("post1"),
        )
        result = mw.process_output("hello")
        assert len(result.hook_results) == 1
        assert result.hook_results[0].hook_name == "post1"

    def test_post_hook_skipped_on_input(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="post1", stage="post"),
            _pass_hook("post1"),
        )
        result = mw.process_input("hello")
        assert result.hook_results == []

    def test_both_hook_runs_on_input_and_output(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="both1", stage="both"),
            _pass_hook("both1"),
        )
        input_result = mw.process_input("hello")
        output_result = mw.process_output("hello")

        assert len(input_result.hook_results) == 1
        assert len(output_result.hook_results) == 1


# ---------------------------------------------------------------------------
# Tests — block action
# ---------------------------------------------------------------------------


class TestBlockAction:
    def test_block_stops_processing(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="blocker", stage="pre", priority=0),
            _block_hook("blocker", keyword="bad"),
        )
        mw.register(
            MiddlewareHook(name="after", stage="pre", priority=1),
            _pass_hook("after"),
        )
        result = mw.process_input("this is bad input")

        assert result.blocked is True
        assert len(result.hook_results) == 1
        assert result.hook_results[0].action == "block"

    def test_block_does_not_trigger_if_keyword_absent(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="blocker", stage="pre"),
            _block_hook("blocker", keyword="bad"),
        )
        result = mw.process_input("this is fine")
        assert result.blocked is False


# ---------------------------------------------------------------------------
# Tests — modify action
# ---------------------------------------------------------------------------


class TestModifyAction:
    def test_modify_updates_text_for_subsequent_hooks(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="replacer", stage="pre", priority=0),
            _modify_hook("replacer", old="foo", new="bar"),
        )

        captured: list[str] = []

        def spy_hook(text: str) -> HookResult:
            captured.append(text)
            return HookResult(hook_name="spy", action="pass")

        mw.register(
            MiddlewareHook(name="spy", stage="pre", priority=1),
            spy_hook,
        )

        result = mw.process_input("foo baz")
        assert result.final_text == "bar baz"
        assert captured == ["bar baz"]

    def test_modify_without_modified_text_is_noop(self):
        mw = SafetyMiddleware()

        def half_modify(text: str) -> HookResult:
            return HookResult(hook_name="half", action="modify", modified_text=None)

        mw.register(MiddlewareHook(name="half", stage="pre"), half_modify)
        result = mw.process_input("unchanged")
        assert result.final_text == "unchanged"

    def test_chained_modifications(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="step1", stage="pre", priority=0),
            _modify_hook("step1", old="a", new="b"),
        )
        mw.register(
            MiddlewareHook(name="step2", stage="pre", priority=1),
            _modify_hook("step2", old="b", new="c"),
        )
        result = mw.process_input("a")
        assert result.final_text == "c"


# ---------------------------------------------------------------------------
# Tests — warn action
# ---------------------------------------------------------------------------


class TestWarnAction:
    def test_warn_does_not_block(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="warner", stage="pre"),
            _warn_hook("warner", keyword="risky"),
        )
        result = mw.process_input("this is risky stuff")
        assert result.blocked is False
        assert result.hook_results[0].action == "warn"


# ---------------------------------------------------------------------------
# Tests — priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_hooks_run_in_priority_order(self):
        mw = SafetyMiddleware()
        order: list[str] = []

        def make_tracker(name: str):
            def fn(text: str) -> HookResult:
                order.append(name)
                return HookResult(hook_name=name, action="pass")
            return fn

        mw.register(MiddlewareHook(name="c", stage="pre", priority=10), make_tracker("c"))
        mw.register(MiddlewareHook(name="a", stage="pre", priority=0), make_tracker("a"))
        mw.register(MiddlewareHook(name="b", stage="pre", priority=5), make_tracker("b"))

        mw.process_input("test")
        assert order == ["a", "b", "c"]

    def test_list_hooks_sorted_by_priority(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="z", stage="pre", priority=99), _pass_hook("z"))
        mw.register(MiddlewareHook(name="a", stage="post", priority=1), _pass_hook("a"))
        mw.register(MiddlewareHook(name="m", stage="both", priority=50), _pass_hook("m"))

        hooks = mw.list_hooks()
        assert [h.name for h in hooks] == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Tests — enable / disable / remove
# ---------------------------------------------------------------------------


class TestHookManagement:
    def test_disable_skips_hook(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="blocker", stage="pre"),
            _block_hook("blocker", keyword="bad"),
        )
        mw.disable("blocker")
        result = mw.process_input("bad input")
        assert result.blocked is False
        assert result.hook_results == []

    def test_enable_after_disable(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="blocker", stage="pre"),
            _block_hook("blocker", keyword="bad"),
        )
        mw.disable("blocker")
        mw.enable("blocker")
        result = mw.process_input("bad input")
        assert result.blocked is True

    def test_remove_hook(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="temp", stage="pre"),
            _pass_hook("temp"),
        )
        mw.remove("temp")
        assert mw.list_hooks() == []

    def test_remove_nonexistent_raises_key_error(self):
        mw = SafetyMiddleware()
        with pytest.raises(KeyError):
            mw.remove("ghost")

    def test_enable_nonexistent_raises_key_error(self):
        mw = SafetyMiddleware()
        with pytest.raises(KeyError):
            mw.enable("ghost")

    def test_disable_nonexistent_raises_key_error(self):
        mw = SafetyMiddleware()
        with pytest.raises(KeyError):
            mw.disable("ghost")


# ---------------------------------------------------------------------------
# Tests — stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_initial(self):
        mw = SafetyMiddleware()
        s = mw.stats()
        assert s.total_processed == 0
        assert s.blocked_count == 0
        assert s.modified_count == 0
        assert s.pass_count == 0
        assert s.hooks_triggered == {}

    def test_stats_after_pass(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="p", stage="pre"), _pass_hook("p"))
        mw.process_input("hello")
        s = mw.stats()
        assert s.total_processed == 1
        assert s.pass_count == 1
        assert s.hooks_triggered == {"p": 1}

    def test_stats_after_block(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="b", stage="pre"),
            _block_hook("b", keyword="bad"),
        )
        mw.process_input("bad")
        s = mw.stats()
        assert s.blocked_count == 1
        assert s.total_processed == 1

    def test_stats_after_modify(self):
        mw = SafetyMiddleware()
        mw.register(
            MiddlewareHook(name="m", stage="pre"),
            _modify_hook("m", old="x", new="y"),
        )
        mw.process_input("x")
        s = mw.stats()
        assert s.modified_count == 1

    def test_hooks_triggered_accumulates(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="h", stage="pre"), _pass_hook("h"))
        mw.process_input("a")
        mw.process_input("b")
        assert mw.stats().hooks_triggered["h"] == 2

    def test_reset_stats(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="h", stage="pre"), _pass_hook("h"))
        mw.process_input("a")
        mw.reset_stats()
        s = mw.stats()
        assert s.total_processed == 0
        assert s.hooks_triggered == {}


# ---------------------------------------------------------------------------
# Tests — elapsed time
# ---------------------------------------------------------------------------


class TestElapsedTime:
    def test_elapsed_ms_is_non_negative(self):
        mw = SafetyMiddleware()
        result = mw.process_input("hello")
        assert result.elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Tests — multiple hooks in chain
# ---------------------------------------------------------------------------


class TestMultipleHooksChain:
    def test_multiple_hooks_all_run(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="a", stage="pre", priority=0), _pass_hook("a"))
        mw.register(MiddlewareHook(name="b", stage="pre", priority=1), _pass_hook("b"))
        mw.register(MiddlewareHook(name="c", stage="pre", priority=2), _pass_hook("c"))

        result = mw.process_input("text")
        assert len(result.hook_results) == 3
        assert [r.hook_name for r in result.hook_results] == ["a", "b", "c"]

    def test_block_in_middle_skips_remaining(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="a", stage="pre", priority=0), _pass_hook("a"))
        mw.register(
            MiddlewareHook(name="b", stage="pre", priority=1),
            _block_hook("b", keyword="stop"),
        )
        mw.register(MiddlewareHook(name="c", stage="pre", priority=2), _pass_hook("c"))

        result = mw.process_input("please stop")
        assert result.blocked is True
        assert [r.hook_name for r in result.hook_results] == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_duplicate_hook_name_overwrites(self):
        mw = SafetyMiddleware()
        mw.register(MiddlewareHook(name="dup", stage="pre"), _pass_hook("dup"))
        mw.register(
            MiddlewareHook(name="dup", stage="pre"),
            _block_hook("dup", keyword="x"),
        )
        result = mw.process_input("x")
        assert result.blocked is True

    def test_result_dataclass_fields(self):
        result = MiddlewareResult(
            original_text="a",
            final_text="b",
            blocked=False,
            hook_results=[],
            elapsed_ms=1.5,
        )
        assert result.original_text == "a"
        assert result.elapsed_ms == 1.5

    def test_stats_dataclass_defaults(self):
        s = MiddlewareStats()
        assert s.total_processed == 0
        assert s.hooks_triggered == {}
