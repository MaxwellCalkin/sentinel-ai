"""Tests for prompt chain pipeline."""

import pytest
from sentinel.prompt_chain import PromptChain, ChainResult, StepResult


# ---------------------------------------------------------------------------
# Basic pipeline
# ---------------------------------------------------------------------------

class TestBasicPipeline:
    def test_single_step(self):
        chain = PromptChain()
        chain.add_step("upper", lambda t: t.upper())
        result = chain.run("hello")
        assert result.output == "HELLO"
        assert result.success
        assert result.step_count == 1

    def test_multi_step(self):
        chain = (PromptChain()
                 .add_step("upper", lambda t: t.upper())
                 .add_step("exclaim", lambda t: t + "!"))
        result = chain.run("hello")
        assert result.output == "HELLO!"
        assert result.step_count == 2

    def test_empty_chain(self):
        chain = PromptChain()
        result = chain.run("hello")
        assert result.output == "hello"
        assert result.success
        assert result.step_count == 0


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpoints:
    def test_passing_checkpoint(self):
        chain = (PromptChain()
                 .add_step("upper", lambda t: t.upper())
                 .add_checkpoint("safety", lambda t: t if len(t) < 100 else None)
                 .add_step("exclaim", lambda t: t + "!"))
        result = chain.run("hello")
        assert result.success
        assert result.output == "HELLO!"

    def test_failing_checkpoint(self):
        chain = (PromptChain()
                 .add_step("upper", lambda t: t.upper())
                 .add_checkpoint("length_check", lambda t: t if len(t) < 3 else None,
                                 message="Too long")
                 .add_step("exclaim", lambda t: t + "!"))
        result = chain.run("hello")
        assert not result.success
        assert result.stopped_at == "length_check"
        assert result.output == "HELLO"  # Stopped before exclaim

    def test_checkpoint_can_modify(self):
        chain = (PromptChain()
                 .add_checkpoint("clean", lambda t: t.strip()))
        result = chain.run("  hello  ")
        assert result.output == "hello"
        assert result.success


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_filter_pass(self):
        chain = (PromptChain()
                 .add_filter("no_bad", lambda t: "bad" not in t)
                 .add_step("process", lambda t: t.upper()))
        result = chain.run("good input")
        assert result.success
        assert result.output == "GOOD INPUT"

    def test_filter_block(self):
        chain = (PromptChain()
                 .add_filter("no_bad", lambda t: "bad" not in t, message="Bad content")
                 .add_step("process", lambda t: t.upper()))
        result = chain.run("bad input")
        assert not result.success
        assert result.stopped_at == "no_bad"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_step_error_stops(self):
        chain = (PromptChain()
                 .add_step("boom", lambda t: (_ for _ in ()).throw(RuntimeError("crash")))
                 .add_step("after", lambda t: t.upper()))
        result = chain.run("hello")
        assert not result.success
        assert result.stopped_at == "boom"

    def test_step_error_skip(self):
        chain = (PromptChain()
                 .add_step("boom", lambda t: (_ for _ in ()).throw(RuntimeError("crash")),
                           on_error="skip")
                 .add_step("upper", lambda t: t.upper()))
        result = chain.run("hello")
        assert result.success
        assert result.output == "HELLO"

    def test_checkpoint_error_stops(self):
        chain = (PromptChain()
                 .add_checkpoint("bad", lambda t: (_ for _ in ()).throw(RuntimeError("crash"))))
        result = chain.run("hello")
        assert not result.success


# ---------------------------------------------------------------------------
# Step results
# ---------------------------------------------------------------------------

class TestStepResults:
    def test_step_details(self):
        chain = (PromptChain()
                 .add_step("upper", lambda t: t.upper())
                 .add_checkpoint("check", lambda t: t))
        result = chain.run("hello")
        assert len(result.steps) == 2
        assert result.steps[0].name == "upper"
        assert result.steps[0].step_type == "transform"
        assert result.steps[0].input_text == "hello"
        assert result.steps[0].output_text == "HELLO"
        assert result.steps[0].passed
        assert result.steps[0].duration_ms >= 0

    def test_failed_step_has_error(self):
        chain = PromptChain().add_checkpoint("fail", lambda t: None, message="Blocked!")
        result = chain.run("test")
        assert result.steps[0].error == "Blocked!"

    def test_duration_tracked(self):
        chain = PromptChain().add_step("slow", lambda t: t)
        result = chain.run("hello")
        assert result.total_duration_ms >= 0


# ---------------------------------------------------------------------------
# Chaining API
# ---------------------------------------------------------------------------

class TestChaining:
    def test_fluent_api(self):
        chain = (PromptChain()
                 .add_step("a", lambda t: t + "A")
                 .add_step("b", lambda t: t + "B")
                 .add_step("c", lambda t: t + "C"))
        assert chain.step_count == 3
        result = chain.run("")
        assert result.output == "ABC"

    def test_checkpoint_count(self):
        chain = (PromptChain()
                 .add_step("a", lambda t: t)
                 .add_checkpoint("c1", lambda t: t)
                 .add_filter("f1", lambda t: True)
                 .add_step("b", lambda t: t))
        assert chain.step_count == 4
        assert chain.checkpoint_count == 2  # checkpoint + filter


# ---------------------------------------------------------------------------
# Real-world scenarios
# ---------------------------------------------------------------------------

class TestRealWorld:
    def test_sanitize_pipeline(self):
        chain = (PromptChain()
                 .add_step("trim", lambda t: t.strip())
                 .add_filter("not_empty", lambda t: len(t) > 0, message="Empty input")
                 .add_step("lower", lambda t: t.lower())
                 .add_filter("max_len", lambda t: len(t) <= 1000, message="Too long")
                 .add_step("clean", lambda t: t.replace("  ", " ")))
        result = chain.run("  Hello  World  ")
        assert result.success
        assert result.output == "hello world"

    def test_content_moderation_chain(self):
        blocked_words = ["hack", "exploit"]

        chain = (PromptChain()
                 .add_filter("block_words",
                             lambda t: not any(w in t.lower() for w in blocked_words),
                             message="Blocked content detected")
                 .add_step("process", lambda t: f"Processed: {t}"))

        good = chain.run("What is the weather?")
        assert good.success

        bad = chain.run("How to hack a system")
        assert not bad.success
