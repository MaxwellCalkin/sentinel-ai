"""Tests for the self-healing structured output generator."""

import json
import pytest
from sentinel.self_heal import StructuredGenerator, GenerationResult, AttemptRecord


PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
    },
    "required": ["name", "age"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm(*responses):
    """Create a mock LLM that returns responses in order."""
    calls = []
    idx = [0]

    def call_llm(prompt):
        calls.append(prompt)
        i = idx[0]
        idx[0] += 1
        if i < len(responses):
            return responses[i]
        return responses[-1]

    call_llm.calls = calls
    return call_llm


# ---------------------------------------------------------------------------
# First-try success
# ---------------------------------------------------------------------------

class TestFirstTrySuccess:
    def test_valid_on_first_try(self):
        llm = make_llm('{"name": "Alice", "age": 30}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person info")
        assert result.valid
        assert result.parsed == {"name": "Alice", "age": 30}
        assert result.attempts == 1
        assert not result.healed

    def test_valid_with_markdown(self):
        llm = make_llm('```json\n{"name": "Bob", "age": 25}\n```')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.parsed["name"] == "Bob"

    def test_valid_embedded_in_text(self):
        llm = make_llm('Here is the result: {"name": "Carol", "age": 40}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid


# ---------------------------------------------------------------------------
# Self-healing (retry success)
# ---------------------------------------------------------------------------

class TestSelfHealing:
    def test_heals_missing_field(self):
        llm = make_llm(
            '{"name": "Alice"}',  # missing age
            '{"name": "Alice", "age": 30}',  # fixed
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.healed
        assert result.attempts == 2
        assert result.parsed["age"] == 30

    def test_heals_wrong_type(self):
        llm = make_llm(
            '{"name": "Alice", "age": "thirty"}',  # wrong type
            '{"name": "Alice", "age": 30}',  # fixed
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.healed
        assert result.attempts == 2

    def test_heals_on_third_attempt(self):
        llm = make_llm(
            '{"name": "Alice"}',  # missing age
            '{"name": "Alice", "age": "oops"}',  # wrong type
            '{"name": "Alice", "age": 30}',  # finally correct
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.healed
        assert result.attempts == 3

    def test_heals_invalid_json(self):
        llm = make_llm(
            'This is not JSON',
            '{"name": "Alice", "age": 30}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.healed
        assert result.attempts == 2

    def test_corrective_prompt_contains_errors(self):
        llm = make_llm(
            '{"name": "Alice"}',
            '{"name": "Alice", "age": 30}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        gen.generate("Extract person")
        # Second call should have error details in prompt
        assert "validation errors" in llm.calls[1]
        assert "age" in llm.calls[1]
        assert "Required field missing" in llm.calls[1]


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------

class TestExhaustedRetries:
    def test_fails_after_max_retries(self):
        llm = make_llm(
            '{"name": "Alice"}',
            '{"name": "Alice"}',
            '{"name": "Alice"}',
            '{"name": "Alice"}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm, max_retries=3)
        result = gen.generate("Extract person")
        assert not result.valid
        assert result.attempts == 4  # 1 initial + 3 retries
        assert not result.healed

    def test_custom_max_retries(self):
        llm = make_llm(
            '{"name": "Alice"}',
            '{"name": "Alice", "age": 30}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm, max_retries=1)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.attempts == 2

    def test_zero_retries(self):
        llm = make_llm('{"name": "Alice"}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm, max_retries=0)
        result = gen.generate("Extract person")
        assert not result.valid
        assert result.attempts == 1


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_records_all_attempts(self):
        llm = make_llm(
            '{"name": "Alice"}',
            '{"name": "Alice", "age": "bad"}',
            '{"name": "Alice", "age": 30}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert len(result.history) == 3
        assert result.history[0].attempt == 1
        assert result.history[1].attempt == 2
        assert result.history[2].attempt == 3

    def test_history_has_corrective_prompts(self):
        llm = make_llm(
            '{"name": "Alice"}',
            '{"name": "Alice", "age": 30}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        # Failed attempt gets corrective prompt for retry
        assert result.history[0].corrective_prompt is not None
        assert "age" in result.history[0].corrective_prompt
        # Successful final attempt has no corrective prompt
        assert result.history[1].corrective_prompt is None

    def test_history_has_raw_output(self):
        llm = make_llm('{"name": "Alice", "age": 30}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.history[0].raw_output == '{"name": "Alice", "age": 30}'


# ---------------------------------------------------------------------------
# Initial prompt
# ---------------------------------------------------------------------------

class TestInitialPrompt:
    def test_initial_prompt_includes_schema(self):
        llm = make_llm('{"name": "Alice", "age": 30}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        gen.generate("Extract person info")
        prompt = llm.calls[0]
        assert "Extract person info" in prompt
        assert '"type": "object"' in prompt
        assert "JSON" in prompt

    def test_initial_prompt_includes_required(self):
        llm = make_llm('{"name": "Alice", "age": 30}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm)
        gen.generate("Extract person")
        assert '"required"' in llm.calls[0]


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_rejects_extra_fields(self):
        llm = make_llm(
            '{"name": "Alice", "age": 30, "hobby": "chess"}',
            '{"name": "Alice", "age": 30}',
        )
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm, strict=True)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.healed
        assert result.attempts == 2

    def test_non_strict_allows_extra(self):
        llm = make_llm('{"name": "Alice", "age": 30, "hobby": "chess"}')
        gen = StructuredGenerator(PERSON_SCHEMA, call_llm=llm, strict=False)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.attempts == 1


# ---------------------------------------------------------------------------
# Complex schemas
# ---------------------------------------------------------------------------

class TestComplexSchemas:
    def test_nested_object_healing(self):
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
            "required": ["person"],
        }
        llm = make_llm(
            '{"person": {"name": "Alice"}}',
            '{"person": {"name": "Alice", "age": 30}}',
        )
        gen = StructuredGenerator(schema, call_llm=llm)
        result = gen.generate("Extract person")
        assert result.valid
        assert result.healed

    def test_array_schema(self):
        schema = {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 2,
        }
        llm = make_llm(
            '[1]',  # too few items
            '[1, 2, 3]',
        )
        gen = StructuredGenerator(schema, call_llm=llm)
        result = gen.generate("Give me numbers")
        assert result.valid
        assert result.parsed == [1, 2, 3]
