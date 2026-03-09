"""Tests for output formatter."""

import pytest
from sentinel.output_formatter import OutputFormatter, FormatResult


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

class TestJsonExtraction:
    def test_json_code_block(self):
        f = OutputFormatter()
        result = f.extract_json('Here is the data:\n```json\n{"key": "value"}\n```')
        assert result.success
        assert result.data == {"key": "value"}

    def test_json_plain_code_block(self):
        f = OutputFormatter()
        result = f.extract_json('```\n{"a": 1}\n```')
        assert result.success
        assert result.data == {"a": 1}

    def test_raw_json_object(self):
        f = OutputFormatter()
        result = f.extract_json('The result is {"name": "test", "count": 5}')
        assert result.success
        assert result.data["name"] == "test"

    def test_raw_json_array(self):
        f = OutputFormatter()
        result = f.extract_json('Items: [1, 2, 3]')
        assert result.success
        assert result.data == [1, 2, 3]

    def test_no_json(self):
        f = OutputFormatter()
        result = f.extract_json("Just plain text with no JSON")
        assert not result.success

    def test_invalid_json_in_block(self):
        f = OutputFormatter()
        result = f.extract_json('```json\n{invalid json}\n```')
        assert not result.success
        assert len(result.issues) > 0


# ---------------------------------------------------------------------------
# List extraction
# ---------------------------------------------------------------------------

class TestListExtraction:
    def test_numbered_list(self):
        f = OutputFormatter()
        result = f.extract_list("1. First item\n2. Second item\n3. Third item")
        assert result.success
        assert len(result.data) == 3
        assert result.data[0] == "First item"

    def test_bullet_list(self):
        f = OutputFormatter()
        result = f.extract_list("- Apple\n- Banana\n- Cherry")
        assert result.success
        assert len(result.data) == 3

    def test_asterisk_list(self):
        f = OutputFormatter()
        result = f.extract_list("* Item A\n* Item B")
        assert result.success
        assert len(result.data) == 2

    def test_no_list(self):
        f = OutputFormatter()
        result = f.extract_list("Just a paragraph of text.")
        assert not result.success


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

class TestCodeExtraction:
    def test_python_code_block(self):
        f = OutputFormatter()
        result = f.extract_code('```python\nprint("hello")\n```')
        assert result.success
        assert 'print("hello")' in result.formatted

    def test_any_code_block(self):
        f = OutputFormatter()
        result = f.extract_code('```js\nconsole.log("hi")\n```')
        assert result.success

    def test_specific_language(self):
        f = OutputFormatter()
        text = '```python\nx = 1\n```\n```javascript\ny = 2\n```'
        result = f.extract_code(text, language="python")
        assert result.success
        assert "x = 1" in result.formatted

    def test_no_code_block(self):
        f = OutputFormatter()
        result = f.extract_code("No code here")
        assert not result.success


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_normalize_multiple_newlines(self):
        f = OutputFormatter()
        result = f.normalize_whitespace("Hello\n\n\n\nWorld")
        assert result.formatted == "Hello\n\nWorld"

    def test_normalize_spaces(self):
        f = OutputFormatter()
        result = f.normalize_whitespace("Too   many    spaces")
        assert result.formatted == "Too many spaces"

    def test_strip_edges(self):
        f = OutputFormatter()
        result = f.normalize_whitespace("  \n  text  \n  ")
        assert result.formatted == "text"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_truncate_long_text(self):
        f = OutputFormatter()
        result = f.truncate("A" * 100, max_chars=50)
        assert len(result.formatted) == 50
        assert result.formatted.endswith("...")

    def test_no_truncation_needed(self):
        f = OutputFormatter()
        result = f.truncate("short", max_chars=100)
        assert result.formatted == "short"
        assert len(result.issues) == 0

    def test_custom_suffix(self):
        f = OutputFormatter()
        result = f.truncate("A" * 100, max_chars=50, suffix="[...]")
        assert result.formatted.endswith("[...]")


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_result_structure(self):
        f = OutputFormatter()
        result = f.extract_json('{"a": 1}')
        assert isinstance(result, FormatResult)
        assert result.format_type == "json"
        assert isinstance(result.issues, list)
