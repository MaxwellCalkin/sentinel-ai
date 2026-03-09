"""Output formatting and normalization for LLM responses.

Standardize LLM outputs into consistent formats: extract
JSON from markdown code blocks, normalize list formats,
enforce structure, and clean up common formatting issues.

Usage:
    from sentinel.output_formatter import OutputFormatter

    fmt = OutputFormatter()
    result = fmt.extract_json('Here is the data: ```json\n{"key": "value"}\n```')
    print(result.data)  # {"key": "value"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormatResult:
    """Result of formatting operation."""
    original: str
    formatted: str
    format_type: str  # "json", "list", "text", "table"
    success: bool
    data: Any = None  # Parsed data if applicable
    issues: list[str] = field(default_factory=list)


class OutputFormatter:
    """Format and normalize LLM outputs.

    Extract structured data from free-text responses,
    normalize formatting, and enforce output conventions.
    """

    def extract_json(self, text: str) -> FormatResult:
        """Extract JSON from LLM output.

        Handles JSON in markdown code blocks, raw JSON,
        and JSON mixed with explanatory text.
        """
        # Try markdown code block first
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if code_block:
            raw = code_block.group(1).strip()
            try:
                data = json.loads(raw)
                return FormatResult(
                    original=text,
                    formatted=json.dumps(data, indent=2),
                    format_type="json",
                    success=True,
                    data=data,
                )
            except json.JSONDecodeError as e:
                return FormatResult(
                    original=text,
                    formatted=text,
                    format_type="json",
                    success=False,
                    issues=[f"Invalid JSON in code block: {e}"],
                )

        # Try finding raw JSON object or array
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, text)
            if match:
                raw = match.group()
                try:
                    data = json.loads(raw)
                    return FormatResult(
                        original=text,
                        formatted=json.dumps(data, indent=2),
                        format_type="json",
                        success=True,
                        data=data,
                    )
                except json.JSONDecodeError:
                    continue

        return FormatResult(
            original=text,
            formatted=text,
            format_type="json",
            success=False,
            issues=["No valid JSON found in output"],
        )

    def extract_list(self, text: str) -> FormatResult:
        """Extract list items from LLM output.

        Handles numbered lists, bullet points, dashes,
        and other common list formats.
        """
        items = []

        # Match numbered lists: "1. item", "1) item"
        numbered = re.findall(r'^\s*\d+[.)]\s+(.+)', text, re.MULTILINE)
        if numbered:
            items = [item.strip() for item in numbered]
        else:
            # Match bullet/dash lists: "- item", "* item", "• item"
            bullets = re.findall(r'^\s*[-*\u2022]\s+(.+)', text, re.MULTILINE)
            if bullets:
                items = [item.strip() for item in bullets]

        if items:
            formatted = "\n".join(f"- {item}" for item in items)
            return FormatResult(
                original=text,
                formatted=formatted,
                format_type="list",
                success=True,
                data=items,
            )

        return FormatResult(
            original=text,
            formatted=text,
            format_type="list",
            success=False,
            issues=["No list structure found"],
        )

    def extract_code(self, text: str, language: str | None = None) -> FormatResult:
        """Extract code from markdown code blocks."""
        if language:
            pattern = rf'```{re.escape(language)}\s*\n(.*?)\n\s*```'
        else:
            pattern = r'```\w*\s*\n(.*?)\n\s*```'

        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return FormatResult(
                original=text,
                formatted=code,
                format_type="code",
                success=True,
                data=code,
            )

        return FormatResult(
            original=text,
            formatted=text,
            format_type="code",
            success=False,
            issues=["No code block found"],
        )

    def normalize_whitespace(self, text: str) -> FormatResult:
        """Normalize whitespace in output."""
        cleaned = re.sub(r'\n{3,}', '\n\n', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = cleaned.strip()
        return FormatResult(
            original=text,
            formatted=cleaned,
            format_type="text",
            success=True,
        )

    def truncate(self, text: str, max_chars: int, suffix: str = "...") -> FormatResult:
        """Truncate output to maximum length."""
        if len(text) <= max_chars:
            return FormatResult(
                original=text,
                formatted=text,
                format_type="text",
                success=True,
            )
        truncated = text[:max_chars - len(suffix)] + suffix
        return FormatResult(
            original=text,
            formatted=truncated,
            format_type="text",
            success=True,
            issues=[f"Truncated from {len(text)} to {max_chars} chars"],
        )
