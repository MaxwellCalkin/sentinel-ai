"""Validate LLM responses against expected schemas, formats, and content constraints.

Ensures LLM responses meet application requirements before delivery
to end users. Supports format checks, content length bounds, keyword
presence/absence, custom validation rules, and confidence scoring.

Usage:
    from sentinel.response_validator import ResponseValidator

    validator = ResponseValidator()
    validator.add_check("is_json", lambda t: _is_json(t), required=True)
    validator.require_keywords(["summary", "conclusion"])
    validator.set_length_bounds(min_chars=50, max_chars=5000)

    result = validator.validate("Some LLM response...")
    print(result.passed)       # True/False
    print(result.confidence)   # 0.0 - 1.0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ResponseCheck:
    """A single validation check to apply to a response."""
    name: str
    check_fn: Callable[[str], bool]
    required: bool = True
    description: str = ""


@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    passed: bool
    message: str
    required: bool = True


@dataclass
class ResponseValidation:
    """Full validation result for a single response."""
    text: str
    passed: bool
    confidence: float
    results: list[CheckResult] = field(default_factory=list)
    failed_required: int = 0
    failed_optional: int = 0


@dataclass
class ResponseValidatorStats:
    """Aggregate statistics from the validator."""
    total_validated: int
    pass_rate: float
    avg_confidence: float


class ResponseValidator:
    """Validate LLM responses against configurable checks.

    Register format checks, keyword requirements, length bounds,
    and custom rules. Each validation produces a confidence score
    reflecting how well the response matches expectations.
    """

    def __init__(self) -> None:
        self._checks: list[ResponseCheck] = []
        self._total_validated: int = 0
        self._total_passed: int = 0
        self._total_confidence: float = 0.0

    def add_check(
        self,
        name: str,
        check_fn: Callable[[str], bool],
        required: bool = True,
        description: str = "",
    ) -> ResponseValidator:
        """Register a custom validation check."""
        self._checks.append(ResponseCheck(
            name=name,
            check_fn=check_fn,
            required=required,
            description=description,
        ))
        return self

    def require_json(self) -> ResponseValidator:
        """Require the response to be valid JSON or contain a JSON block."""
        return self.add_check(
            name="json_parseable",
            check_fn=_is_json_parseable,
            required=True,
            description="Response must contain valid JSON",
        )

    def require_list_format(self) -> ResponseValidator:
        """Require the response to contain a numbered or bulleted list."""
        return self.add_check(
            name="list_format",
            check_fn=_contains_list,
            required=True,
            description="Response must contain a list",
        )

    def require_code_block(self) -> ResponseValidator:
        """Require the response to contain a fenced code block."""
        return self.add_check(
            name="code_block",
            check_fn=_contains_code_block,
            required=True,
            description="Response must contain a code block",
        )

    def set_length_bounds(
        self,
        min_chars: int | None = None,
        max_chars: int | None = None,
    ) -> ResponseValidator:
        """Enforce character length bounds on the response."""
        if min_chars is not None:
            lower = min_chars
            self._checks.append(ResponseCheck(
                name="min_length",
                check_fn=lambda text, bound=lower: len(text) >= bound,
                required=True,
                description=f"Response must be at least {lower} characters",
            ))
        if max_chars is not None:
            upper = max_chars
            self._checks.append(ResponseCheck(
                name="max_length",
                check_fn=lambda text, bound=upper: len(text) <= bound,
                required=True,
                description=f"Response must be at most {upper} characters",
            ))
        return self

    def require_keywords(
        self,
        keywords: list[str],
        case_sensitive: bool = False,
    ) -> ResponseValidator:
        """Require all specified keywords to appear in the response."""
        for keyword in keywords:
            word = keyword
            self._checks.append(ResponseCheck(
                name=f"keyword_present:{word}",
                check_fn=_make_keyword_present_check(word, case_sensitive),
                required=True,
                description=f"Response must contain '{word}'",
            ))
        return self

    def forbid_keywords(
        self,
        keywords: list[str],
        case_sensitive: bool = False,
    ) -> ResponseValidator:
        """Forbid specified keywords from appearing in the response."""
        for keyword in keywords:
            word = keyword
            self._checks.append(ResponseCheck(
                name=f"keyword_absent:{word}",
                check_fn=_make_keyword_absent_check(word, case_sensitive),
                required=True,
                description=f"Response must not contain '{word}'",
            ))
        return self

    @property
    def check_count(self) -> int:
        return len(self._checks)

    def validate(self, text: str) -> ResponseValidation:
        """Validate a single response against all registered checks."""
        results = _run_checks(self._checks, text)
        failed_required = _count_failures(results, required_only=True)
        failed_optional = _count_failures(results, required_only=False)
        confidence = _compute_confidence(self._checks, results)
        passed = failed_required == 0

        self._record_stats(passed, confidence)

        return ResponseValidation(
            text=text,
            passed=passed,
            confidence=confidence,
            results=results,
            failed_required=failed_required,
            failed_optional=failed_optional,
        )

    def validate_batch(self, texts: list[str]) -> list[ResponseValidation]:
        """Validate multiple responses."""
        return [self.validate(text) for text in texts]

    def stats(self) -> ResponseValidatorStats:
        """Return aggregate validation statistics."""
        if self._total_validated == 0:
            return ResponseValidatorStats(
                total_validated=0,
                pass_rate=0.0,
                avg_confidence=0.0,
            )
        return ResponseValidatorStats(
            total_validated=self._total_validated,
            pass_rate=self._total_passed / self._total_validated,
            avg_confidence=self._total_confidence / self._total_validated,
        )

    def _record_stats(self, passed: bool, confidence: float) -> None:
        self._total_validated += 1
        if passed:
            self._total_passed += 1
        self._total_confidence += confidence


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run_checks(checks: list[ResponseCheck], text: str) -> list[CheckResult]:
    """Execute all checks against the text and collect results."""
    results: list[CheckResult] = []
    for check in checks:
        passed = _execute_check(check, text)
        message = "passed" if passed else f"failed: {check.description or check.name}"
        results.append(CheckResult(
            name=check.name,
            passed=passed,
            message=message,
            required=check.required,
        ))
    return results


def _execute_check(check: ResponseCheck, text: str) -> bool:
    """Run a single check, catching exceptions as failures."""
    try:
        return bool(check.check_fn(text))
    except Exception:
        return False


def _count_failures(results: list[CheckResult], required_only: bool) -> int:
    """Count failed checks, optionally filtering by required/optional."""
    if required_only:
        return sum(1 for r in results if not r.passed and r.required)
    return sum(1 for r in results if not r.passed and not r.required)


def _compute_confidence(
    checks: list[ResponseCheck],
    results: list[CheckResult],
) -> float:
    """Compute a 0.0-1.0 confidence score weighted by check importance."""
    if not checks:
        return 1.0

    total_weight = 0.0
    earned_weight = 0.0

    for check, result in zip(checks, results):
        weight = 2.0 if check.required else 1.0
        total_weight += weight
        if result.passed:
            earned_weight += weight

    return round(earned_weight / total_weight, 4) if total_weight > 0 else 1.0


def _is_json_parseable(text: str) -> bool:
    """Check if text is valid JSON or contains a JSON code block."""
    stripped = text.strip()
    try:
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1).strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass

    return False


def _contains_list(text: str) -> bool:
    """Check if text contains a numbered or bulleted list."""
    return bool(re.search(r"(?m)^(?:\s*[-*+]\s|^\s*\d+[.)]\s)", text))


def _contains_code_block(text: str) -> bool:
    """Check if text contains a fenced code block."""
    return bool(re.search(r"```[\s\S]*?```", text))


def _make_keyword_present_check(
    keyword: str,
    case_sensitive: bool,
) -> Callable[[str], bool]:
    """Create a check function that verifies a keyword is present."""
    def check(text: str) -> bool:
        if case_sensitive:
            return keyword in text
        return keyword.lower() in text.lower()
    return check


def _make_keyword_absent_check(
    keyword: str,
    case_sensitive: bool,
) -> Callable[[str], bool]:
    """Create a check function that verifies a keyword is absent."""
    def check(text: str) -> bool:
        if case_sensitive:
            return keyword not in text
        return keyword.lower() not in text.lower()
    return check
