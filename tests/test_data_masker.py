"""Tests for data masker."""

import pytest
from sentinel.data_masker import DataMasker, MaskResult


# ---------------------------------------------------------------------------
# Email masking
# ---------------------------------------------------------------------------

class TestEmailMasking:
    def test_mask_email(self):
        m = DataMasker()
        result = m.mask("Contact user@example.com for info")
        assert "user@example.com" not in result.masked
        assert "[EMAIL_1]" in result.masked
        assert result.items_masked == 1

    def test_multiple_emails(self):
        m = DataMasker()
        result = m.mask("Send to a@b.com and c@d.com")
        assert result.items_masked == 2
        assert "[EMAIL_1]" in result.masked
        assert "[EMAIL_2]" in result.masked

    def test_email_disabled(self):
        m = DataMasker(mask_emails=False)
        result = m.mask("Contact user@example.com")
        assert "user@example.com" in result.masked


# ---------------------------------------------------------------------------
# Phone masking
# ---------------------------------------------------------------------------

class TestPhoneMasking:
    def test_mask_phone(self):
        m = DataMasker()
        result = m.mask("Call 555-123-4567 now")
        assert "555-123-4567" not in result.masked
        assert "[PHONE_1]" in result.masked

    def test_phone_dots(self):
        m = DataMasker()
        result = m.mask("Call 555.123.4567")
        assert "555.123.4567" not in result.masked


# ---------------------------------------------------------------------------
# SSN masking
# ---------------------------------------------------------------------------

class TestSSNMasking:
    def test_mask_ssn(self):
        m = DataMasker()
        result = m.mask("SSN: 123-45-6789")
        assert "123-45-6789" not in result.masked
        assert "[SSN_1]" in result.masked


# ---------------------------------------------------------------------------
# IP masking
# ---------------------------------------------------------------------------

class TestIPMasking:
    def test_mask_ip(self):
        m = DataMasker()
        result = m.mask("Server at 192.168.1.100")
        assert "192.168.1.100" not in result.masked
        assert "[IP_1]" in result.masked


# ---------------------------------------------------------------------------
# Name masking
# ---------------------------------------------------------------------------

class TestNameMasking:
    def test_mask_names(self):
        m = DataMasker(mask_names=True)
        result = m.mask("Talk to John about the project")
        assert "John" not in result.masked
        assert "[NAME_1]" in result.masked

    def test_names_disabled_by_default(self):
        m = DataMasker()
        result = m.mask("Talk to John about it")
        assert "John" in result.masked


# ---------------------------------------------------------------------------
# Unmasking
# ---------------------------------------------------------------------------

class TestUnmasking:
    def test_roundtrip(self):
        m = DataMasker()
        original = "Email user@test.com or call 555-123-4567"
        result = m.mask(original)
        restored = m.unmask(result.masked, result.mapping)
        assert restored == original

    def test_unmask_preserves_context(self):
        m = DataMasker()
        result = m.mask("Contact admin@corp.com for help")
        # Simulate LLM processing
        processed = result.masked.replace("Contact", "Please reach out to")
        restored = m.unmask(processed, result.mapping)
        assert "admin@corp.com" in restored
        assert "Please reach out to" in restored


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustomPatterns:
    def test_custom_pattern(self):
        m = DataMasker(custom_patterns=[
            (r'ACCT-\d{6}', "ACCOUNT"),
        ])
        result = m.mask("Account ACCT-123456 balance")
        assert "ACCT-123456" not in result.masked
        assert "[ACCOUNT_1]" in result.masked


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_mask(self):
        m = DataMasker()
        results = m.mask_batch([
            "Email user@test.com",
            "Call 555-123-4567",
            "Normal text",
        ])
        assert len(results) == 3
        assert results[0].items_masked == 1
        assert results[1].items_masked == 1
        assert results[2].items_masked == 0


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_result_structure(self):
        m = DataMasker()
        result = m.mask("Email user@test.com")
        assert isinstance(result, MaskResult)
        assert result.original == "Email user@test.com"
        assert result.items_masked == 1
        assert "EMAIL" in result.categories
        assert isinstance(result.mapping, dict)

    def test_no_sensitive_data(self):
        m = DataMasker()
        result = m.mask("Just a normal sentence")
        assert result.items_masked == 0
        assert result.masked == result.original
        assert len(result.categories) == 0

    def test_empty_text(self):
        m = DataMasker()
        result = m.mask("")
        assert result.items_masked == 0
