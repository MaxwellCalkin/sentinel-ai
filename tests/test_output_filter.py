"""Tests for configurable output filtering."""

import pytest
from sentinel.output_filter import (
    OutputFilter,
    FilterRule,
    FilterMatch,
    FilterOutput,
    FilterStats,
)


# ---------------------------------------------------------------------------
# Clean text passthrough
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_text_passes_unchanged(self):
        f = OutputFilter()
        result = f.filter("The weather is nice today.")
        assert result.filtered == "The weather is nice today."
        assert result.is_modified is False
        assert result.matches == []
        assert result.rules_applied == 0

    def test_empty_string(self):
        f = OutputFilter()
        result = f.filter("")
        assert result.filtered == ""
        assert result.is_modified is False


# ---------------------------------------------------------------------------
# Email redaction
# ---------------------------------------------------------------------------

class TestEmailRedaction:
    def test_redacts_email(self):
        f = OutputFilter()
        result = f.filter("Contact me at user@example.com please")
        assert "user@example.com" not in result.filtered
        assert "[REDACTED]" in result.filtered
        assert result.is_modified is True

    def test_email_match_details(self):
        f = OutputFilter()
        result = f.filter("Email: admin@test.org")
        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.rule_name == "email"
        assert match.original == "admin@test.org"
        assert match.action == "redact"


# ---------------------------------------------------------------------------
# Phone number redaction
# ---------------------------------------------------------------------------

class TestPhoneRedaction:
    def test_redacts_phone_dashes(self):
        f = OutputFilter()
        result = f.filter("Call 555-123-4567 for info")
        assert "555-123-4567" not in result.filtered
        assert "[REDACTED]" in result.filtered

    def test_redacts_phone_parens(self):
        f = OutputFilter()
        result = f.filter("Call (555) 123-4567 today")
        assert "(555) 123-4567" not in result.filtered


# ---------------------------------------------------------------------------
# URL redaction
# ---------------------------------------------------------------------------

class TestUrlRedaction:
    def test_redacts_http_url(self):
        f = OutputFilter()
        result = f.filter("Visit http://example.com for details")
        assert "http://example.com" not in result.filtered
        assert "[REDACTED]" in result.filtered

    def test_redacts_https_url(self):
        f = OutputFilter()
        result = f.filter("See https://docs.example.com/api/v2")
        assert "https://docs.example.com/api/v2" not in result.filtered


# ---------------------------------------------------------------------------
# Credit card redaction
# ---------------------------------------------------------------------------

class TestCreditCardRedaction:
    def test_redacts_card_with_spaces(self):
        f = OutputFilter()
        result = f.filter("Card: 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in result.filtered
        assert "[REDACTED]" in result.filtered

    def test_redacts_card_with_dashes(self):
        f = OutputFilter()
        result = f.filter("CC 4111-1111-1111-1111 on file")
        assert "4111-1111-1111-1111" not in result.filtered


# ---------------------------------------------------------------------------
# SSN redaction
# ---------------------------------------------------------------------------

class TestSsnRedaction:
    def test_redacts_ssn(self):
        f = OutputFilter()
        result = f.filter("SSN is 123-45-6789")
        assert "123-45-6789" not in result.filtered
        assert "[REDACTED]" in result.filtered

    def test_ssn_match_position(self):
        f = OutputFilter()
        result = f.filter("SSN is 123-45-6789")
        ssn_matches = [m for m in result.matches if m.rule_name == "ssn"]
        assert len(ssn_matches) >= 1
        match = ssn_matches[0]
        assert match.original == "123-45-6789"
        start, end = match.position
        assert "SSN is 123-45-6789"[start:end] == "123-45-6789"


# ---------------------------------------------------------------------------
# Custom rule addition
# ---------------------------------------------------------------------------

class TestCustomRules:
    def test_add_custom_rule(self):
        f = OutputFilter()
        rule = FilterRule(
            name="project_code",
            pattern=r"PROJ-\d+",
            action="redact",
            replacement="[PROJECT]",
        )
        f.add_rule(rule)
        result = f.filter("See PROJ-42 for details")
        assert "PROJ-42" not in result.filtered
        assert "[PROJECT]" in result.filtered

    def test_custom_rule_appears_in_list(self):
        f = OutputFilter()
        rule = FilterRule(name="custom", pattern=r"secret", action="redact")
        f.add_rule(rule)
        names = [r.name for r in f.list_rules()]
        assert "custom" in names


# ---------------------------------------------------------------------------
# Remove action
# ---------------------------------------------------------------------------

class TestRemoveAction:
    def test_remove_deletes_match(self):
        f = OutputFilter()
        rule = FilterRule(
            name="remove_tags",
            pattern=r"<[^>]+>",
            action="remove",
        )
        f.add_rule(rule)
        result = f.filter("Hello <b>world</b>!")
        assert result.filtered == "Hello world!"
        assert result.is_modified is True

    def test_remove_match_has_empty_replacement(self):
        f = OutputFilter()
        rule = FilterRule(name="rm", pattern=r"\bfoo\b", action="remove")
        f.add_rule(rule)
        result = f.filter("bar foo baz")
        match = [m for m in result.matches if m.rule_name == "rm"][0]
        assert match.replacement == ""


# ---------------------------------------------------------------------------
# Replace action
# ---------------------------------------------------------------------------

class TestReplaceAction:
    def test_replace_swaps_text(self):
        f = OutputFilter()
        rule = FilterRule(
            name="replace_curse",
            pattern=r"\bdamn\b",
            action="replace",
            replacement="darn",
        )
        f.add_rule(rule)
        result = f.filter("Well damn, that worked")
        assert result.filtered == "Well darn, that worked"
        assert result.is_modified is True


# ---------------------------------------------------------------------------
# Flag action (no modification)
# ---------------------------------------------------------------------------

class TestFlagAction:
    def test_flag_does_not_modify_text(self):
        f = OutputFilter()
        rule = FilterRule(
            name="flag_keyword",
            pattern=r"\bconfidential\b",
            action="flag",
        )
        f.add_rule(rule)
        result = f.filter("This is confidential info")
        assert result.filtered == "This is confidential info"
        assert result.is_modified is False

    def test_flag_still_records_match(self):
        f = OutputFilter()
        rule = FilterRule(
            name="flag_keyword",
            pattern=r"\bconfidential\b",
            action="flag",
        )
        f.add_rule(rule)
        result = f.filter("This is confidential info")
        assert len(result.matches) >= 1
        flag_matches = [m for m in result.matches if m.rule_name == "flag_keyword"]
        assert len(flag_matches) == 1
        assert flag_matches[0].action == "flag"
        assert result.rules_applied >= 1


# ---------------------------------------------------------------------------
# Rule priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_list_rules_sorted_by_priority(self):
        f = OutputFilter()
        f.add_rule(FilterRule(name="low", pattern=r"a", action="redact", priority=1))
        f.add_rule(FilterRule(name="high", pattern=r"b", action="redact", priority=100))
        f.add_rule(FilterRule(name="mid", pattern=r"c", action="redact", priority=50))
        rules = f.list_rules()
        priorities = [r.priority for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_higher_priority_rule_first_in_list(self):
        f = OutputFilter()
        f.add_rule(FilterRule(name="low", pattern=r"x", action="redact", priority=1))
        f.add_rule(FilterRule(name="high", pattern=r"x", action="redact", priority=99))
        assert f.list_rules()[0].name == "high"


# ---------------------------------------------------------------------------
# Enable / disable rules
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable_rule_skips_it(self):
        f = OutputFilter()
        f.disable("email")
        result = f.filter("Contact user@example.com")
        email_matches = [m for m in result.matches if m.rule_name == "email"]
        assert len(email_matches) == 0

    def test_re_enable_rule_activates_it(self):
        f = OutputFilter()
        f.disable("email")
        f.enable("email")
        result = f.filter("Contact user@example.com")
        email_matches = [m for m in result.matches if m.rule_name == "email"]
        assert len(email_matches) == 1

    def test_disable_nonexistent_raises_key_error(self):
        f = OutputFilter()
        with pytest.raises(KeyError):
            f.disable("nonexistent_rule")

    def test_enable_nonexistent_raises_key_error(self):
        f = OutputFilter()
        with pytest.raises(KeyError):
            f.enable("nonexistent_rule")


# ---------------------------------------------------------------------------
# Batch filtering
# ---------------------------------------------------------------------------

class TestBatchFiltering:
    def test_batch_returns_correct_count(self):
        f = OutputFilter()
        texts = ["Clean text", "Email: a@b.com", "Another clean one"]
        results = f.filter_batch(texts)
        assert len(results) == 3

    def test_batch_filters_each_text(self):
        f = OutputFilter()
        results = f.filter_batch(["a@b.com", "555-111-2222"])
        assert results[0].is_modified is True
        assert results[1].is_modified is True

    def test_batch_empty_list(self):
        f = OutputFilter()
        assert f.filter_batch([]) == []


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStatsTracking:
    def test_stats_after_single_filter(self):
        f = OutputFilter()
        f.filter("Email: a@b.com")
        stats = f.stats()
        assert stats.total_filtered == 1
        assert stats.total_matches >= 1
        assert stats.by_rule.get("email", 0) >= 1
        assert stats.by_action.get("redact", 0) >= 1

    def test_stats_accumulate_across_calls(self):
        f = OutputFilter()
        f.filter("a@b.com")
        f.filter("c@d.com")
        stats = f.stats()
        assert stats.total_filtered == 2
        assert stats.by_rule.get("email", 0) >= 2

    def test_stats_zero_for_clean_text(self):
        f = OutputFilter()
        f.filter("No sensitive data here")
        stats = f.stats()
        assert stats.total_filtered == 1
        assert stats.total_matches == 0


# ---------------------------------------------------------------------------
# Multiple matches in one text
# ---------------------------------------------------------------------------

class TestMultipleMatches:
    def test_multiple_emails(self):
        f = OutputFilter()
        result = f.filter("Send to a@b.com and c@d.com")
        email_matches = [m for m in result.matches if m.rule_name == "email"]
        assert len(email_matches) == 2
        assert "a@b.com" not in result.filtered
        assert "c@d.com" not in result.filtered

    def test_mixed_pattern_types(self):
        f = OutputFilter()
        text = "Email user@test.com or visit https://example.com"
        result = f.filter(text)
        rule_names = {m.rule_name for m in result.matches}
        assert "email" in rule_names
        assert "url" in rule_names
        assert result.rules_applied >= 2


# ---------------------------------------------------------------------------
# Rule removal
# ---------------------------------------------------------------------------

class TestRuleRemoval:
    def test_remove_existing_rule(self):
        f = OutputFilter()
        f.remove_rule("email")
        names = [r.name for r in f.list_rules()]
        assert "email" not in names

    def test_remove_nonexistent_raises_key_error(self):
        f = OutputFilter()
        with pytest.raises(KeyError):
            f.remove_rule("does_not_exist")

    def test_removed_rule_no_longer_filters(self):
        f = OutputFilter()
        f.remove_rule("email")
        result = f.filter("Contact user@example.com")
        email_matches = [m for m in result.matches if m.rule_name == "email"]
        assert len(email_matches) == 0


# ---------------------------------------------------------------------------
# FilterOutput dataclass
# ---------------------------------------------------------------------------

class TestFilterOutput:
    def test_original_preserved(self):
        f = OutputFilter()
        text = "My SSN is 111-22-3333"
        result = f.filter(text)
        assert result.original == text
        assert result.filtered != text
