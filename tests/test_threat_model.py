"""Tests for STRIDE-based threat modeling."""

import pytest
from sentinel.threat_model import (
    ThreatModel,
    ThreatAnalysis,
    Component,
    Threat,
    STRIDE_CATEGORIES,
    VALID_SEVERITIES,
)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class TestComponent:
    def test_add_component(self):
        model = ThreatModel("test-system")
        model.add_component("llm", component_type="model", trust_level="internal")
        analysis = model.analyze()
        assert len(analysis.components) == 1
        assert analysis.components[0].name == "llm"
        assert analysis.components[0].component_type == "model"
        assert analysis.components[0].trust_level == "internal"

    def test_invalid_component_type(self):
        model = ThreatModel("test-system")
        with pytest.raises(ValueError, match="Invalid component type"):
            model.add_component("thing", component_type="widget")

    def test_trust_levels(self):
        model = ThreatModel("test-system")
        model.add_component("frontend", component_type="user_interface", trust_level="external")
        model.add_component("gateway", component_type="api", trust_level="boundary")
        model.add_component("backend", component_type="service", trust_level="internal")
        analysis = model.analyze()
        trust_levels = [c.trust_level for c in analysis.components]
        assert "external" in trust_levels
        assert "boundary" in trust_levels
        assert "internal" in trust_levels

    def test_invalid_trust_level(self):
        model = ThreatModel("test-system")
        with pytest.raises(ValueError, match="Invalid trust level"):
            model.add_component("svc", trust_level="secret")


# ---------------------------------------------------------------------------
# Threats
# ---------------------------------------------------------------------------

class TestThreat:
    def test_add_threat(self):
        model = ThreatModel("test-system")
        model.add_component("llm", component_type="model")
        model.add_threat("llm", "tampering", "Prompt injection", severity="high")
        analysis = model.analyze()
        assert analysis.total_threats == 1
        assert analysis.threats[0].category == "tampering"
        assert analysis.threats[0].severity == "high"

    def test_invalid_category(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        with pytest.raises(ValueError, match="Invalid STRIDE category"):
            model.add_threat("svc", "hacking", "Not a real category")

    def test_invalid_severity(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        with pytest.raises(ValueError, match="Invalid severity"):
            model.add_threat("svc", "spoofing", "Bad severity", severity="extreme")

    def test_missing_component(self):
        model = ThreatModel("test-system")
        with pytest.raises(KeyError, match="not found"):
            model.add_threat("ghost", "spoofing", "No such component")

    def test_default_severity_is_medium(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "Default severity test")
        assert model.get_threats()[0].severity == "medium"

    def test_threat_with_mitigation_is_mitigated(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "With fix", mitigation="Use MFA")
        assert model.get_threats()[0].mitigated is True

    def test_threat_without_mitigation_is_unmitigated(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "No fix")
        assert model.get_threats()[0].mitigated is False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_analyze_basic(self):
        model = ThreatModel("chatbot", description="Support chatbot")
        model.add_component("llm", component_type="model")
        model.add_component("api", component_type="api")
        model.add_threat("llm", "tampering", "Prompt injection", severity="high")
        model.add_threat("api", "denial_of_service", "Rate limit bypass", severity="medium")

        analysis = model.analyze()
        assert analysis.system_name == "chatbot"
        assert analysis.total_threats == 2
        assert len(analysis.components) == 2

    def test_risk_score(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        # One unmitigated critical threat: weight 4, max 4*1=4, score = 4/4 = 1.0
        model.add_threat("svc", "elevation_of_privilege", "Root access", severity="critical")
        analysis = model.analyze()
        assert analysis.risk_score == 1.0

    def test_risk_score_with_mixed_threats(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        # Unmitigated high (3) + mitigated critical (0) = 3, max = 2*4 = 8
        model.add_threat("svc", "tampering", "Injection", severity="high")
        model.add_threat(
            "svc", "spoofing", "Identity theft", severity="critical",
            mitigation="Use strong auth",
        )
        analysis = model.analyze()
        assert analysis.risk_score == pytest.approx(3 / 8, abs=0.001)

    def test_mitigated_vs_unmitigated(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "Unmitigated threat")
        model.add_threat("svc", "tampering", "Mitigated threat", mitigation="Input validation")
        model.add_threat("svc", "repudiation", "Another unmitigated")
        analysis = model.analyze()
        assert analysis.mitigated_count == 1
        assert analysis.unmitigated_count == 2

    def test_risk_score_zero_when_all_mitigated(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "Fixed", severity="critical", mitigation="MFA")
        model.add_threat("svc", "tampering", "Fixed", severity="high", mitigation="Validation")
        analysis = model.analyze()
        assert analysis.risk_score == 0.0

    def test_risk_score_zero_when_no_threats(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        analysis = model.analyze()
        assert analysis.risk_score == 0.0

    def test_by_category_counts(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "A")
        model.add_threat("svc", "spoofing", "B")
        model.add_threat("svc", "tampering", "C")
        analysis = model.analyze()
        assert analysis.by_category["spoofing"] == 2
        assert analysis.by_category["tampering"] == 1

    def test_by_severity_counts(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "A", severity="low")
        model.add_threat("svc", "tampering", "B", severity="low")
        model.add_threat("svc", "repudiation", "C", severity="critical")
        analysis = model.analyze()
        assert analysis.by_severity["low"] == 2
        assert analysis.by_severity["critical"] == 1


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilter:
    def _build_model_with_threats(self) -> ThreatModel:
        model = ThreatModel("filter-test")
        model.add_component("llm", component_type="model")
        model.add_component("api", component_type="api")
        model.add_threat("llm", "tampering", "Prompt injection")
        model.add_threat("llm", "information_disclosure", "Data leakage")
        model.add_threat("api", "denial_of_service", "Rate limit bypass")
        model.add_threat("api", "spoofing", "Token forgery")
        return model

    def test_get_by_component(self):
        model = self._build_model_with_threats()
        llm_threats = model.get_threats(component="llm")
        assert len(llm_threats) == 2
        assert all(t.component == "llm" for t in llm_threats)

    def test_get_by_category(self):
        model = self._build_model_with_threats()
        tampering = model.get_threats(category="tampering")
        assert len(tampering) == 1
        assert tampering[0].description == "Prompt injection"

    def test_get_all(self):
        model = self._build_model_with_threats()
        all_threats = model.get_threats()
        assert len(all_threats) == 4

    def test_get_by_component_and_category(self):
        model = self._build_model_with_threats()
        result = model.get_threats(component="api", category="spoofing")
        assert len(result) == 1
        assert result[0].description == "Token forgery"

    def test_get_returns_empty_for_no_match(self):
        model = self._build_model_with_threats()
        result = model.get_threats(component="nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------

class TestMitigation:
    def test_add_mitigation(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "Identity issue")
        model.add_mitigation(0, "Implement MFA")
        threats = model.get_threats()
        assert threats[0].mitigation == "Implement MFA"

    def test_mitigation_marks_mitigated(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "Identity issue")
        assert model.get_threats()[0].mitigated is False
        model.add_mitigation(0, "Implement MFA")
        assert model.get_threats()[0].mitigated is True

    def test_mitigation_invalid_index(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "One threat")
        with pytest.raises(IndexError):
            model.add_mitigation(5, "Out of range")

    def test_mitigation_reduces_risk_score(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "tampering", "Injection", severity="critical")
        score_before = model.analyze().risk_score
        model.add_mitigation(0, "Input validation")
        score_after = model.analyze().risk_score
        assert score_after < score_before


# ---------------------------------------------------------------------------
# Risk matrix
# ---------------------------------------------------------------------------

class TestMatrix:
    def test_risk_matrix(self):
        model = ThreatModel("test-system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "A", severity="high")
        model.add_threat("svc", "spoofing", "B", severity="high")
        model.add_threat("svc", "tampering", "C", severity="low")
        model.add_threat("svc", "denial_of_service", "D", severity="critical")

        matrix = model.risk_matrix()

        assert matrix["high"]["spoofing"] == 2
        assert matrix["low"]["tampering"] == 1
        assert matrix["critical"]["denial_of_service"] == 1
        assert matrix["medium"]["spoofing"] == 0

    def test_risk_matrix_has_all_severities_and_categories(self):
        model = ThreatModel("empty-matrix")
        matrix = model.risk_matrix()
        for severity in VALID_SEVERITIES:
            assert severity in matrix
            for category in STRIDE_CATEGORIES:
                assert category in matrix[severity]
                assert matrix[severity][category] == 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export(self):
        model = ThreatModel("export-test", description="Test system")
        model.add_component("llm", component_type="model", trust_level="internal")
        model.add_threat("llm", "tampering", "Injection", severity="high", mitigation="Filter")

        exported = model.export()

        assert exported["system_name"] == "export-test"
        assert exported["description"] == "Test system"
        assert len(exported["components"]) == 1
        assert exported["components"][0]["name"] == "llm"
        assert exported["components"][0]["component_type"] == "model"
        assert len(exported["threats"]) == 1
        assert exported["threats"][0]["category"] == "tampering"
        assert exported["threats"][0]["mitigated"] is True

    def test_summary(self):
        model = ThreatModel("summary-test", description="A test system")
        model.add_component("svc")
        model.add_threat("svc", "spoofing", "Threat A", severity="high")
        model.add_threat("svc", "tampering", "Threat B", mitigation="Fixed")

        text = model.summary()

        assert "summary-test" in text
        assert "A test system" in text
        assert "Total threats: 2" in text
        assert "Mitigated: 1" in text
        assert "Unmitigated: 1" in text
        assert "Risk score:" in text

    def test_summary_without_description(self):
        model = ThreatModel("minimal")
        text = model.summary()
        assert "minimal" in text
        assert "Description:" not in text
