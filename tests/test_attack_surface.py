"""Tests for sentinel.attack_surface — Attack Surface Mapper."""

import pytest

from sentinel.attack_surface import (
    AttackSurface,
    DataFlow,
    Endpoint,
    SurfaceAnalysis,
    Vulnerability,
)


# --- TestEndpoint ---


class TestEndpoint:
    def test_add_endpoint(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("api-gw", endpoint_type="api", exposure="public")
        endpoints = surface.get_exposed_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].name == "api-gw"
        assert endpoints[0].endpoint_type == "api"
        assert endpoints[0].exposure == "public"
        assert endpoints[0].auth_required is True

    def test_invalid_type(self):
        surface = AttackSurface("test-app")
        with pytest.raises(ValueError, match="Invalid endpoint type"):
            surface.add_endpoint("bad", endpoint_type="ftp")

    def test_invalid_exposure(self):
        surface = AttackSurface("test-app")
        with pytest.raises(ValueError, match="Invalid exposure"):
            surface.add_endpoint("bad", endpoint_type="api", exposure="secret")


# --- TestDataFlow ---


class TestDataFlow:
    def test_add_flow(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("src", endpoint_type="api")
        surface.add_endpoint("dst", endpoint_type="api")
        surface.add_data_flow("src", "dst", data_type="user_input")
        exported = surface.export()
        assert len(exported["data_flows"]) == 1
        assert exported["data_flows"][0]["source"] == "src"
        assert exported["data_flows"][0]["encrypted"] is True

    def test_unencrypted_flow(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("src", endpoint_type="api")
        surface.add_endpoint("dst", endpoint_type="api")
        surface.add_data_flow("src", "dst", data_type="pii", encrypted=False)
        unencrypted = surface.get_unencrypted_flows()
        assert len(unencrypted) == 1
        assert unencrypted[0].data_type == "pii"

    def test_missing_endpoint(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("src", endpoint_type="api")
        with pytest.raises(KeyError, match="not found"):
            surface.add_data_flow("src", "nonexistent", data_type="data")


# --- TestVulnerability ---


class TestVulnerability:
    def test_add_vulnerability(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("api", endpoint_type="api")
        surface.add_vulnerability("api", "high", "No rate limiting", cwe="CWE-770")
        exported = surface.export()
        assert len(exported["vulnerabilities"]) == 1
        assert exported["vulnerabilities"][0]["severity"] == "high"
        assert exported["vulnerabilities"][0]["cwe"] == "CWE-770"

    def test_invalid_severity(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("api", endpoint_type="api")
        with pytest.raises(ValueError, match="Invalid severity"):
            surface.add_vulnerability("api", "extreme", "Bad vuln")

    def test_missing_endpoint_vuln(self):
        surface = AttackSurface("test-app")
        with pytest.raises(KeyError, match="not found"):
            surface.add_vulnerability("ghost", "high", "Missing endpoint")


# --- TestAnalysis ---


class TestAnalysis:
    def test_analyze_basic(self):
        surface = AttackSurface("my-app")
        surface.add_endpoint("api", endpoint_type="api", exposure="public")
        surface.add_endpoint("db", endpoint_type="grpc", exposure="internal")
        analysis = surface.analyze()
        assert isinstance(analysis, SurfaceAnalysis)
        assert analysis.application == "my-app"
        assert analysis.total_endpoints == 2
        assert analysis.public_count == 1

    def test_risk_score(self):
        surface = AttackSurface("risky-app")
        surface.add_endpoint(
            "open-api", endpoint_type="api",
            exposure="public", auth_required=False,
        )
        surface.add_endpoint("db", endpoint_type="grpc")
        surface.add_data_flow(
            "open-api", "db", data_type="pii", encrypted=False,
        )
        surface.add_vulnerability("open-api", "critical", "SQL injection")
        # public=0.1, no_auth=0.15, unencrypted=0.1, critical=0.3 => 0.65
        score = surface.risk_score()
        assert score == pytest.approx(0.65, abs=0.01)

    def test_risk_level(self):
        surface = AttackSurface("app")
        surface.add_endpoint("a", endpoint_type="api", exposure="public")
        analysis = surface.analyze()
        # score = 0.1 => low
        assert analysis.risk_level == "low"

        surface_high = AttackSurface("app2")
        for i in range(7):
            surface_high.add_endpoint(
                f"ep-{i}", endpoint_type="api", exposure="public",
            )
        analysis_high = surface_high.analyze()
        # score = 7 * 0.1 = 0.7 => high
        assert analysis_high.risk_level == "high"


# --- TestExposed ---


class TestExposed:
    def test_exposed_endpoints(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("pub", endpoint_type="api", exposure="public")
        surface.add_endpoint("partner", endpoint_type="webhook", exposure="partner")
        surface.add_endpoint("internal", endpoint_type="grpc", exposure="internal")
        exposed = surface.get_exposed_endpoints()
        names = [ep.name for ep in exposed]
        assert "pub" in names
        assert "partner" in names
        assert "internal" not in names

    def test_unauthenticated(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("secure", endpoint_type="api", auth_required=True)
        surface.add_endpoint("open", endpoint_type="ui", auth_required=False)
        unauthed = surface.get_unauthenticated()
        assert len(unauthed) == 1
        assert unauthed[0].name == "open"

    def test_unencrypted_flows(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("a", endpoint_type="api")
        surface.add_endpoint("b", endpoint_type="api")
        surface.add_data_flow("a", "b", data_type="tokens", encrypted=True)
        surface.add_data_flow("a", "b", data_type="pii", encrypted=False)
        unencrypted = surface.get_unencrypted_flows()
        assert len(unencrypted) == 1
        assert unencrypted[0].data_type == "pii"


# --- TestRecommendations ---


class TestRecommendations:
    def test_auto_recommendations(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint(
            "open-api", endpoint_type="api",
            exposure="public", auth_required=False,
        )
        surface.add_endpoint("db", endpoint_type="grpc")
        surface.add_data_flow(
            "open-api", "db", data_type="pii", encrypted=False,
        )
        surface.add_vulnerability("open-api", "critical", "RCE")
        surface.add_vulnerability("open-api", "high", "No rate limiting")
        analysis = surface.analyze()
        recommendations_text = " ".join(analysis.recommendations)
        assert "authentication" in recommendations_text.lower()
        assert "encrypt" in recommendations_text.lower()
        assert "critical" in recommendations_text.lower()
        assert "high" in recommendations_text.lower()
        assert "exposed" in recommendations_text.lower()


# --- TestExport ---


class TestExport:
    def test_export_dict(self):
        surface = AttackSurface("export-app")
        surface.add_endpoint("api", endpoint_type="api", exposure="public")
        surface.add_endpoint("db", endpoint_type="grpc")
        surface.add_data_flow("api", "db", data_type="queries")
        surface.add_vulnerability("api", "medium", "Weak auth")
        exported = surface.export()
        assert exported["application"] == "export-app"
        assert len(exported["endpoints"]) == 2
        assert len(exported["data_flows"]) == 1
        assert len(exported["vulnerabilities"]) == 1
        assert exported["vulnerabilities"][0]["severity"] == "medium"

    def test_summary(self):
        surface = AttackSurface("summary-app")
        surface.add_endpoint(
            "api", endpoint_type="api",
            exposure="public", auth_required=False,
        )
        text = surface.summary()
        assert "summary-app" in text
        assert "Endpoints: 1" in text
        assert "Public/Partner: 1" in text
        assert "Unauthenticated: 1" in text
        assert "Risk score:" in text
        assert "Recommendations:" in text


# --- TestEdgeCases ---


class TestEdgeCases:
    def test_empty_surface(self):
        surface = AttackSurface("empty")
        analysis = surface.analyze()
        assert analysis.total_endpoints == 0
        assert analysis.risk_score == 0.0
        assert analysis.risk_level == "low"
        assert analysis.recommendations == []

    def test_risk_score_clamped_to_one(self):
        surface = AttackSurface("overloaded")
        for i in range(20):
            surface.add_endpoint(
                f"ep-{i}", endpoint_type="api",
                exposure="public", auth_required=False,
            )
        # 20 * (0.1 + 0.15) = 5.0, clamped to 1.0
        assert surface.risk_score() == 1.0

    def test_risk_level_critical_at_high_score(self):
        surface = AttackSurface("critical-app")
        for i in range(10):
            surface.add_endpoint(
                f"ep-{i}", endpoint_type="api",
                exposure="public", auth_required=False,
            )
        analysis = surface.analyze()
        assert analysis.risk_level == "critical"

    def test_data_flow_missing_source(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint("dst", endpoint_type="api")
        with pytest.raises(KeyError, match="not found"):
            surface.add_data_flow("missing-src", "dst", data_type="data")

    def test_endpoint_description_preserved(self):
        surface = AttackSurface("test-app")
        surface.add_endpoint(
            "api", endpoint_type="api",
            description="Main API gateway",
        )
        exported = surface.export()
        assert exported["endpoints"][0]["description"] == "Main API gateway"

    def test_all_endpoint_types_accepted(self):
        surface = AttackSurface("types-app")
        for etype in ("api", "websocket", "webhook", "grpc", "ui"):
            surface.add_endpoint(f"ep-{etype}", endpoint_type=etype)
        assert len(surface.export()["endpoints"]) == 5
