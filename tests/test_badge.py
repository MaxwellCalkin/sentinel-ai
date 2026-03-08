"""Tests for security badge generator."""

import pytest
from pathlib import Path

from sentinel.badge import generate_badge, generate_risk_badge, generate_badge_for_project


class TestGenerateBadge:
    def test_perfect_score(self):
        svg = generate_badge(100)
        assert "100/100" in svg
        assert "#4c1" in svg  # bright green
        assert "<svg" in svg

    def test_zero_score(self):
        svg = generate_badge(0)
        assert "0/100" in svg
        assert "#e05d44" in svg  # red

    def test_medium_score(self):
        svg = generate_badge(75)
        assert "75/100" in svg
        assert "#a4a61d" in svg  # yellow-green

    def test_custom_label(self):
        svg = generate_badge(90, label="security score")
        assert "security score" in svg

    def test_valid_svg(self):
        svg = generate_badge(85)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "xmlns" in svg

    def test_score_colors(self):
        # 90+ = bright green
        assert "#4c1" in generate_badge(95)
        # 80-89 = green
        assert "#97ca00" in generate_badge(85)
        # 70-79 = yellow-green
        assert "#a4a61d" in generate_badge(75)
        # 60-69 = yellow
        assert "#dfb317" in generate_badge(65)
        # 50-59 = orange
        assert "#fe7d37" in generate_badge(55)
        # <50 = red
        assert "#e05d44" in generate_badge(40)

    def test_aria_label(self):
        svg = generate_badge(88)
        assert 'aria-label="security: 88/100"' in svg


class TestGenerateRiskBadge:
    def test_no_risk(self):
        svg = generate_risk_badge("none")
        assert "none" in svg
        assert "#4c1" in svg

    def test_critical_risk(self):
        svg = generate_risk_badge("critical", findings=3)
        assert "critical (3)" in svg
        assert "#e05d44" in svg

    def test_high_risk(self):
        svg = generate_risk_badge("high")
        assert "high" in svg
        assert "#fe7d37" in svg

    def test_custom_label(self):
        svg = generate_risk_badge("low", label="risk level")
        assert "risk level" in svg


class TestGenerateBadgeForProject:
    def test_clean_project(self, tmp_path):
        svg, score = generate_badge_for_project(tmp_path)
        assert score >= 0
        assert "<svg" in svg

    def test_output_file(self, tmp_path):
        output_path = tmp_path / "badge.svg"
        svg, score = generate_badge_for_project(tmp_path, output=output_path)
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == svg

    def test_project_with_issues(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("colourama==0.1\n", encoding="utf-8")
        svg, score = generate_badge_for_project(tmp_path)
        assert score < 100
