"""Tests for secrets scanner."""

import pytest
from pathlib import Path

from sentinel.scanners.secrets_scanner import SecretsScanner, _shannon_entropy, _is_false_positive


class TestAWSKeys:
    def test_detects_aws_access_key(self):
        scanner = SecretsScanner()
        code = 'aws_key = "AKIAIOSFODNN7EXAMPLE"'
        findings = scanner.scan(code)
        assert any(f.category == "aws_access_key" for f in findings)

    def test_detects_aws_asia_key(self):
        scanner = SecretsScanner()
        code = 'key = "ASIAISAMPLEKEYHERE12"'
        findings = scanner.scan(code)
        assert any(f.category == "aws_access_key" for f in findings)

    def test_detects_aws_secret_key(self):
        scanner = SecretsScanner()
        code = 'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        findings = scanner.scan(code)
        assert any(f.category == "aws_secret_key" for f in findings)


class TestGitHubTokens:
    def test_detects_ghp_token(self):
        scanner = SecretsScanner()
        code = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        findings = scanner.scan(code)
        assert any(f.category == "github_token" for f in findings)

    def test_detects_gho_token(self):
        scanner = SecretsScanner()
        code = 'GITHUB_TOKEN = "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        findings = scanner.scan(code)
        assert any(f.category == "github_oauth_token" for f in findings)

    def test_detects_ghs_token(self):
        scanner = SecretsScanner()
        code = 'token = "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        findings = scanner.scan(code)
        assert any(f.category == "github_app_token" for f in findings)

    def test_detects_github_pat(self):
        scanner = SecretsScanner()
        code = 'token = "github_pat_1234567890abcdefghijkl_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567"'
        findings = scanner.scan(code)
        assert any(f.category == "github_pat" for f in findings)


class TestGoogleKeys:
    def test_detects_google_api_key(self):
        scanner = SecretsScanner()
        code = 'GOOGLE_API_KEY = "AIzaSyA1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q"'
        findings = scanner.scan(code)
        assert any(f.category == "google_api_key" for f in findings)

    def test_detects_google_oauth_secret(self):
        scanner = SecretsScanner()
        code = 'client_secret = "GOCSPX-ABCDEFGHIJKLMNOPQRSTUVWXyz01"'
        findings = scanner.scan(code)
        assert any(f.category == "google_oauth_secret" for f in findings)


class TestStripeKeys:
    def test_detects_stripe_secret_key(self):
        scanner = SecretsScanner()
        prefix = "sk_" + "live_"
        code = f'STRIPE_KEY = "{prefix}{"a" * 24}"'
        findings = scanner.scan(code)
        assert any(f.category == "stripe_secret_key" for f in findings)

    def test_detects_stripe_publishable_key(self):
        scanner = SecretsScanner()
        prefix = "pk_" + "live_"
        code = f'PK = "{prefix}{"a" * 24}"'
        findings = scanner.scan(code)
        assert any(f.category == "stripe_publishable_key" for f in findings)


class TestSlackTokens:
    def test_detects_slack_bot_token(self):
        scanner = SecretsScanner()
        # Construct programmatically to avoid push protection
        token = "xoxb" + "-1234567890-1234567890-" + "A" * 24
        code = f'SLACK_TOKEN = "{token}"'
        findings = scanner.scan(code)
        assert any(f.category == "slack_bot_token" for f in findings)

    def test_detects_slack_webhook(self):
        scanner = SecretsScanner()
        # Construct programmatically to avoid push protection
        hook_url = "https://hooks.slack.com/services/" + "T" + "0" * 8 + "/B" + "0" * 8 + "/" + "A" * 24
        code = f'url = "{hook_url}"'
        findings = scanner.scan(code)
        assert any(f.category == "slack_webhook" for f in findings)


class TestAIProviderKeys:
    def test_detects_openai_project_key(self):
        scanner = SecretsScanner()
        code = 'OPENAI_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdef"'
        findings = scanner.scan(code)
        assert any(f.category == "openai_project_key" for f in findings)

    def test_detects_anthropic_key(self):
        scanner = SecretsScanner()
        key = "sk-ant-api03-" + "A" * 93
        code = f'ANTHROPIC_API_KEY = "{key}"'
        findings = scanner.scan(code)
        assert any(f.category == "anthropic_api_key" for f in findings)


class TestOtherProviders:
    def test_detects_sendgrid_key(self):
        scanner = SecretsScanner()
        # Construct programmatically to avoid push protection
        sg_key = "SG." + "A" * 22 + "." + "B" * 43
        code = f'SENDGRID_KEY = "{sg_key}"'
        findings = scanner.scan(code)
        assert any(f.category == "sendgrid_api_key" for f in findings)

    def test_detects_npm_token(self):
        scanner = SecretsScanner()
        code = 'NPM_TOKEN = "npm_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        findings = scanner.scan(code)
        assert any(f.category == "npm_token" for f in findings)

    def test_detects_pypi_token(self):
        scanner = SecretsScanner()
        code = 'PYPI = "pypi-' + "A" * 55 + '"'
        findings = scanner.scan(code)
        assert any(f.category == "pypi_token" for f in findings)

    def test_detects_mailgun_key(self):
        scanner = SecretsScanner()
        code = 'MAILGUN = "key-abcdef1234567890abcdef1234567890"'
        findings = scanner.scan(code)
        assert any(f.category == "mailgun_api_key" for f in findings)


class TestPrivateKeys:
    def test_detects_rsa_private_key(self):
        scanner = SecretsScanner()
        code = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ...'
        findings = scanner.scan(code)
        assert any(f.category == "rsa_private_key" for f in findings)

    def test_detects_ec_private_key(self):
        scanner = SecretsScanner()
        code = '-----BEGIN EC PRIVATE KEY-----\nMHQCAQEE...'
        findings = scanner.scan(code)
        assert any(f.category == "ec_private_key" for f in findings)

    def test_detects_generic_private_key(self):
        scanner = SecretsScanner()
        code = 'key = """-----BEGIN PRIVATE KEY-----\nMIIEvQI..."""'
        findings = scanner.scan(code)
        assert any(f.category == "generic_private_key" for f in findings)

    def test_detects_openssh_private_key(self):
        scanner = SecretsScanner()
        code = '-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC...'
        findings = scanner.scan(code)
        assert any(f.category == "openssh_private_key" for f in findings)

    def test_detects_pgp_private_key(self):
        scanner = SecretsScanner()
        code = '-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: 1...'
        findings = scanner.scan(code)
        assert any(f.category == "pgp_private_key" for f in findings)


class TestGenericSecrets:
    def test_detects_hardcoded_password(self):
        scanner = SecretsScanner()
        code = 'password = "Sup3rS3cr3t!Pass"'
        findings = scanner.scan(code)
        assert any(f.category == "hardcoded_password" for f in findings)

    def test_detects_hardcoded_api_key(self):
        scanner = SecretsScanner()
        code = 'api_key = "a1b2c3d4e5f6g7h8i9j0"'
        findings = scanner.scan(code)
        assert any(f.category == "hardcoded_api_key" for f in findings)

    def test_detects_hardcoded_secret(self):
        scanner = SecretsScanner()
        code = 'client_secret = "myR3allyS3cretV4lue!"'
        findings = scanner.scan(code)
        assert any(f.category == "hardcoded_secret" for f in findings)

    def test_detects_bearer_token(self):
        scanner = SecretsScanner()
        code = 'Authorization = "Bearer eyJhbGciOiJIUzI1NiJ9.payload"'
        findings = scanner.scan(code)
        assert any(f.category == "bearer_token" for f in findings)


class TestConnectionStrings:
    def test_detects_postgres_url(self):
        scanner = SecretsScanner()
        code = 'DATABASE_URL = "postgresql://admin:s3cretP@ss@db.example.com:5432/mydb"'
        findings = scanner.scan(code)
        assert any(f.category == "database_url" for f in findings)
        assert any(f.risk == RiskLevel.CRITICAL for f in findings)

    def test_detects_mongodb_url(self):
        scanner = SecretsScanner()
        code = 'MONGO_URI = "mongodb+srv://user:s3cretPass@cluster.mongodb.net/db"'
        findings = scanner.scan(code)
        assert any(f.category == "database_url" for f in findings)

    def test_detects_redis_url(self):
        scanner = SecretsScanner()
        code = 'REDIS_URL = "redis://default:mYr3d1sP@ss@redis.example.com:6379"'
        findings = scanner.scan(code)
        assert any(f.category == "database_url" for f in findings)


class TestFalsePositives:
    def test_ignores_placeholder_password(self):
        scanner = SecretsScanner()
        code = 'password = "your_password_here"'
        findings = scanner.scan(code)
        assert not any(f.category == "hardcoded_password" for f in findings)

    def test_ignores_changeme(self):
        scanner = SecretsScanner()
        code = 'password = "changeme"'
        findings = scanner.scan(code)
        password_findings = [f for f in findings if f.category == "hardcoded_password"]
        assert len(password_findings) == 0

    def test_ignores_env_variable_reference(self):
        scanner = SecretsScanner()
        code = 'api_key = "${API_KEY}"'
        findings = scanner.scan(code)
        assert not any(f.category == "hardcoded_api_key" for f in findings)

    def test_ignores_os_environ(self):
        scanner = SecretsScanner()
        code = 'api_key = os.environ["API_KEY"]'
        findings = scanner.scan(code)
        assert not any(f.category == "hardcoded_api_key" for f in findings)

    def test_ignores_template_variables(self):
        scanner = SecretsScanner()
        code = 'secret = "{{SECRET_VALUE}}"'
        findings = scanner.scan(code)
        assert not any(f.category == "hardcoded_secret" for f in findings)

    def test_ignores_test_dummy_values(self):
        scanner = SecretsScanner()
        code = 'password = "test"'
        findings = scanner.scan(code)
        assert not any(f.category == "hardcoded_password" for f in findings)

    def test_ignores_low_entropy_values(self):
        scanner = SecretsScanner()
        code = 'api_key = "aaaaaaaa"'
        findings = scanner.scan(code)
        assert not any(f.category == "hardcoded_api_key" for f in findings)

    def test_ignores_comments_python(self):
        scanner = SecretsScanner()
        code = '# password = "realPassword123!"'
        findings = scanner.scan(code, filename="config.py")
        assert not any(f.category == "hardcoded_password" for f in findings)

    def test_ignores_comments_js(self):
        scanner = SecretsScanner()
        code = '// api_key = "realApiKey12345678"'
        findings = scanner.scan(code, filename="config.js")
        assert not any(f.category == "hardcoded_api_key" for f in findings)


class TestCleanCode:
    def test_clean_code_no_findings(self):
        scanner = SecretsScanner()
        code = """
import os

DB_PASSWORD = os.environ["DB_PASSWORD"]
API_KEY = os.getenv("API_KEY")
SECRET = config.get("secret")
"""
        findings = scanner.scan(code)
        assert len(findings) == 0

    def test_env_file_reference_clean(self):
        scanner = SecretsScanner()
        code = """
from dotenv import load_dotenv
load_dotenv()
key = os.environ["STRIPE_KEY"]
"""
        findings = scanner.scan(code)
        assert len(findings) == 0


class TestScannerMetadata:
    def test_finding_has_line_number(self):
        scanner = SecretsScanner()
        code = 'x = 1\ny = 2\ntoken = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        findings = scanner.scan(code)
        assert findings[0].metadata["line"] == 3

    def test_finding_redacts_secret(self):
        scanner = SecretsScanner()
        code = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        findings = scanner.scan(code)
        match = findings[0].metadata["match"]
        # Should be partially redacted
        assert "****" in match

    def test_scanner_name(self):
        scanner = SecretsScanner()
        code = 'x = "AKIAIOSFODNN7EXAMPLE"'
        findings = scanner.scan(code)
        assert findings[0].scanner == "secrets_scanner"

    def test_risk_level_critical(self):
        scanner = SecretsScanner()
        code = '-----BEGIN RSA PRIVATE KEY-----\nfoo'
        findings = scanner.scan(code)
        assert findings[0].risk == RiskLevel.CRITICAL


class TestSkipFiles:
    def test_skips_minified_js(self):
        scanner = SecretsScanner()
        code = 'var key="AKIAIOSFODNN7EXAMPLE"'
        findings = scanner.scan(code, filename="bundle.min.js")
        assert len(findings) == 0

    def test_skips_lock_files(self):
        scanner = SecretsScanner()
        code = 'key="AKIAIOSFODNN7EXAMPLE"'
        findings = scanner.scan(code, filename="package-lock.json")
        assert len(findings) == 0

    def test_skips_empty_code(self):
        scanner = SecretsScanner()
        findings = scanner.scan("")
        assert len(findings) == 0

    def test_skips_short_code(self):
        scanner = SecretsScanner()
        findings = scanner.scan("x = 1")
        assert len(findings) == 0


class TestScanFile:
    def test_scan_file(self, tmp_path):
        f = tmp_path / "config.py"
        f.write_text('AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
        scanner = SecretsScanner()
        findings = scanner.scan_file(f)
        assert any(fc.category == "aws_access_key" for fc in findings)

    def test_scan_file_nonexistent(self, tmp_path):
        scanner = SecretsScanner()
        findings = scanner.scan_file(tmp_path / "nope.py")
        assert len(findings) == 0


class TestScanDirectory:
    def test_scan_directory(self, tmp_path):
        (tmp_path / "app.py").write_text('KEY = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
        (tmp_path / "config.py").write_text('-----BEGIN RSA PRIVATE KEY-----\n', encoding="utf-8")
        scanner = SecretsScanner()
        findings = scanner.scan_directory(tmp_path)
        assert len(findings) >= 2

    def test_scan_directory_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "leaked.js").write_text('KEY = "AKIAIOSFODNN7EXAMPLE"', encoding="utf-8")
        scanner = SecretsScanner()
        findings = scanner.scan_directory(tmp_path)
        assert len(findings) == 0

    def test_scan_directory_max_files(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f'KEY{i} = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
        scanner = SecretsScanner()
        findings = scanner.scan_directory(tmp_path, max_files=3)
        # Only 3 files scanned
        files_scanned = {f.metadata["filename"] for f in findings}
        assert len(files_scanned) <= 3

    def test_scan_directory_empty(self, tmp_path):
        scanner = SecretsScanner()
        findings = scanner.scan_directory(tmp_path)
        assert len(findings) == 0


class TestEntropyAnalysis:
    def test_high_entropy_string(self):
        # A random-looking string should have high entropy
        entropy = _shannon_entropy("aB3$kL9mNp2Q")
        assert entropy > 3.0

    def test_low_entropy_string(self):
        entropy = _shannon_entropy("aaaaaaaa")
        assert entropy < 1.0

    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0


class TestIsFalsePositive:
    def test_placeholder(self):
        assert _is_false_positive("your_password_here")

    def test_env_reference(self):
        assert _is_false_positive("${API_KEY}")

    def test_template(self):
        assert _is_false_positive("{{SECRET}}")

    def test_real_secret(self):
        assert not _is_false_positive("aR3alS3cr3tV4lue!")

    def test_uniform_chars(self):
        assert _is_false_positive("xxxxxxxx")


class TestMultipleFindings:
    def test_multiple_secrets_in_file(self):
        scanner = SecretsScanner()
        code = """
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
-----BEGIN RSA PRIVATE KEY-----
"""
        findings = scanner.scan(code)
        categories = {f.category for f in findings}
        assert "aws_access_key" in categories
        assert "github_token" in categories
        assert "rsa_private_key" in categories

    def test_no_duplicate_findings(self):
        scanner = SecretsScanner()
        code = 'KEY = "AKIAIOSFODNN7EXAMPLE"'
        findings = scanner.scan(code)
        # Should only report once
        aws_findings = [f for f in findings if f.category == "aws_access_key"]
        assert len(aws_findings) == 1


from sentinel.core import RiskLevel
