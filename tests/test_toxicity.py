"""Tests for the toxicity scanner."""

from sentinel.scanners.toxicity import ToxicityScanner
from sentinel.core import RiskLevel


class TestToxicityScanner:
    def test_threat_detection(self):
        scanner = ToxicityScanner()
        findings = scanner.scan("I will kill you if you don't comply")
        assert len(findings) >= 1
        assert any(f.risk == RiskLevel.CRITICAL for f in findings)
        assert any(f.metadata.get("toxicity_type") == "threat" for f in findings)

    def test_severe_insult(self):
        scanner = ToxicityScanner()
        findings = scanner.scan("You stupid worthless person")
        assert any(f.metadata.get("toxicity_type") == "severe_insult" for f in findings)

    def test_profanity(self):
        scanner = ToxicityScanner()
        findings = scanner.scan("What the fuck is this bullshit")
        assert any(f.metadata.get("toxicity_type") == "profanity" for f in findings)

    def test_clean_text_no_findings(self):
        scanner = ToxicityScanner()
        findings = scanner.scan("Thank you for your help, I appreciate it.")
        assert len(findings) == 0

    def test_aggressive_caps(self):
        scanner = ToxicityScanner()
        findings = scanner.scan("STOP DOING THIS RIGHT NOW!! I AM SO ANGRY AT THIS!!")
        assert any(f.metadata.get("toxicity_type") == "aggressive_tone" for f in findings)

    def test_kys_detection(self):
        scanner = ToxicityScanner()
        findings = scanner.scan("just kys already")
        assert any(f.risk >= RiskLevel.HIGH for f in findings)
