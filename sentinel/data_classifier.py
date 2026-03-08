"""Data classification scanner for enterprise compliance.

Detects and classifies sensitive data into regulatory categories:
financial (PCI-DSS, SOX), health (HIPAA), personal (GDPR, CCPA),
and legal (attorney-client, trade secrets).

Usage:
    from sentinel.data_classifier import DataClassifier, DataCategory

    classifier = DataClassifier()
    result = classifier.classify("Patient John Doe, DOB 01/15/1990, MRN 12345")
    for finding in result.findings:
        print(f"{finding.category}: {finding.description}")
    # DataCategory.HEALTH: Medical Record Number detected
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from sentinel.core import Finding, RiskLevel


class DataCategory(Enum):
    """Regulatory data categories."""
    FINANCIAL = "financial"      # PCI-DSS, SOX, GLBA
    HEALTH = "health"            # HIPAA, HITECH
    PERSONAL = "personal"        # GDPR, CCPA, PIPEDA
    LEGAL = "legal"              # Attorney-client, trade secrets
    GOVERNMENT = "government"    # Classified, ITAR, EAR
    AUTHENTICATION = "auth"      # Credentials, tokens, keys


@dataclass
class ClassificationResult:
    """Result of data classification."""
    text: str
    findings: list[ClassifiedFinding] = field(default_factory=list)
    categories: set[DataCategory] = field(default_factory=set)
    highest_risk: RiskLevel = RiskLevel.NONE

    @property
    def has_sensitive_data(self) -> bool:
        return len(self.findings) > 0

    @property
    def summary(self) -> dict:
        return {
            "sensitive": self.has_sensitive_data,
            "categories": [c.value for c in sorted(self.categories, key=lambda c: c.value)],
            "finding_count": len(self.findings),
            "highest_risk": self.highest_risk.value,
        }


@dataclass
class ClassifiedFinding:
    """A classified data finding with regulatory context."""
    category: DataCategory
    subcategory: str
    description: str
    risk: RiskLevel
    span: tuple[int, int] | None = None
    regulations: list[str] = field(default_factory=list)

    def to_finding(self) -> Finding:
        return Finding(
            scanner="data_classifier",
            category=f"data_{self.category.value}",
            description=self.description,
            risk=self.risk,
            span=self.span,
            metadata={
                "data_category": self.category.value,
                "subcategory": self.subcategory,
                "regulations": self.regulations,
            },
        )


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_FINANCIAL_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, list[str]]] = [
    (
        "credit_card",
        re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[- ]?\d{4}[- ]?\d{4}[- ]?\d{3,4}\b"),
        RiskLevel.CRITICAL,
        ["PCI-DSS"],
    ),
    (
        "bank_account",
        re.compile(r"\b(?:account\s*(?:number|#|no\.?)\s*:?\s*)\d{8,17}\b", re.IGNORECASE),
        RiskLevel.HIGH,
        ["GLBA", "SOX"],
    ),
    (
        "routing_number",
        re.compile(r"\b(?:routing\s*(?:number|#|no\.?)\s*:?\s*)\d{9}\b", re.IGNORECASE),
        RiskLevel.HIGH,
        ["GLBA"],
    ),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b"),
        RiskLevel.HIGH,
        ["PCI-DSS", "GDPR"],
    ),
    (
        "swift_code",
        re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
        RiskLevel.MEDIUM,
        ["GLBA"],
    ),
]

_HEALTH_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, list[str]]] = [
    (
        "medical_record_number",
        re.compile(r"\b(?:MRN|medical\s*record\s*(?:number|#|no\.?))\s*:?\s*\d{4,10}\b", re.IGNORECASE),
        RiskLevel.CRITICAL,
        ["HIPAA"],
    ),
    (
        "diagnosis_code",
        re.compile(r"\b(?:ICD[- ]?10[- ]?(?:CM|PCS)?)\s*:?\s*[A-Z]\d{2}(?:\.\d{1,4})?\b", re.IGNORECASE),
        RiskLevel.HIGH,
        ["HIPAA"],
    ),
    (
        "npi_number",
        re.compile(r"\b(?:NPI)\s*:?\s*\d{10}\b", re.IGNORECASE),
        RiskLevel.HIGH,
        ["HIPAA"],
    ),
    (
        "health_condition",
        re.compile(
            r"\b(?:diagnosed?\s+with|patient\s+has|suffering\s+from|treatment\s+for)\s+"
            r"[A-Za-z\s]{3,40}\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["HIPAA", "HITECH"],
    ),
    (
        "prescription",
        re.compile(
            r"\b(?:prescribed?|medication|dosage|rx)\s*:?\s*[A-Za-z]+\s+\d+\s*(?:mg|ml|mcg|units?)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["HIPAA"],
    ),
]

_PERSONAL_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, list[str]]] = [
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        RiskLevel.CRITICAL,
        ["GDPR", "CCPA", "PIPEDA"],
    ),
    (
        "date_of_birth",
        re.compile(
            r"\b(?:DOB|date\s*of\s*birth|born\s*(?:on)?)\s*:?\s*"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["GDPR", "CCPA", "HIPAA"],
    ),
    (
        "passport",
        re.compile(r"\b(?:passport\s*(?:number|#|no\.?))\s*:?\s*[A-Z0-9]{6,9}\b", re.IGNORECASE),
        RiskLevel.CRITICAL,
        ["GDPR", "CCPA"],
    ),
    (
        "drivers_license",
        re.compile(
            r"\b(?:driver'?s?\s*licen[sc]e\s*(?:number|#|no\.?))\s*:?\s*[A-Z0-9]{5,15}\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["GDPR", "CCPA"],
    ),
    (
        "national_id",
        re.compile(
            r"\b(?:national\s*id|citizen\s*id|ID\s*number)\s*:?\s*[A-Z0-9]{6,15}\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["GDPR"],
    ),
    (
        "biometric",
        re.compile(
            r"\b(?:fingerprint|retina|iris|facial\s*recognition|biometric)\s+"
            r"(?:data|scan|template|hash)\b",
            re.IGNORECASE,
        ),
        RiskLevel.CRITICAL,
        ["GDPR", "BIPA", "CCPA"],
    ),
]

_LEGAL_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, list[str]]] = [
    (
        "attorney_client",
        re.compile(
            r"\b(?:attorney[- ]client\s*privilege|privileged\s*(?:and\s*)?confidential|"
            r"legal\s*privilege|work\s*product\s*doctrine)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["Attorney-Client Privilege"],
    ),
    (
        "trade_secret",
        re.compile(
            r"\b(?:trade\s*secret|proprietary\s*(?:information|formula|process)|"
            r"confidential\s*(?:business|intellectual)\s*property)\b",
            re.IGNORECASE,
        ),
        RiskLevel.HIGH,
        ["DTSA", "UTSA"],
    ),
    (
        "nda_content",
        re.compile(
            r"\b(?:non[- ]?disclosure\s*agreement|NDA|confidentiality\s*agreement)\b",
            re.IGNORECASE,
        ),
        RiskLevel.MEDIUM,
        ["Contract Law"],
    ),
]

_GOVERNMENT_PATTERNS: list[tuple[str, re.Pattern, RiskLevel, list[str]]] = [
    (
        "classification_marking",
        re.compile(
            r"\b(?:TOP\s*SECRET|SECRET|CONFIDENTIAL|RESTRICTED|UNCLASSIFIED)[/\\]+"
            r"(?:NOFORN|REL\s*TO|ORCON|SCI)\b",
            re.IGNORECASE,
        ),
        RiskLevel.CRITICAL,
        ["EO 13526"],
    ),
    (
        "export_controlled",
        re.compile(
            r"\b(?:ITAR|EAR|export[- ]controlled|ECCN\s*\d[A-Z]\d{3})\b",
            re.IGNORECASE,
        ),
        RiskLevel.CRITICAL,
        ["ITAR", "EAR"],
    ),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DataClassifier:
    """Classify text into regulatory data categories.

    Scans text for patterns matching financial, health, personal, legal,
    and government data categories, returning findings tagged with the
    applicable regulations.
    """

    def __init__(
        self,
        categories: Sequence[DataCategory] | None = None,
        min_risk: RiskLevel = RiskLevel.MEDIUM,
    ):
        """
        Args:
            categories: Which categories to scan for (None = all).
            min_risk: Minimum risk level to include in results.
        """
        self._min_risk = min_risk
        self._patterns: list[tuple[DataCategory, str, re.Pattern, RiskLevel, list[str]]] = []

        active = set(categories) if categories else set(DataCategory)

        if DataCategory.FINANCIAL in active:
            for name, pat, risk, regs in _FINANCIAL_PATTERNS:
                self._patterns.append((DataCategory.FINANCIAL, name, pat, risk, regs))

        if DataCategory.HEALTH in active:
            for name, pat, risk, regs in _HEALTH_PATTERNS:
                self._patterns.append((DataCategory.HEALTH, name, pat, risk, regs))

        if DataCategory.PERSONAL in active:
            for name, pat, risk, regs in _PERSONAL_PATTERNS:
                self._patterns.append((DataCategory.PERSONAL, name, pat, risk, regs))

        if DataCategory.LEGAL in active:
            for name, pat, risk, regs in _LEGAL_PATTERNS:
                self._patterns.append((DataCategory.LEGAL, name, pat, risk, regs))

        if DataCategory.GOVERNMENT in active:
            for name, pat, risk, regs in _GOVERNMENT_PATTERNS:
                self._patterns.append((DataCategory.GOVERNMENT, name, pat, risk, regs))

    def classify(self, text: str) -> ClassificationResult:
        """Classify text and return all findings."""
        findings: list[ClassifiedFinding] = []
        categories: set[DataCategory] = set()
        highest_risk = RiskLevel.NONE

        for category, subcategory, pattern, risk, regulations in self._patterns:
            if risk < self._min_risk:
                continue

            for match in pattern.finditer(text):
                finding = ClassifiedFinding(
                    category=category,
                    subcategory=subcategory,
                    description=f"{subcategory.replace('_', ' ').title()} detected",
                    risk=risk,
                    span=(match.start(), match.end()),
                    regulations=regulations,
                )
                findings.append(finding)
                categories.add(category)
                if risk > highest_risk:
                    highest_risk = risk

        return ClassificationResult(
            text=text,
            findings=findings,
            categories=categories,
            highest_risk=highest_risk,
        )

    def classify_as_findings(self, text: str) -> list[Finding]:
        """Classify and return standard Sentinel Finding objects."""
        result = self.classify(text)
        return [f.to_finding() for f in result.findings]
