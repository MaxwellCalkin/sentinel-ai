"""Tests for the data classification scanner."""

import pytest
from sentinel.data_classifier import (
    DataClassifier,
    DataCategory,
    ClassificationResult,
    ClassifiedFinding,
)
from sentinel.core import RiskLevel


# ---------------------------------------------------------------------------
# Financial data
# ---------------------------------------------------------------------------

class TestFinancialData:
    def test_credit_card_visa(self):
        c = DataClassifier()
        result = c.classify("Card: 4111-1111-1111-1111")
        assert DataCategory.FINANCIAL in result.categories
        assert any(f.subcategory == "credit_card" for f in result.findings)
        assert result.highest_risk == RiskLevel.CRITICAL

    def test_credit_card_mastercard(self):
        c = DataClassifier()
        result = c.classify("Card: 5500 0000 0000 0004")
        assert any(f.subcategory == "credit_card" for f in result.findings)

    def test_credit_card_amex(self):
        c = DataClassifier()
        result = c.classify("Card: 3782-8224-6310-005")
        assert any(f.subcategory == "credit_card" for f in result.findings)

    def test_bank_account(self):
        c = DataClassifier()
        result = c.classify("Account Number: 12345678901")
        assert any(f.subcategory == "bank_account" for f in result.findings)
        assert "GLBA" in result.findings[0].regulations

    def test_routing_number(self):
        c = DataClassifier()
        result = c.classify("Routing number: 021000021")
        assert any(f.subcategory == "routing_number" for f in result.findings)

    def test_iban(self):
        c = DataClassifier()
        result = c.classify("IBAN: GB29NWBK60161331926819")
        assert any(f.subcategory == "iban" for f in result.findings)

    def test_no_false_positive_on_regular_number(self):
        c = DataClassifier()
        result = c.classify("The answer is 42.")
        financial = [f for f in result.findings if f.category == DataCategory.FINANCIAL]
        assert len(financial) == 0


# ---------------------------------------------------------------------------
# Health data (HIPAA)
# ---------------------------------------------------------------------------

class TestHealthData:
    def test_medical_record_number(self):
        c = DataClassifier()
        result = c.classify("Patient MRN: 123456")
        assert DataCategory.HEALTH in result.categories
        assert any(f.subcategory == "medical_record_number" for f in result.findings)
        assert "HIPAA" in result.findings[0].regulations

    def test_diagnosis_code(self):
        c = DataClassifier()
        result = c.classify("Diagnosis: ICD-10 E11.65")
        assert any(f.subcategory == "diagnosis_code" for f in result.findings)

    def test_npi_number(self):
        c = DataClassifier()
        result = c.classify("Provider NPI: 1234567890")
        assert any(f.subcategory == "npi_number" for f in result.findings)

    def test_health_condition(self):
        c = DataClassifier()
        result = c.classify("Patient has diagnosed with type 2 diabetes")
        assert any(f.subcategory == "health_condition" for f in result.findings)

    def test_prescription(self):
        c = DataClassifier()
        result = c.classify("Prescribed Metformin 500 mg")
        assert any(f.subcategory == "prescription" for f in result.findings)


# ---------------------------------------------------------------------------
# Personal data (GDPR/CCPA)
# ---------------------------------------------------------------------------

class TestPersonalData:
    def test_ssn(self):
        c = DataClassifier()
        result = c.classify("SSN: 123-45-6789")
        assert DataCategory.PERSONAL in result.categories
        assert any(f.subcategory == "ssn" for f in result.findings)
        assert "GDPR" in result.findings[0].regulations

    def test_date_of_birth(self):
        c = DataClassifier()
        result = c.classify("DOB: 01/15/1990")
        assert any(f.subcategory == "date_of_birth" for f in result.findings)

    def test_passport(self):
        c = DataClassifier()
        result = c.classify("Passport number: AB1234567")
        assert any(f.subcategory == "passport" for f in result.findings)

    def test_drivers_license(self):
        c = DataClassifier()
        result = c.classify("Driver's license number: D1234567")
        assert any(f.subcategory == "drivers_license" for f in result.findings)

    def test_biometric(self):
        c = DataClassifier()
        result = c.classify("Fingerprint data collected for enrollment")
        assert any(f.subcategory == "biometric" for f in result.findings)
        assert "BIPA" in result.findings[0].regulations

    def test_national_id(self):
        c = DataClassifier()
        result = c.classify("National ID: AB12345678")
        assert any(f.subcategory == "national_id" for f in result.findings)


# ---------------------------------------------------------------------------
# Legal data
# ---------------------------------------------------------------------------

class TestLegalData:
    def test_attorney_client(self):
        c = DataClassifier()
        result = c.classify("This document is protected by attorney-client privilege")
        assert DataCategory.LEGAL in result.categories
        assert any(f.subcategory == "attorney_client" for f in result.findings)

    def test_privileged_confidential(self):
        c = DataClassifier()
        result = c.classify("PRIVILEGED AND CONFIDENTIAL")
        assert any(f.subcategory == "attorney_client" for f in result.findings)

    def test_trade_secret(self):
        c = DataClassifier()
        result = c.classify("This contains proprietary information about our process")
        assert any(f.subcategory == "trade_secret" for f in result.findings)

    def test_nda(self):
        c = DataClassifier()
        result = c.classify("Covered by Non-Disclosure Agreement #4521")
        assert any(f.subcategory == "nda_content" for f in result.findings)


# ---------------------------------------------------------------------------
# Government data
# ---------------------------------------------------------------------------

class TestGovernmentData:
    def test_classification_marking(self):
        c = DataClassifier()
        result = c.classify("TOP SECRET//NOFORN")
        assert DataCategory.GOVERNMENT in result.categories
        assert any(f.subcategory == "classification_marking" for f in result.findings)

    def test_export_controlled(self):
        c = DataClassifier()
        result = c.classify("This item is ITAR controlled")
        assert any(f.subcategory == "export_controlled" for f in result.findings)

    def test_eccn(self):
        c = DataClassifier()
        result = c.classify("ECCN 5A002 applies to this component")
        assert any(f.subcategory == "export_controlled" for f in result.findings)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_category_filter(self):
        c = DataClassifier(categories=[DataCategory.HEALTH])
        result = c.classify("SSN: 123-45-6789 and MRN: 12345")
        # Should only find health data, not personal
        assert all(f.category == DataCategory.HEALTH for f in result.findings)

    def test_min_risk_filter(self):
        c = DataClassifier(min_risk=RiskLevel.CRITICAL)
        result = c.classify("Non-disclosure agreement signed")
        # NDA is MEDIUM risk, should be filtered out
        assert not any(f.subcategory == "nda_content" for f in result.findings)

    def test_all_categories_default(self):
        c = DataClassifier()
        text = (
            "Card: 4111-1111-1111-1111, SSN: 123-45-6789, "
            "MRN: 12345, attorney-client privilege, TOP SECRET//NOFORN"
        )
        result = c.classify(text)
        assert DataCategory.FINANCIAL in result.categories
        assert DataCategory.PERSONAL in result.categories
        assert DataCategory.HEALTH in result.categories
        assert DataCategory.LEGAL in result.categories
        assert DataCategory.GOVERNMENT in result.categories


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_has_sensitive_data(self):
        c = DataClassifier()
        safe = c.classify("The weather is nice today.")
        assert not safe.has_sensitive_data

        sensitive = c.classify("SSN: 123-45-6789")
        assert sensitive.has_sensitive_data

    def test_summary(self):
        c = DataClassifier()
        result = c.classify("Card: 4111-1111-1111-1111")
        s = result.summary
        assert s["sensitive"] is True
        assert "financial" in s["categories"]
        assert s["finding_count"] > 0

    def test_empty_text(self):
        c = DataClassifier()
        result = c.classify("")
        assert not result.has_sensitive_data
        assert result.highest_risk == RiskLevel.NONE


# ---------------------------------------------------------------------------
# ClassifiedFinding
# ---------------------------------------------------------------------------

class TestClassifiedFinding:
    def test_to_finding(self):
        f = ClassifiedFinding(
            category=DataCategory.HEALTH,
            subcategory="medical_record_number",
            description="MRN detected",
            risk=RiskLevel.CRITICAL,
            regulations=["HIPAA"],
        )
        finding = f.to_finding()
        assert finding.scanner == "data_classifier"
        assert finding.category == "data_health"
        assert finding.risk == RiskLevel.CRITICAL
        assert "HIPAA" in finding.metadata["regulations"]


# ---------------------------------------------------------------------------
# Multi-category detection
# ---------------------------------------------------------------------------

class TestMultiCategory:
    def test_mixed_data(self):
        c = DataClassifier()
        text = (
            "Patient John Doe, DOB: 03/15/1985, SSN: 987-65-4321. "
            "Diagnosed with hypertension. Prescribed Lisinopril 10 mg. "
            "Bill to card 4111-1111-1111-1111. "
            "PRIVILEGED AND CONFIDENTIAL"
        )
        result = c.classify(text)
        assert len(result.categories) >= 3  # health, personal, financial, legal
        assert result.highest_risk == RiskLevel.CRITICAL
        assert len(result.findings) >= 4

    def test_classify_as_findings(self):
        c = DataClassifier()
        findings = c.classify_as_findings("SSN: 123-45-6789")
        assert len(findings) > 0
        assert findings[0].scanner == "data_classifier"
