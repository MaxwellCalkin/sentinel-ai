"""Tests for model card metadata."""

import pytest
from sentinel.model_card import ModelCard, CardValidation, SafetyRating, IntendedUse


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

class TestCreation:
    def test_basic_creation(self):
        card = ModelCard("test-model", "1.0")
        assert card.name == "test-model"
        assert card.version == "1.0"
        assert card.description == ""

    def test_with_description(self):
        card = ModelCard("test-model", "1.0", description="A test model")
        assert card.description == "A test model"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_set_capabilities(self):
        card = ModelCard("m", "1.0")
        card.set_capabilities(["text-generation", "summarization"])
        exported = card.to_dict()
        assert exported["capabilities"] == ["text-generation", "summarization"]

    def test_empty_capabilities(self):
        card = ModelCard("m", "1.0")
        card.set_capabilities([])
        exported = card.to_dict()
        assert exported["capabilities"] == []


# ---------------------------------------------------------------------------
# Safety ratings
# ---------------------------------------------------------------------------

class TestSafety:
    def test_safety_ratings(self):
        card = ModelCard("m", "1.0")
        card.set_safety_ratings({"toxicity": "low", "bias": "medium"})
        exported = card.to_dict()
        assert exported["safety_ratings"] == {"toxicity": "low", "bias": "medium"}

    def test_invalid_safety_level(self):
        card = ModelCard("m", "1.0")
        with pytest.raises(ValueError, match="Invalid safety level"):
            card.set_safety_ratings({"toxicity": "banana"})

    def test_all_valid_levels(self):
        card = ModelCard("m", "1.0")
        card.set_safety_ratings({
            "a": "low",
            "b": "medium",
            "c": "high",
            "d": "critical",
        })
        exported = card.to_dict()
        assert len(exported["safety_ratings"]) == 4

    def test_safety_rating_dataclass(self):
        rating = SafetyRating(category="toxicity", level="low")
        assert rating.category == "toxicity"
        assert rating.level == "low"

    def test_safety_rating_dataclass_invalid(self):
        with pytest.raises(ValueError):
            SafetyRating(category="toxicity", level="invalid")


# ---------------------------------------------------------------------------
# Intended use
# ---------------------------------------------------------------------------

class TestIntendedUse:
    def test_set_intended_use(self):
        card = ModelCard("m", "1.0")
        card.set_intended_use("Customer support chatbot")
        exported = card.to_dict()
        assert exported["intended_use"]["primary"] == "Customer support chatbot"

    def test_out_of_scope(self):
        card = ModelCard("m", "1.0")
        card.set_intended_use(
            "Customer support",
            out_of_scope=["Medical advice", "Legal counsel"],
        )
        exported = card.to_dict()
        assert exported["intended_use"]["out_of_scope"] == [
            "Medical advice",
            "Legal counsel",
        ]

    def test_intended_use_dataclass(self):
        use = IntendedUse(primary="Testing", out_of_scope=["Production"])
        assert use.primary == "Testing"
        assert use.out_of_scope == ["Production"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_card(self):
        card = _build_complete_card()
        result = card.validate()
        assert result.valid is True
        assert result.missing_fields == []

    def test_missing_capabilities(self):
        card = ModelCard("m", "1.0")
        card.set_intended_use("Testing")
        result = card.validate()
        assert result.valid is False
        assert "capabilities" in result.missing_fields

    def test_missing_intended_use(self):
        card = ModelCard("m", "1.0")
        card.set_capabilities(["text-gen"])
        result = card.validate()
        assert result.valid is False
        assert "intended_use" in result.missing_fields

    def test_warnings_no_safety(self):
        card = ModelCard("m", "1.0")
        card.set_capabilities(["text-gen"])
        card.set_intended_use("Testing")
        result = card.validate()
        assert result.valid is True
        assert any("safety" in w.lower() for w in result.warnings)

    def test_warnings_no_limitations(self):
        card = ModelCard("m", "1.0")
        card.set_capabilities(["text-gen"])
        card.set_intended_use("Testing")
        result = card.validate()
        assert any("limitations" in w.lower() for w in result.warnings)

    def test_no_warnings_when_complete(self):
        card = _build_complete_card()
        card.set_limitations(["Cannot do math"])
        result = card.validate()
        assert result.warnings == []

    def test_card_validation_dataclass(self):
        validation = CardValidation(valid=True)
        assert validation.valid is True
        assert validation.missing_fields == []
        assert validation.warnings == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self):
        card = _build_complete_card()
        data = card.to_dict()
        assert data["name"] == "test-model"
        assert data["version"] == "2.0"
        assert isinstance(data["capabilities"], list)
        assert isinstance(data["safety_ratings"], dict)

    def test_from_dict(self):
        data = {
            "name": "imported-model",
            "version": "3.0",
            "description": "Imported",
            "capabilities": ["qa"],
            "limitations": ["slow"],
            "safety_ratings": {"toxicity": "low"},
            "intended_use": {
                "primary": "QA bot",
                "out_of_scope": ["Finance"],
            },
            "training_data": {
                "description": "Web corpus",
                "size": "1B tokens",
                "sources": ["web"],
            },
            "metrics": {"accuracy": 0.95},
        }
        card = ModelCard.from_dict(data)
        assert card.name == "imported-model"
        assert card.version == "3.0"
        assert card.description == "Imported"
        exported = card.to_dict()
        assert exported["capabilities"] == ["qa"]
        assert exported["limitations"] == ["slow"]
        assert exported["safety_ratings"] == {"toxicity": "low"}
        assert exported["intended_use"]["primary"] == "QA bot"
        assert exported["training_data"]["size"] == "1B tokens"
        assert exported["metrics"]["accuracy"] == 0.95

    def test_roundtrip(self):
        original = _build_complete_card()
        original.set_limitations(["Cannot do math"])
        original.set_training_data("Web text", size="10B tokens", sources=["web", "books"])
        original.set_metrics({"accuracy": 0.92, "f1": 0.88})

        data = original.to_dict()
        restored = ModelCard.from_dict(data)
        restored_data = restored.to_dict()

        assert data == restored_data

    def test_from_dict_minimal(self):
        card = ModelCard.from_dict({"name": "bare", "version": "0.1"})
        assert card.name == "bare"
        assert card.version == "0.1"
        assert card.to_dict()["capabilities"] == []

    def test_to_dict_no_intended_use(self):
        card = ModelCard("m", "1.0")
        data = card.to_dict()
        assert data["intended_use"] is None

    def test_to_dict_no_training_data(self):
        card = ModelCard("m", "1.0")
        data = card.to_dict()
        assert data["training_data"] is None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_format(self):
        card = _build_complete_card()
        card.set_metrics({"accuracy": 0.95})
        text = card.summary()
        assert "test-model" in text
        assert "v2.0" in text
        assert "text-generation" in text
        assert "toxicity=low" in text
        assert "Customer support" in text
        assert "accuracy=0.95" in text

    def test_summary_minimal(self):
        card = ModelCard("bare", "0.1")
        text = card.summary()
        assert "bare" in text
        assert "v0.1" in text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_set_metrics(self):
        card = ModelCard("m", "1.0")
        card.set_metrics({"accuracy": 0.95, "latency_ms": 12.5})
        exported = card.to_dict()
        assert exported["metrics"] == {"accuracy": 0.95, "latency_ms": 12.5}

    def test_metrics_empty_by_default(self):
        card = ModelCard("m", "1.0")
        assert card.to_dict()["metrics"] == {}


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

class TestTrainingData:
    def test_set_training_data(self):
        card = ModelCard("m", "1.0")
        card.set_training_data("Web corpus", size="10B tokens", sources=["web"])
        exported = card.to_dict()
        assert exported["training_data"]["description"] == "Web corpus"
        assert exported["training_data"]["size"] == "10B tokens"
        assert exported["training_data"]["sources"] == ["web"]

    def test_training_data_minimal(self):
        card = ModelCard("m", "1.0")
        card.set_training_data("Some data")
        exported = card.to_dict()
        assert exported["training_data"]["description"] == "Some data"
        assert exported["training_data"]["size"] == ""
        assert exported["training_data"]["sources"] == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_complete_card() -> ModelCard:
    """Build a fully populated card for reuse in tests."""
    card = ModelCard("test-model", "2.0", description="A test model")
    card.set_capabilities(["text-generation", "summarization"])
    card.set_intended_use("Customer support chatbot")
    card.set_safety_ratings({"toxicity": "low", "bias": "medium"})
    return card
