"""Structured model deployment metadata with safety characteristics.

Inspired by Google's Model Cards for Model Reporting. Provides a
standardized way to document model capabilities, limitations,
safety ratings, and intended use before deployment.

Usage:
    from sentinel.model_card import ModelCard, CardValidation

    card = ModelCard("my-llm", "1.0", description="General assistant")
    card.set_capabilities(["text-generation", "summarization"])
    card.set_intended_use("Customer support chatbot")
    card.set_safety_ratings({"toxicity": "low", "bias": "medium"})

    validation = card.validate()
    print(validation.valid)  # True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_SAFETY_LEVELS = ("low", "medium", "high", "critical")


@dataclass
class CardValidation:
    """Result of model card validation."""

    valid: bool
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SafetyRating:
    """A single safety rating for a category."""

    category: str
    level: str  # low, medium, high, critical

    def __post_init__(self) -> None:
        if self.level not in VALID_SAFETY_LEVELS:
            raise ValueError(
                f"Invalid safety level '{self.level}'. "
                f"Must be one of: {', '.join(VALID_SAFETY_LEVELS)}"
            )


@dataclass
class IntendedUse:
    """Intended use declaration for a model."""

    primary: str
    out_of_scope: list[str] = field(default_factory=list)


@dataclass
class TrainingData:
    """Training data metadata."""

    description: str
    size: str = ""
    sources: list[str] = field(default_factory=list)


class ModelCard:
    """Structured model deployment metadata.

    Documents model capabilities, limitations, safety ratings,
    and intended use in a standardized format suitable for
    compliance review and deployment gating.
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str = "",
    ) -> None:
        """Create a model card.

        Args:
            name: Model name.
            version: Model version string.
            description: Optional human-readable description.
        """
        self.name = name
        self.version = version
        self.description = description
        self._capabilities: list[str] = []
        self._limitations: list[str] = []
        self._safety_ratings: list[SafetyRating] = []
        self._intended_use: IntendedUse | None = None
        self._training_data: TrainingData | None = None
        self._metrics: dict[str, float] = {}

    def set_capabilities(self, capabilities: list[str]) -> None:
        """Set model capabilities.

        Args:
            capabilities: List of capability strings.
        """
        self._capabilities = list(capabilities)

    def set_limitations(self, limitations: list[str]) -> None:
        """Set known model limitations.

        Args:
            limitations: List of limitation descriptions.
        """
        self._limitations = list(limitations)

    def set_safety_ratings(self, ratings: dict[str, str]) -> None:
        """Set safety ratings by category.

        Args:
            ratings: Mapping of category to level (low/medium/high/critical).

        Raises:
            ValueError: If a safety level is not recognized.
        """
        self._safety_ratings = [
            SafetyRating(category=category, level=level)
            for category, level in ratings.items()
        ]

    def set_intended_use(
        self,
        primary: str,
        out_of_scope: list[str] | None = None,
    ) -> None:
        """Set intended use declaration.

        Args:
            primary: Primary intended use case.
            out_of_scope: Use cases explicitly out of scope.
        """
        self._intended_use = IntendedUse(
            primary=primary,
            out_of_scope=out_of_scope or [],
        )

    def set_training_data(
        self,
        description: str,
        size: str = "",
        sources: list[str] | None = None,
    ) -> None:
        """Set training data metadata.

        Args:
            description: Description of the training data.
            size: Human-readable size (e.g. "10B tokens").
            sources: List of data source names.
        """
        self._training_data = TrainingData(
            description=description,
            size=size,
            sources=sources or [],
        )

    def set_metrics(self, metrics: dict[str, float]) -> None:
        """Set performance metrics.

        Args:
            metrics: Mapping of metric name to value.
        """
        self._metrics = dict(metrics)

    def validate(self) -> CardValidation:
        """Check if the card meets minimum documentation requirements.

        Requires: name, version, at least one capability, intended use.
        Warns if no safety ratings or limitations are set.

        Returns:
            CardValidation with validity status and any issues.
        """
        missing_fields: list[str] = []
        warnings: list[str] = []

        if not self.name:
            missing_fields.append("name")
        if not self.version:
            missing_fields.append("version")
        if not self._capabilities:
            missing_fields.append("capabilities")
        if self._intended_use is None:
            missing_fields.append("intended_use")

        if not self._safety_ratings:
            warnings.append("No safety ratings specified")
        if not self._limitations:
            warnings.append("No limitations documented")

        return CardValidation(
            valid=len(missing_fields) == 0,
            missing_fields=missing_fields,
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export model card as a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self._capabilities,
            "limitations": self._limitations,
            "safety_ratings": {
                r.category: r.level for r in self._safety_ratings
            },
            "metrics": self._metrics,
        }

        if self._intended_use is not None:
            data["intended_use"] = {
                "primary": self._intended_use.primary,
                "out_of_scope": self._intended_use.out_of_scope,
            }
        else:
            data["intended_use"] = None

        if self._training_data is not None:
            data["training_data"] = {
                "description": self._training_data.description,
                "size": self._training_data.size,
                "sources": self._training_data.sources,
            }
        else:
            data["training_data"] = None

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        """Create a ModelCard from a dictionary.

        Args:
            data: Dictionary with model card fields.

        Returns:
            Populated ModelCard instance.
        """
        card = cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
        )

        if data.get("capabilities"):
            card.set_capabilities(data["capabilities"])

        if data.get("limitations"):
            card.set_limitations(data["limitations"])

        if data.get("safety_ratings"):
            card.set_safety_ratings(data["safety_ratings"])

        if data.get("intended_use"):
            intended = data["intended_use"]
            card.set_intended_use(
                primary=intended["primary"],
                out_of_scope=intended.get("out_of_scope"),
            )

        if data.get("training_data"):
            td = data["training_data"]
            card.set_training_data(
                description=td["description"],
                size=td.get("size", ""),
                sources=td.get("sources"),
            )

        if data.get("metrics"):
            card.set_metrics(data["metrics"])

        return card

    def summary(self) -> str:
        """Generate a human-readable summary string.

        Returns:
            Multi-line summary of the model card.
        """
        lines = [
            f"Model: {self.name} v{self.version}",
        ]

        if self.description:
            lines.append(f"Description: {self.description}")

        if self._capabilities:
            lines.append(f"Capabilities: {', '.join(self._capabilities)}")

        if self._limitations:
            lines.append(f"Limitations: {', '.join(self._limitations)}")

        if self._safety_ratings:
            rating_parts = [
                f"{r.category}={r.level}" for r in self._safety_ratings
            ]
            lines.append(f"Safety: {', '.join(rating_parts)}")

        if self._intended_use is not None:
            lines.append(f"Intended use: {self._intended_use.primary}")

        if self._metrics:
            metric_parts = [
                f"{name}={value}" for name, value in self._metrics.items()
            ]
            lines.append(f"Metrics: {', '.join(metric_parts)}")

        return "\n".join(lines)
