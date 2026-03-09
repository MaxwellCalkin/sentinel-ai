"""Context isolation for multi-tenant LLM applications.

Enforce tenant boundaries in shared LLM deployments. Prevent
cross-tenant data leakage by isolating system prompts,
conversation history, and metadata.

Usage:
    from sentinel.context_isolation import ContextIsolation

    iso = ContextIsolation()
    iso.create_tenant("tenant_a", system_prompt="You help tenant A only.")
    result = iso.validate_message("tenant_a", "What did tenant B say?")
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tenant:
    """A tenant context."""
    tenant_id: str
    system_prompt: str = ""
    allowed_topics: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_history: int = 100


@dataclass
class IsolationResult:
    """Result of isolation validation."""
    allowed: bool
    tenant_id: str
    violations: list[str]
    cross_tenant_refs: list[str]


@dataclass
class IsolationStats:
    """Stats for isolation enforcement."""
    total_tenants: int
    total_validations: int
    total_violations: int
    violation_rate: float


class ContextIsolation:
    """Multi-tenant context isolation.

    Create isolated tenant contexts and validate messages
    to prevent cross-tenant data leakage.
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Args:
            strict: If True, block any cross-tenant references.
        """
        self._strict = strict
        self._tenants: dict[str, Tenant] = {}
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._validations = 0
        self._violations = 0

    def create_tenant(
        self,
        tenant_id: str,
        system_prompt: str = "",
        allowed_topics: list[str] | None = None,
        blocked_patterns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        max_history: int = 100,
    ) -> Tenant:
        """Create a tenant context.

        Args:
            tenant_id: Unique tenant identifier.
            system_prompt: Tenant-specific system prompt.
            allowed_topics: Allowed conversation topics.
            blocked_patterns: Regex patterns to block.
            metadata: Tenant metadata.
            max_history: Max conversation history length.
        """
        tenant = Tenant(
            tenant_id=tenant_id,
            system_prompt=system_prompt,
            allowed_topics=allowed_topics or [],
            blocked_patterns=blocked_patterns or [],
            metadata=metadata or {},
            max_history=max_history,
        )
        self._tenants[tenant_id] = tenant
        self._histories[tenant_id] = []
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def delete_tenant(self, tenant_id: str) -> bool:
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            self._histories.pop(tenant_id, None)
            return True
        return False

    def validate_message(
        self,
        tenant_id: str,
        message: str,
    ) -> IsolationResult:
        """Validate a message for tenant isolation.

        Checks for:
        - Cross-tenant references (mentions of other tenant IDs)
        - Blocked patterns
        - System prompt extraction attempts

        Args:
            tenant_id: Current tenant.
            message: Message to validate.

        Returns:
            IsolationResult with violations.
        """
        self._validations += 1
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return IsolationResult(
                allowed=False,
                tenant_id=tenant_id,
                violations=["Unknown tenant"],
                cross_tenant_refs=[],
            )

        violations: list[str] = []
        cross_refs: list[str] = []
        msg_lower = message.lower()

        # Check cross-tenant references
        if self._strict:
            for other_id in self._tenants:
                if other_id != tenant_id and other_id.lower() in msg_lower:
                    cross_refs.append(other_id)
                    violations.append(f"Cross-tenant reference to '{other_id}'")

        # Check blocked patterns
        for pattern in tenant.blocked_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                violations.append(f"Blocked pattern matched: {pattern}")

        # Check for system prompt extraction
        extraction_signals = [
            "system prompt", "system message", "your instructions",
            "your rules", "your guidelines", "reveal your",
        ]
        for signal in extraction_signals:
            if signal in msg_lower:
                violations.append(f"System prompt extraction attempt: '{signal}'")
                break

        if violations:
            self._violations += 1

        return IsolationResult(
            allowed=len(violations) == 0,
            tenant_id=tenant_id,
            violations=violations,
            cross_tenant_refs=cross_refs,
        )

    def add_to_history(
        self,
        tenant_id: str,
        role: str,
        content: str,
    ) -> bool:
        """Add a message to tenant's conversation history.

        Returns:
            True if added, False if tenant not found.
        """
        if tenant_id not in self._tenants:
            return False

        tenant = self._tenants[tenant_id]
        history = self._histories[tenant_id]
        history.append({"role": role, "content": content})

        # Trim to max history
        while len(history) > tenant.max_history:
            history.pop(0)

        return True

    def get_history(self, tenant_id: str) -> list[dict[str, str]]:
        """Get conversation history for tenant."""
        return list(self._histories.get(tenant_id, []))

    def clear_history(self, tenant_id: str) -> bool:
        if tenant_id in self._histories:
            self._histories[tenant_id] = []
            return True
        return False

    def build_context(self, tenant_id: str) -> list[dict[str, str]]:
        """Build full context (system prompt + history) for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return []

        messages: list[dict[str, str]] = []
        if tenant.system_prompt:
            messages.append({"role": "system", "content": tenant.system_prompt})
        messages.extend(self.get_history(tenant_id))
        return messages

    @property
    def tenant_count(self) -> int:
        return len(self._tenants)

    def stats(self) -> IsolationStats:
        return IsolationStats(
            total_tenants=len(self._tenants),
            total_validations=self._validations,
            total_violations=self._violations,
            violation_rate=(
                self._violations / self._validations
                if self._validations > 0 else 0.0
            ),
        )
