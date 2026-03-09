"""Role-based access control for LLM features.

Control which users/roles can access specific LLM features,
tools, and models. Support role hierarchies and permission
inheritance.

Usage:
    from sentinel.access_control import AccessControl

    ac = AccessControl()
    ac.add_role("admin", permissions=["*"])
    ac.add_role("user", permissions=["chat", "summarize"])
    ac.assign_role("alice", "admin")
    print(ac.check("alice", "delete_data"))  # True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Role:
    """A role with permissions."""
    name: str
    permissions: set[str]
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessResult:
    """Result of access check."""
    allowed: bool
    user: str
    permission: str
    role: str
    reason: str


class AccessControl:
    """Role-based access control for LLM features."""

    def __init__(self, default_deny: bool = True) -> None:
        self._roles: dict[str, Role] = {}
        self._user_roles: dict[str, str] = {}
        self._default_deny = default_deny

    def add_role(
        self,
        name: str,
        permissions: list[str] | None = None,
        parent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a role with permissions."""
        self._roles[name] = Role(
            name=name,
            permissions=set(permissions or []),
            parent=parent,
            metadata=metadata or {},
        )

    def assign_role(self, user: str, role: str) -> bool:
        """Assign a role to a user."""
        if role not in self._roles:
            return False
        self._user_roles[user] = role
        return True

    def check(self, user: str, permission: str) -> AccessResult:
        """Check if user has a specific permission."""
        role_name = self._user_roles.get(user)
        if not role_name or role_name not in self._roles:
            return AccessResult(
                allowed=not self._default_deny,
                user=user, permission=permission,
                role="none", reason="No role assigned",
            )

        # Check permission through role hierarchy
        allowed, resolved_role = self._check_role(role_name, permission)

        reason = f"Granted by role '{resolved_role}'" if allowed else f"Not in role '{role_name}' permissions"

        return AccessResult(
            allowed=allowed, user=user,
            permission=permission, role=role_name,
            reason=reason,
        )

    def get_permissions(self, user: str) -> set[str]:
        """Get all permissions for a user (including inherited)."""
        role_name = self._user_roles.get(user)
        if not role_name:
            return set()
        return self._collect_permissions(role_name)

    def revoke_role(self, user: str) -> bool:
        """Remove role assignment from user."""
        if user in self._user_roles:
            del self._user_roles[user]
            return True
        return False

    def list_roles(self) -> list[str]:
        """List all role names."""
        return list(self._roles.keys())

    def list_users(self, role: str | None = None) -> list[str]:
        """List users, optionally filtered by role."""
        if role:
            return [u for u, r in self._user_roles.items() if r == role]
        return list(self._user_roles.keys())

    def _check_role(self, role_name: str, permission: str) -> tuple[bool, str]:
        """Check permission through role hierarchy."""
        role = self._roles.get(role_name)
        if not role:
            return False, role_name

        # Wildcard check
        if "*" in role.permissions:
            return True, role_name

        # Direct match
        if permission in role.permissions:
            return True, role_name

        # Pattern match (e.g., "chat:*" matches "chat:read")
        for perm in role.permissions:
            if perm.endswith(":*"):
                prefix = perm[:-2]
                if permission.startswith(prefix + ":"):
                    return True, role_name

        # Check parent role
        if role.parent and role.parent in self._roles:
            return self._check_role(role.parent, permission)

        return False, role_name

    def _collect_permissions(self, role_name: str) -> set[str]:
        """Collect all permissions including inherited."""
        role = self._roles.get(role_name)
        if not role:
            return set()
        perms = set(role.permissions)
        if role.parent and role.parent in self._roles:
            perms |= self._collect_permissions(role.parent)
        return perms
