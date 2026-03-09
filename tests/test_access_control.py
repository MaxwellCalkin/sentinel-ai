"""Tests for access control."""

import pytest
from sentinel.access_control import AccessControl, AccessResult


class TestBasicAccess:
    def test_grant_permission(self):
        ac = AccessControl()
        ac.add_role("user", permissions=["chat", "summarize"])
        ac.assign_role("alice", "user")
        result = ac.check("alice", "chat")
        assert result.allowed

    def test_deny_permission(self):
        ac = AccessControl()
        ac.add_role("user", permissions=["chat"])
        ac.assign_role("alice", "user")
        result = ac.check("alice", "delete_data")
        assert not result.allowed

    def test_wildcard_permission(self):
        ac = AccessControl()
        ac.add_role("admin", permissions=["*"])
        ac.assign_role("bob", "admin")
        result = ac.check("bob", "anything")
        assert result.allowed

    def test_no_role_assigned(self):
        ac = AccessControl()
        result = ac.check("unknown_user", "chat")
        assert not result.allowed
        assert "No role" in result.reason


class TestRoleHierarchy:
    def test_parent_role(self):
        ac = AccessControl()
        ac.add_role("base", permissions=["chat"])
        ac.add_role("power_user", permissions=["summarize"], parent="base")
        ac.assign_role("alice", "power_user")
        assert ac.check("alice", "chat").allowed
        assert ac.check("alice", "summarize").allowed

    def test_no_parent_leak(self):
        ac = AccessControl()
        ac.add_role("base", permissions=["chat"])
        ac.add_role("limited", permissions=["read"])
        ac.assign_role("alice", "limited")
        assert not ac.check("alice", "chat").allowed


class TestPatternPermissions:
    def test_prefix_wildcard(self):
        ac = AccessControl()
        ac.add_role("reader", permissions=["data:*"])
        ac.assign_role("alice", "reader")
        assert ac.check("alice", "data:read").allowed
        assert ac.check("alice", "data:list").allowed
        assert not ac.check("alice", "admin:delete").allowed


class TestUserManagement:
    def test_assign_role(self):
        ac = AccessControl()
        ac.add_role("user")
        assert ac.assign_role("alice", "user")

    def test_assign_invalid_role(self):
        ac = AccessControl()
        assert not ac.assign_role("alice", "nonexistent")

    def test_revoke_role(self):
        ac = AccessControl()
        ac.add_role("user")
        ac.assign_role("alice", "user")
        assert ac.revoke_role("alice")
        assert not ac.check("alice", "chat").allowed

    def test_revoke_missing(self):
        ac = AccessControl()
        assert not ac.revoke_role("nonexistent")

    def test_list_users(self):
        ac = AccessControl()
        ac.add_role("admin")
        ac.add_role("user")
        ac.assign_role("alice", "admin")
        ac.assign_role("bob", "user")
        assert len(ac.list_users()) == 2
        assert len(ac.list_users(role="admin")) == 1

    def test_list_roles(self):
        ac = AccessControl()
        ac.add_role("admin")
        ac.add_role("user")
        assert len(ac.list_roles()) == 2


class TestGetPermissions:
    def test_direct_permissions(self):
        ac = AccessControl()
        ac.add_role("user", permissions=["chat", "read"])
        ac.assign_role("alice", "user")
        perms = ac.get_permissions("alice")
        assert "chat" in perms
        assert "read" in perms

    def test_inherited_permissions(self):
        ac = AccessControl()
        ac.add_role("base", permissions=["chat"])
        ac.add_role("power", permissions=["admin"], parent="base")
        ac.assign_role("alice", "power")
        perms = ac.get_permissions("alice")
        assert "chat" in perms
        assert "admin" in perms

    def test_no_permissions(self):
        ac = AccessControl()
        perms = ac.get_permissions("unknown")
        assert len(perms) == 0


class TestStructure:
    def test_result_structure(self):
        ac = AccessControl()
        ac.add_role("user", permissions=["chat"])
        ac.assign_role("alice", "user")
        result = ac.check("alice", "chat")
        assert isinstance(result, AccessResult)
        assert result.user == "alice"
        assert result.permission == "chat"
        assert isinstance(result.reason, str)
