"""Tests for the LLM API firewall proxy."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from sentinel.proxy import (
    ProxyConfig,
    _extract_user_text,
    _extract_response_text,
    _extract_tool_calls,
    create_proxy_app,
)
from sentinel.core import RiskLevel


class TestExtractUserText:
    def test_string_content(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        assert _extract_user_text(body) == "Hello"

    def test_list_content(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "World"},
                    ],
                }
            ]
        }
        assert "Hello" in _extract_user_text(body)
        assert "World" in _extract_user_text(body)

    def test_system_prompt(self):
        body = {"system": "Be helpful", "messages": []}
        assert "Be helpful" in _extract_user_text(body)

    def test_system_prompt_list(self):
        body = {
            "system": [{"type": "text", "text": "Be safe"}],
            "messages": [],
        }
        assert "Be safe" in _extract_user_text(body)

    def test_skips_assistant_messages(self):
        body = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ]
        }
        text = _extract_user_text(body)
        assert "Hi" in text
        assert "Bye" in text
        assert "Hello" not in text

    def test_empty_messages(self):
        assert _extract_user_text({"messages": []}) == ""
        assert _extract_user_text({}) == ""


class TestExtractResponseText:
    def test_text_content(self):
        body = {"content": [{"type": "text", "text": "Hello!"}]}
        assert _extract_response_text(body) == "Hello!"

    def test_multiple_blocks(self):
        body = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "tool_use", "name": "test"},
                {"type": "text", "text": "Part 2"},
            ]
        }
        text = _extract_response_text(body)
        assert "Part 1" in text
        assert "Part 2" in text

    def test_empty(self):
        assert _extract_response_text({}) == ""
        assert _extract_response_text({"content": []}) == ""


class TestExtractToolCalls:
    def test_extracts_tool_use(self):
        body = {
            "content": [
                {"type": "text", "text": "Let me help."},
                {"type": "tool_use", "name": "bash", "input": {"command": "ls"}},
            ]
        }
        tools = _extract_tool_calls(body)
        assert len(tools) == 1
        assert tools[0]["name"] == "bash"
        assert tools[0]["input"] == {"command": "ls"}

    def test_no_tools(self):
        body = {"content": [{"type": "text", "text": "Hello"}]}
        assert _extract_tool_calls(body) == []

    def test_empty(self):
        assert _extract_tool_calls({}) == []


class TestProxyConfig:
    def test_defaults(self):
        config = ProxyConfig()
        assert config.target_url == "https://api.anthropic.com"
        assert config.port == 8330
        assert config.scan_input is True
        assert config.scan_output is True
        assert config.block_threshold == RiskLevel.HIGH
        assert config.redact_pii is True

    def test_custom(self):
        config = ProxyConfig(
            target_url="https://api.openai.com",
            port=9000,
            scan_input=False,
            block_threshold=RiskLevel.CRITICAL,
        )
        assert config.target_url == "https://api.openai.com"
        assert config.port == 9000
        assert config.scan_input is False
        assert config.block_threshold == RiskLevel.CRITICAL


class TestProxyApp:
    def test_creates_app(self):
        app = create_proxy_app()
        assert app is not None
        assert app.title == "Sentinel AI Firewall"

    def test_creates_app_with_config(self):
        config = ProxyConfig(target_url="https://api.openai.com", port=9999)
        app = create_proxy_app(config=config)
        assert app is not None

    def test_health_endpoint_exists(self):
        app = create_proxy_app()
        routes = [r.path for r in app.routes]
        assert "/_sentinel/health" in routes

    def test_stats_endpoint_exists(self):
        app = create_proxy_app()
        routes = [r.path for r in app.routes]
        assert "/_sentinel/stats" in routes
