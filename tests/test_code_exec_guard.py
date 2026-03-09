"""Tests for code execution guard."""

import pytest
from sentinel.code_exec_guard import CodeExecGuard, CodeExecResult, CodeThreat


# ---------------------------------------------------------------------------
# Safe code
# ---------------------------------------------------------------------------

class TestSafeCode:
    def test_simple_math(self):
        g = CodeExecGuard()
        result = g.check("x = 1 + 2\nprint(x)")
        assert result.safe
        assert len(result.threats) == 0
        assert result.risk_score == 0.0

    def test_list_operations(self):
        g = CodeExecGuard()
        result = g.check("data = [1, 2, 3]\nresult = sum(data)")
        assert result.safe

    def test_string_operations(self):
        g = CodeExecGuard()
        result = g.check("name = 'hello'.upper()\nprint(name)")
        assert result.safe

    def test_safe_imports(self):
        g = CodeExecGuard()
        result = g.check("import math\nimport json\nprint(math.pi)")
        assert result.safe


# ---------------------------------------------------------------------------
# Dangerous imports
# ---------------------------------------------------------------------------

class TestDangerousImports:
    def test_subprocess(self):
        g = CodeExecGuard()
        result = g.check("import subprocess\nsubprocess.run(['ls'])")
        assert not result.safe
        assert any(t.category == "dangerous_import" for t in result.threats)

    def test_os_system(self):
        g = CodeExecGuard()
        result = g.check("import os\nos.system('ls')")
        assert not result.safe

    def test_shutil(self):
        g = CodeExecGuard()
        result = g.check("import shutil")
        assert any(t.severity == "critical" for t in result.threats)

    def test_socket(self):
        g = CodeExecGuard()
        result = g.check("import socket")
        assert any(t.category == "dangerous_import" for t in result.threats)

    def test_allowed_import(self):
        g = CodeExecGuard(allow_imports=["subprocess"])
        result = g.check("import subprocess")
        # subprocess import allowed, but subprocess.run() still flagged
        import_threats = [t for t in result.threats if t.category == "dangerous_import"]
        assert len(import_threats) == 0

    def test_from_import(self):
        g = CodeExecGuard()
        result = g.check("from subprocess import run")
        assert any(t.category == "dangerous_import" for t in result.threats)


# ---------------------------------------------------------------------------
# Dangerous calls
# ---------------------------------------------------------------------------

class TestDangerousCalls:
    def test_eval(self):
        g = CodeExecGuard()
        result = g.check("result = eval('1+1')")
        assert not result.safe
        assert any("eval" in t.description for t in result.threats)

    def test_exec(self):
        g = CodeExecGuard()
        result = g.check("exec('print(1)')")
        assert not result.safe

    def test_os_remove(self):
        g = CodeExecGuard()
        result = g.check("import os\nos.remove('/tmp/file')")
        assert not result.safe

    def test_file_write(self):
        g = CodeExecGuard()
        result = g.check("f = open('file.txt', 'w')")
        assert any("write" in t.description.lower() for t in result.threats)

    def test_sys_exit(self):
        g = CodeExecGuard()
        result = g.check("import sys\nsys.exit(1)")
        assert any("exit" in t.description.lower() for t in result.threats)


# ---------------------------------------------------------------------------
# Resource exhaustion
# ---------------------------------------------------------------------------

class TestResourceExhaustion:
    def test_infinite_loop(self):
        g = CodeExecGuard()
        result = g.check("while True:\n    pass")
        assert any(t.category == "resource_exhaustion" for t in result.threats)

    def test_large_range(self):
        g = CodeExecGuard()
        result = g.check("for i in range(100000000):\n    pass")
        assert any(t.category == "resource_exhaustion" for t in result.threats)


# ---------------------------------------------------------------------------
# Network allowance
# ---------------------------------------------------------------------------

class TestNetworkAllow:
    def test_network_blocked_by_default(self):
        g = CodeExecGuard()
        result = g.check("import requests")
        assert any(t.category == "dangerous_import" for t in result.threats)

    def test_network_allowed(self):
        g = CodeExecGuard(allow_network=True)
        result = g.check("import requests")
        import_threats = [t for t in result.threats if t.category == "dangerous_import"]
        assert len(import_threats) == 0


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------

class TestCustomRules:
    def test_custom_pattern(self):
        g = CodeExecGuard(custom_rules=[
            (r'password\s*=', "Password hardcoded", "high"),
        ])
        result = g.check("password = 'secret123'")
        assert any(t.category == "custom" for t in result.threats)


# ---------------------------------------------------------------------------
# Generic language
# ---------------------------------------------------------------------------

class TestGenericLanguage:
    def test_generic_system_call(self):
        g = CodeExecGuard()
        result = g.check("system('rm -rf /')", language="c")
        assert not result.safe

    def test_generic_safe(self):
        g = CodeExecGuard()
        result = g.check("int x = 42;", language="c")
        assert result.safe


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_result_structure(self):
        g = CodeExecGuard()
        result = g.check("x = 1")
        assert isinstance(result, CodeExecResult)
        assert result.language == "python"
        assert result.lines_analyzed == 1

    def test_threat_line_numbers(self):
        g = CodeExecGuard()
        code = "x = 1\nimport subprocess\ny = 2"
        result = g.check(code)
        import_threats = [t for t in result.threats if t.category == "dangerous_import"]
        assert import_threats[0].line == 2

    def test_critical_threats_property(self):
        g = CodeExecGuard()
        result = g.check("eval('x')")
        assert len(result.critical_threats) > 0

    def test_risk_score_range(self):
        g = CodeExecGuard()
        safe = g.check("x = 1")
        dangerous = g.check("import subprocess\nos.system('rm')\neval('x')")
        assert safe.risk_score == 0.0
        assert dangerous.risk_score > 0.5
        assert dangerous.risk_score <= 1.0
