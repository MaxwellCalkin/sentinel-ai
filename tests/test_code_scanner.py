"""Tests for the code vulnerability scanner."""

from sentinel.scanners.code_scanner import CodeScanner
from sentinel.core import RiskLevel


class TestSQLInjection:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_fstring_in_execute(self):
        code = '''cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'''
        findings = self.scanner.scan(code)
        assert len(findings) > 0
        assert any(f.category == "sql_injection" for f in findings)

    def test_string_concat_in_query(self):
        code = '''cursor.execute("SELECT * FROM users WHERE name = '" + request.args.get("name") + "'")'''
        findings = self.scanner.scan(code)
        assert any(f.category == "sql_injection" for f in findings)

    def test_parameterized_query_safe(self):
        code = '''cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))'''
        findings = self.scanner.scan(code)
        sql_findings = [f for f in findings if f.category == "sql_injection"]
        assert len(sql_findings) == 0

    def test_format_in_query(self):
        code = '''db.query("SELECT * FROM users WHERE id = {}".format(user_input))'''
        findings = self.scanner.scan(code)
        assert any(f.category == "sql_injection" for f in findings)


class TestCommandInjection:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_subprocess_shell_true(self):
        code = '''subprocess.call(cmd, shell=True)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "command_injection" for f in findings)

    def test_subprocess_shell_false_safe(self):
        code = '''subprocess.run(["ls", "-la"], shell=False)'''
        findings = self.scanner.scan(code)
        cmd_findings = [f for f in findings if f.category == "command_injection"]
        assert len(cmd_findings) == 0

    def test_os_system_fstring(self):
        code = '''os.system(f"ping {user_input}")'''
        findings = self.scanner.scan(code)
        assert any(f.category == "command_injection" for f in findings)

    def test_js_exec_with_user_input(self):
        code = '''child_process.exec("ls " + req.params.dir)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "command_injection" for f in findings)


class TestXSS:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_innerhtml_with_user_input(self):
        code = '''element.innerHTML = request.data'''
        findings = self.scanner.scan(code)
        assert any(f.category == "xss" for f in findings)

    def test_jinja_safe_filter(self):
        code = '''{{ user_content | safe }}'''
        findings = self.scanner.scan(code)
        assert any(f.category == "xss" for f in findings)

    def test_markup_with_request(self):
        code = '''Markup(request.form["bio"])'''
        findings = self.scanner.scan(code)
        assert any(f.category == "xss" for f in findings)

    def test_textcontent_safe(self):
        code = '''element.textContent = userInput'''
        findings = self.scanner.scan(code)
        xss_findings = [f for f in findings if f.category == "xss"]
        assert len(xss_findings) == 0


class TestPathTraversal:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_open_with_user_path(self):
        code = '''open(os.path.join(base, request.args["filename"]))'''
        findings = self.scanner.scan(code)
        assert any(f.category == "path_traversal" for f in findings)

    def test_open_static_path_safe(self):
        code = '''open("config.json")'''
        findings = self.scanner.scan(code)
        path_findings = [f for f in findings if f.category == "path_traversal"]
        assert len(path_findings) == 0


class TestDeserialization:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_pickle_load(self):
        code = '''data = pickle.load(f)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "insecure_deserialization" for f in findings)

    def test_yaml_load_without_safe(self):
        code = '''config = yaml.load(f)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "insecure_deserialization" for f in findings)

    def test_yaml_safe_load_ok(self):
        code = '''config = yaml.safe_load(f)'''
        findings = self.scanner.scan(code)
        deser_findings = [f for f in findings if f.category == "insecure_deserialization"]
        assert len(deser_findings) == 0

    def test_eval_with_dynamic(self):
        code = '''result = eval(user_expression)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "insecure_deserialization" for f in findings)

    def test_eval_string_literal_safe(self):
        code = '''result = eval("2 + 2")'''
        findings = self.scanner.scan(code)
        deser_findings = [f for f in findings if f.category == "insecure_deserialization"]
        assert len(deser_findings) == 0


class TestHardcodedSecrets:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_hardcoded_password(self):
        code = '''password = "SuperSecret123!"'''
        findings = self.scanner.scan(code)
        assert any(f.category == "hardcoded_secret" for f in findings)

    def test_hardcoded_api_key(self):
        code = '''api_key = "sk-1234567890abcdefghij"'''
        findings = self.scanner.scan(code)
        assert any(f.category == "hardcoded_secret" for f in findings)

    def test_aws_key(self):
        code = '''AWS_KEY = "AKIAIOSFODNN7EXAMPLE"'''
        findings = self.scanner.scan(code)
        assert any(f.category == "hardcoded_secret" for f in findings)

    def test_env_var_safe(self):
        code = '''password = os.environ.get("DB_PASSWORD")'''
        findings = self.scanner.scan(code)
        secret_findings = [f for f in findings if f.category == "hardcoded_secret"]
        assert len(secret_findings) == 0


class TestInsecureCrypto:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_md5_usage(self):
        code = '''digest = hashlib.md5(data)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "insecure_crypto" for f in findings)

    def test_sha256_safe(self):
        code = '''digest = hashlib.sha256(data)'''
        findings = self.scanner.scan(code)
        crypto_findings = [f for f in findings if f.category == "insecure_crypto"]
        assert len(crypto_findings) == 0

    def test_aes_ecb_mode(self):
        code = '''cipher = AES.new(key, AES.MODE_ECB)'''
        findings = self.scanner.scan(code)
        assert any(f.category == "insecure_crypto" for f in findings)


class TestSSRF:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_requests_with_user_url(self):
        code = '''response = requests.get(request.args["url"])'''
        findings = self.scanner.scan(code)
        assert any(f.category == "ssrf" for f in findings)

    def test_requests_static_url_safe(self):
        code = '''response = requests.get("https://api.example.com/data")'''
        findings = self.scanner.scan(code)
        ssrf_findings = [f for f in findings if f.category == "ssrf"]
        assert len(ssrf_findings) == 0


class TestToolOutput:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_scan_write_tool(self):
        output = {
            "file_path": "app.py",
            "content": '''cursor.execute(f"SELECT * FROM users WHERE id = {uid}")''',
        }
        findings = self.scanner.scan_tool_output("Write", output)
        assert len(findings) > 0

    def test_scan_edit_tool(self):
        output = {
            "file_path": "server.js",
            "new_string": '''child_process.exec("rm " + req.body.file)''',
        }
        findings = self.scanner.scan_tool_output("Edit", output)
        assert len(findings) > 0

    def test_ignores_non_write_tools(self):
        output = {"command": "ls -la"}
        findings = self.scanner.scan_tool_output("Bash", output)
        assert len(findings) == 0

    def test_empty_content(self):
        output = {"file_path": "app.py", "content": ""}
        findings = self.scanner.scan_tool_output("Write", output)
        assert len(findings) == 0

    def test_line_numbers(self):
        code = '''import os
import sys

cursor.execute(f"SELECT * FROM users WHERE id = {uid}")
'''
        findings = self.scanner.scan(code)
        assert findings[0].metadata["line"] == 4
