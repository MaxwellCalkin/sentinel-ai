"""Example: Generate SARIF output for GitHub Code Scanning.

SARIF (Static Analysis Results Interchange Format) is the industry standard
for static analysis results. Sentinel AI generates SARIF v2.1.0 output that
integrates with GitHub Code Scanning, Azure DevOps, and other tools.
"""

from sentinel import SentinelGuard
from sentinel.scanners.code_scanner import CodeScanner
from sentinel.sarif import findings_to_sarif, scan_result_to_sarif, sarif_to_json


# Example 1: Scan text and generate SARIF
guard = SentinelGuard.default()
result = guard.scan("My SSN is 123-45-6789 and email is alice@corp.com")

sarif = scan_result_to_sarif(result, artifact_uri="user_input.txt")
print("=== Text scan SARIF ===")
print(sarif_to_json(sarif))
print()

# Example 2: Scan code and generate SARIF
scanner = CodeScanner()
findings = scanner.scan(
    '''
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
os.system(f"ping {host}")
password = "SuperSecret123!"
''',
    filename="app.py",
)

sarif = findings_to_sarif(findings, artifact_uri="app.py")
print("=== Code scan SARIF ===")
print(sarif_to_json(sarif))
print()

# Example 3: CLI usage
print("=== CLI commands ===")
print("sentinel scan 'text to scan' --format sarif > results.sarif")
print("sentinel code-scan --file app.py --format sarif > results.sarif")
