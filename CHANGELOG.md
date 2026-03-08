# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2026-03-08

### Added
- **SARIF v2.1.0 Output** — Generate SARIF format for GitHub Code Scanning, Azure DevOps, and other static analysis tools
  - `sentinel scan --format sarif` and `sentinel code-scan --format sarif` CLI support
  - `scan_result_to_sarif()` and `findings_to_sarif()` Python API
  - Maps RiskLevel to SARIF severity levels, Finding spans to SARIF regions
  - Includes rule deduplication and metadata propagation
- GitHub Action `upload-sarif` option for automatic GitHub Code Scanning integration
- 22 new SARIF tests

### Changed
- Test count: 536 → 558

## [0.7.1] - 2026-03-08

### Added
- **Streaming Safety Scanning** — `guarded_stream()` and `guarded_stream_async()` wrap the Anthropic streaming API with real-time output scanning via `StreamingGuard`. Blocks dangerous content mid-stream.
- **Anthropic SDK Wrapper Test Suite** — 30 tests covering `guarded_message`, `guarded_stream`, tool use scanning, PII redaction, and async variants
- Shared `_scan_inputs()` helper to reduce code duplication in wrapper

### Changed
- Test count: 481 → 511

## [0.7.0] - 2026-03-08

### Added
- **Obfuscation Detection Scanner** — Detects encoded/obfuscated attack payloads:
  - Base64-encoded malicious content (command injection, SQL injection, prompt override)
  - Hex-encoded commands (`\x` escape sequences, `0x` format)
  - ROT13-encoded dangerous instructions
  - Unicode escape sequences hiding attack payloads
  - Leetspeak variants of dangerous terms (e.g., `1gn0r3 1n5truct10n5`)
- 24 new tests for obfuscation detection

### Changed
- Scanner count: 9 → 10
- Test count: 457 → 481

## [0.6.0] - 2026-03-07

### Added
- **Code Vulnerability Scanner** — OWASP Top 10 detection in LLM-generated code:
  - SQL injection (f-strings, string concatenation, `.format()`)
  - Command injection (`shell=True`, `os.system`, `eval`)
  - XSS (`innerHTML`, `document.write`, unsafe Jinja2)
  - Path traversal, insecure deserialization, hardcoded secrets
  - Insecure cryptography (MD5, SHA1, AES-ECB)
  - SSRF (user-controlled URLs in requests)
- **MCP Safety Proxy** — `sentinel mcp-proxy` wraps any MCP server with safety scanning
  - Scans all tool arguments before forwarding
  - Auto-redacts PII in tool responses
  - Blocks dangerous operations at transport layer
- CLI commands: `sentinel code-scan`, `sentinel mcp-proxy`
- GitHub Action support for code scanning (`code-scan` input)
- SECURITY.md responsible disclosure policy
- 46 new tests (33 code scanner + 13 MCP proxy)

## [0.5.2] - 2026-03-06

### Added
- Claude Code attack vector detection (CVE-2025-59536, CVE-2026-21852)
- Multilingual prompt injection detection (12 languages)
- Live demo site at maxwellcalkin.github.io/sentinel-ai

## [0.5.0] - 2026-03-05

### Added
- MCP server with 7 safety tools
- Streaming safety scanner (`StreamingGuard`)
- Conversation-level guard
- Adversarial testing framework
- RSP risk report generator
- Policy engine with YAML/JSON configuration
- GitHub Actions CI/CD workflow
- 530-case benchmark suite

## [0.4.0] - 2026-03-04

### Added
- Tool-use scanner for MCP/function-calling safety
- Structured output validator
- Blocked terms scanner
- Claude Code hook integration (`sentinel hook`)

## [0.3.0] - 2026-03-03

### Added
- Hallucination detector (hedging language, confidence scoring)
- Toxicity scanner
- Harmful content scanner (weapons, drugs, self-harm, illegal activity)

## [0.2.0] - 2026-03-02

### Added
- PII scanner with auto-redaction (emails, SSNs, credit cards, phones, IP addresses)
- FastAPI server for REST API deployment
- Async scanning support

## [0.1.0] - 2026-03-01

### Added
- Initial release
- Prompt injection detection scanner
- Core orchestration engine (`SentinelGuard`)
- `ScanResult` with risk levels (none/low/medium/high/critical)
