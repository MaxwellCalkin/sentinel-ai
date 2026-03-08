# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.5.x   | Yes                |
| < 0.5   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in Sentinel AI, please report it responsibly:

1. **Do NOT open a public issue.**
2. Email: [security@sentinel-ai.dev](mailto:security@sentinel-ai.dev) or use [GitHub Security Advisories](https://github.com/MaxwellCalkin/sentinel-ai/security/advisories/new).
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Impact assessment
   - Suggested fix (if any)

We aim to acknowledge reports within 48 hours and provide a fix within 7 days for critical issues.

## Scope

The following are in scope:
- Scanner bypass techniques (e.g., evasion patterns that bypass prompt injection detection)
- False negative gaps in PII detection
- Vulnerabilities in the API server (`sentinel serve`)
- Authentication/authorization bypass in the auth module
- Injection vulnerabilities in the CLI or MCP server

Out of scope:
- Denial of service via large input (we document no size limits)
- Issues in optional dependencies (FastAPI, uvicorn, etc.)

## Security Design

Sentinel AI is designed with security as a core principle:

- **No network calls**: Core scanning is entirely local — no data leaves your machine.
- **No persistent storage**: Scanned text is never written to disk or logged by default.
- **Minimal dependencies**: Only `regex` in the core library to minimize supply chain risk.
- **Pattern-based**: Deterministic behavior — no model inference, no side channels.
