# Follow-up Email Draft (send Wed March 11 if no reply)

## To: chrismacleod@anthropic.com
## Subject: Re: Fellow applicant from Sept 2025 — built an AI safety guardrails SDK

Hi Christopher,

Quick follow-up on my message from Saturday. I've been shipping fast — Sentinel AI is now at v0.6.0 with some features I think are particularly relevant to Anthropic:

**MCP Safety Proxy** — `sentinel mcp-proxy` wraps any MCP server with safety scanning. Intercepts all tool calls, blocks dangerous operations, auto-redacts PII in responses. This addresses the hook bypass issues in Claude Code (#21460 — hooks not enforced on subagent tool calls) by operating at the transport layer.

**Code Vulnerability Scanner** — Scans LLM-generated code for OWASP Top 10 vulnerabilities (SQL injection, XSS, command injection, hardcoded secrets, insecure deserialization). Works as a PostToolUse hook for Write/Edit tools.

**Claude Code attack detection** — Detects the exact vectors from CVE-2025-59536 and CVE-2026-21852 (poisoned repos via HTML comments, authority impersonation, API base URL override for credential exfiltration).

Current state: 9 scanners, 457 tests, 530-case benchmark at 100% accuracy, sub-millisecond latency, one dependency (`regex`). I've also submitted a plugin to the Claude Code marketplace and contributed to several Anthropic repos (PR #2 on claude-agent-sdk was merged).

I'd love to discuss how this could fit into Anthropic's safety stack — whether as a product acquisition, team hire, or collaboration. Happy to do a 15-minute demo.

Best,
Maxwell Calkin
https://github.com/MaxwellCalkin/sentinel-ai

---

## To: jblack@anthropic.com, vgupta@anthropic.com
## Subject: Re: Sentinel AI — safety guardrails SDK for the Claude ecosystem

Hi James and Vishal,

Following up from Saturday. Sentinel AI has evolved significantly — here's a quick snapshot:

**v0.6.0 highlights:**
- MCP Safety Proxy — transparent safety layer for any MCP server
- Code vulnerability scanner — OWASP detection in LLM-generated code
- Claude Code attack vector detection (CVE-2025-59536, CVE-2026-21852)
- 9 scanners, 457 tests, 530-case benchmark at 100% accuracy

**Anthropic ecosystem integration:**
- PR #2 on claude-agent-sdk — merged
- PRs open on claude-quickstarts, claude-code-security-review, knowledge-work-plugins
- Plugin submitted to Claude Code marketplace
- Active engagement on Claude Code security issues (PII redaction, hook bypass, marketplace safety)

The value prop: as Claude becomes more agentic (MCP tools, code generation, autonomous workflows), safety scanning at the tool-call boundary becomes critical infrastructure. Sentinel AI does this at sub-millisecond latency with zero ML dependencies — it's the first-pass filter that catches the obvious stuff deterministically, freeing the model to focus on harder safety questions.

I believe this is an acquisition opportunity — the technology, integrations, and my understanding of your safety needs all align. Would welcome a conversation about what makes sense.

Best,
Maxwell Calkin
https://github.com/MaxwellCalkin/sentinel-ai
