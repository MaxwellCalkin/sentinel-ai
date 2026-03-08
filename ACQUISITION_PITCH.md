# Sentinel AI — Acquisition One-Pager

## What It Is

Sentinel AI is an open-source, real-time safety guardrails SDK for LLM applications. It scans all inputs/outputs for prompt injection, PII leaks, harmful content, hallucinations, toxicity, and dangerous tool calls — with sub-millisecond latency and zero heavy dependencies.

## Why It Matters for Anthropic

**1. Native Claude Ecosystem Integration (already built)**
- Claude Code hooks: `sentinel hook` as PreToolUse safety layer
- MCP server: 7 tools for Claude Desktop/Code
- Claude Code plugin: marketplace-ready
- Claude SDK middleware: drop-in wrapper for guarded API calls
- `sentinel init`: one-command setup for entire Claude toolchain

**2. Solves the Agentic Safety Problem**
As Claude becomes more agentic (tool use, computer use, multi-step tasks), safety scanning at the tool-call boundary becomes critical. Sentinel AI already does this:
- Scans every tool call for dangerous shell commands, credential access, data exfiltration
- Multi-turn conversation tracking detects gradual jailbreak escalation
- Structured output validation catches XSS/SQLi hidden in LLM-generated JSON
- Adversarial robustness testing with 10+ evasion technique variants
- **Multilingual injection detection** in 12 languages — critical for Claude's global user base

**3. Aligns with RSP Commitments**
Built-in RSP v3.0-aligned risk report generation. Threat domain assessments map directly to Anthropic's Responsible Scaling Policy risk categories.

## Technical Highlights

| Metric | Value |
|--------|-------|
| Scan latency | ~0.05ms average |
| Benchmark accuracy | 100% (500 cases) |
| Test coverage | 393 tests |
| Multilingual injection | 12 languages + cross-lingual |
| Core dependencies | 1 (`regex`) |
| Scanners | 8 pluggable modules |
| Lines of code | ~4,000 (Python) |

**Architecture**: Scanner protocol pattern — each scanner is a pluggable module with a `.scan()` method returning `list[Finding]`. SentinelGuard orchestrates all scanners and aggregates results. This makes it trivial to add new scanners (e.g., for new threat categories Anthropic identifies).

## What Anthropic Gets

1. **Production-ready safety scanning layer** that can be integrated into Claude API, Claude Code, Claude Desktop, and customer-facing safety products
2. **Agentic tool-use safety** — critical as Claude gains more autonomy
3. **Drop-in middleware** for all major LLM frameworks (OpenAI, LangChain, LlamaIndex)
4. **Red-teaming infrastructure** — adversarial variant generation for testing safety systems
5. **The developer** — Maxwell Calkin, previously "quite promising" Anthropic Fellows applicant, deep understanding of the safety problem space

## Open Source Strategy

Apache 2.0 licensed — Anthropic can:
- Integrate directly into Claude products
- Offer as a managed service alongside Claude API
- Use internally for safety testing/evaluation
- Continue developing as open-source to build ecosystem trust

## Links

- GitHub: https://github.com/MaxwellCalkin/sentinel-ai
- License: Apache 2.0
- Contact: mcalkinmusic@gmail.com
