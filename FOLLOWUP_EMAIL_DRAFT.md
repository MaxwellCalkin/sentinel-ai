# Follow-up Email Draft (send Wed March 11 if no reply)

## To: chrismacleod@anthropic.com
## Subject: Re: Fellow applicant from Sept 2025 — built an AI safety guardrails SDK

Hi Christopher,

Quick follow-up on my message from Saturday. Since then I've added a few features that are specifically relevant to Claude Code:

- **`sentinel init`** — one command auto-configures Claude Code hooks and MCP server
- **`sentinel hook`** — works as a Claude Code PreToolUse hook, scanning every tool call for dangerous commands before execution
- Full Claude Code plugin structure ready for the marketplace

The core idea: as Claude becomes more agentic, safety scanning at the tool-call boundary becomes critical. Sentinel AI already does this with sub-millisecond latency and zero heavy dependencies.

Would love 15 minutes to show you a demo. Happy to work around your schedule.

Best,
Maxwell Calkin
https://github.com/MaxwellCalkin/sentinel-ai

---

## To: jblack@anthropic.com, vgupta@anthropic.com
## Subject: Re: Sentinel AI — safety guardrails SDK built for the Claude ecosystem

Hi James and Vishal,

Following up briefly — I've continued shipping since Saturday:

- Claude Code hooks integration (`sentinel hook` as PreToolUse safety layer)
- One-command setup: `sentinel init` configures hooks + MCP + policy
- 349 tests, 100% benchmark accuracy, ~0.05ms scan latency

The key value prop for Anthropic: production-ready agentic safety scanning that integrates natively with Claude Code, MCP, and the Claude SDK. As Claude gains more autonomy, this becomes table stakes.

Happy to put together a more detailed technical brief or do a live demo. What works best?

Best,
Maxwell Calkin
https://github.com/MaxwellCalkin/sentinel-ai
