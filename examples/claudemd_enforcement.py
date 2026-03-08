"""Example: Enforcing CLAUDE.md rules as deterministic checks.

CLAUDE.md rules are just suggestions that the model can ignore.
ClaudeMdEnforcer converts them into deterministic checks that
actually block violations before they execute.

This example shows:
1. Parsing CLAUDE.md to extract enforceable rules
2. Checking tool calls against those rules
3. Generating a guard policy from CLAUDE.md
4. Using the policy with SessionGuard for full session protection
"""

from sentinel import ClaudeMdEnforcer, EnforcementVerdict

# --- Example CLAUDE.md content ---
EXAMPLE_CLAUDEMD = """
# Project Rules

## Security
- Never use `rm -rf`
- Do not access `.env` or `.aws/credentials`
- Never run `git push --force`

## Development
- Do not modify files in `tests/` without asking
- Never use `npm publish` directly
- Avoid using `curl` to download from untrusted sources

Blocked commands: mkfs, dd, fdisk
"""

# --- 1. Parse CLAUDE.md ---
enforcer = ClaudeMdEnforcer.from_text(EXAMPLE_CLAUDEMD)
print(enforcer.summary())
print()

# --- 2. Check tool calls ---
checks = [
    ("bash", {"command": "rm -rf /tmp/data"}),
    ("bash", {"command": "git push --force origin main"}),
    ("bash", {"command": "ls -la"}),              # safe
    ("bash", {"command": "cat .env"}),
    ("read_file", {"path": ".aws/credentials"}),
    ("bash", {"command": "git status"}),           # safe
    ("bash", {"command": "mkfs /dev/sda1"}),
]

print("--- Tool Call Checks ---")
for tool, args in checks:
    verdict: EnforcementVerdict = enforcer.check(tool, args)
    status = "ALLOWED" if verdict.allowed else "BLOCKED"
    detail = ""
    if not verdict.allowed:
        detail = f" — {verdict.violated_rules[0]}"
    print(f"  [{status}] {tool}: {args.get('command', args.get('path', ''))}{detail}")

print()

# --- 3. Generate guard policy ---
policy_dict = enforcer.to_guard_policy_dict()
print("--- Generated Guard Policy ---")
print(f"  Block on: {policy_dict['block_on']}")
print(f"  Blocked commands: {policy_dict['blocked_commands']}")
print(f"  Sensitive paths: {policy_dict['sensitive_paths']}")
print(f"  Denied tools: {policy_dict['denied_tools']}")
print()

# --- 4. Use with SessionGuard ---
policy = enforcer.to_guard_policy()
guard = policy.create_guard(session_id="demo-session")

print("--- SessionGuard Integration ---")
verdict = guard.check("bash", {"command": "rm -rf /tmp/data"})
print(f"  guard.check('bash', 'rm -rf /tmp/data') -> allowed={verdict.allowed}")

verdict = guard.check("bash", {"command": "echo hello"})
print(f"  guard.check('bash', 'echo hello') -> allowed={verdict.allowed}")
print()

# --- 5. Hook setup instructions ---
print("--- Hook Setup ---")
print("To enforce these rules in Claude Code, run:")
print("  sentinel init")
print()
print("Or add to .claude/settings.json:")
print('  {"hooks": {"PreToolUse": [{"matcher": ".*",')
print('    "hooks": [{"type": "command", "command": "sentinel hook"}]}]}}')
print()
print("The hook automatically reads CLAUDE.md and enforces rules.")
