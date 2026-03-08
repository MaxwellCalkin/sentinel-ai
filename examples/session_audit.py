"""Session Audit Trail — example usage.

Demonstrates tamper-evident audit logging for agentic AI sessions.
"""

from sentinel.session_audit import SessionAudit


def main():
    # Create an audit trail for a session
    audit = SessionAudit(
        session_id="demo-session-001",
        user_id="developer@company.com",
        agent_id="claude-agent-1",
        model="claude-4",
    )

    # --- Scenario: Normal development workflow ---
    print("=== Normal Development ===")
    audit.log_tool_call("bash", {"command": "ls src/"}, risk="none")
    audit.log_tool_call("read_file", {"path": "src/main.py"}, risk="none")
    audit.log_tool_call("bash", {"command": "pytest tests/"}, risk="none")

    # --- Scenario: Credential access detected ---
    print("\n=== Credential Access ===")
    audit.log_tool_call(
        "read_file",
        {"path": ".env"},
        risk="high",
        findings=["credential_access"],
    )

    # --- Scenario: Blocked destructive command ---
    print("\n=== Blocked Action ===")
    audit.log_blocked(
        "bash",
        {"command": "rm -rf /"},
        reason="destructive_command",
        risk="critical",
        findings=["destructive_command"],
    )

    # --- Scenario: Anomaly detected ---
    print("\n=== Anomaly ===")
    audit.log_anomaly("Runaway loop: 50 tool calls in 30 seconds", risk="medium")

    # --- Scenario: Attack chain detected ---
    print("\n=== Attack Chain ===")
    audit.log_chain(
        "recon_credential_exfiltrate",
        "Reconnaissance → Credential Access → Exfiltration",
        stages=["reconnaissance", "credential_access", "exfiltration"],
    )

    # --- Verify integrity ---
    print(f"\nIntegrity verified: {audit.verify_integrity()}")
    print(f"Total entries: {audit.total_entries}")

    # --- Export report ---
    report = audit.export()
    print(f"\n=== Export Summary ===")
    print(f"Session: {report['session_id']}")
    print(f"User: {report['user_id']}")
    print(f"Duration: {report['duration_seconds']}s")
    print(f"Total calls: {report['summary']['total_calls']}")
    print(f"Tool calls: {report['summary']['tool_calls']}")
    print(f"Blocked: {report['summary']['blocked_calls']}")
    print(f"Anomalies: {report['summary']['anomalies_detected']}")
    print(f"Chains: {report['summary']['chains_detected']}")
    print(f"Max risk: {report['summary']['risk_level']}")
    print(f"Tool counts: {report['summary']['tool_counts']}")
    print(f"Finding categories: {report['summary']['finding_categories']}")

    # --- Risk timeline ---
    print(f"\n=== Risk Timeline ===")
    for point in audit.risk_timeline():
        print(f"  [{point['risk']:>8}] {point['event_type']}: {point['tool_name']}")

    # --- JSON export (for SIEM) ---
    print(f"\n=== JSON Export (first 200 chars) ===")
    print(audit.export_json()[:200] + "...")


if __name__ == "__main__":
    main()
