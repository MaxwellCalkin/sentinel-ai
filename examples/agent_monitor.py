"""Example: Monitor agentic AI sessions for safety anomalies.

Tracks tool-use patterns and detects destructive commands, data exfiltration,
credential access, runaway loops, and more.
"""

from sentinel.agent_monitor import AgentMonitor

monitor = AgentMonitor(
    max_repeat_calls=5,       # Alert after 5 identical consecutive calls
    write_spike_threshold=20,  # Alert on 20+ writes per minute
    window_seconds=60,         # Sliding window for rate calculations
)

# Simulate an agentic session
print("=== Agent Session Monitor ===\n")

# Normal operations
v = monitor.record("read_file", {"path": "src/main.py"})
print(f"1. read_file src/main.py → risk={v.risk.value}, alert={v.alert}")

v = monitor.record("bash", {"command": "python -m pytest tests/"})
print(f"2. bash pytest → risk={v.risk.value}, alert={v.alert}")

v = monitor.record("write_file", {"path": "src/main.py", "content": "# updated"})
print(f"3. write_file → risk={v.risk.value}, alert={v.alert}")

# Dangerous operation detected!
v = monitor.record("bash", {"command": "rm -rf /tmp/project"})
print(f"4. bash rm -rf → risk={v.risk.value}, alert={v.alert}")
if v.anomalies:
    for a in v.anomalies:
        print(f"   ANOMALY: [{a.category}] {a.description}")

# Credential access
v = monitor.record("read_file", {"path": ".env"})
print(f"5. read_file .env → risk={v.risk.value}, alert={v.alert}")
if v.anomalies:
    for a in v.anomalies:
        print(f"   ANOMALY: [{a.category}] {a.description}")

# Session summary
print("\n=== Session Summary ===")
summary = monitor.summarize()
print(f"Total calls: {summary.total_calls}")
print(f"Unique tools: {summary.unique_tools}")
print(f"Session risk: {summary.risk_level.value}")
print(f"Anomalies detected: {summary.anomaly_count}")
print(f"Tool usage: {summary.tool_counts}")
