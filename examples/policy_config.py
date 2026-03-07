"""Enterprise policy configuration with Sentinel AI.

Define custom safety policies per environment, team, or use case.
"""

from sentinel import Policy

# Define a strict production policy
policy = Policy.from_dict({
    "name": "production-strict",
    "block_threshold": "high",
    "redact_pii": True,
    "scanners": {
        "prompt_injection": {"enabled": True},
        "pii": {"enabled": True},
        "harmful_content": {"enabled": True},
        "hallucination": {"enabled": False},  # less relevant for production
    },
    "custom_blocked_terms": [
        "internal_project_alpha",
        "competitor_product_name",
    ],
})

guard = policy.build_guard()

# Blocked terms are detected
result = guard.scan("Let's discuss internal_project_alpha in the meeting")
print(f"Blocked term found: {result.findings[0].description}")

# PII still redacted
result = guard.scan("Send the report to alice@company.com")
print(f"Redacted: {result.redacted_text}")
