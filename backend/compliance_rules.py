"""
Compliance Rules — Step 11.

A small, hackathon-scoped rule set for the Compliance Checker. This is NOT
meant to encode every Factory Act / PESO regulation — it's a working
prototype that demonstrates the pattern: compare extracted entities against
simple structural rules and flag what's missing.

Each rule has:
  - "name": human-readable label shown in the report.
  - "condition": a dict of {entity_type: required_value}. The rule only
        applies if every condition is satisfied by the extracted entities
        (i.e. that type/value pair appears among them). An empty dict means
        the rule always applies.
  - "required": a list of entity types that must be present (regardless of
        their value) for the rule to PASS. Anything missing is reported.

Add more rules here as the compliance scope grows — main.py and
compliance_checker.py don't need to change.
"""

COMPLIANCE_RULES = [
    {
        "name": "Breakdown Ticket Required",
        "condition": {
            "Status": "breakdown"
        },
        "required": [
            "Ticket_ID"
        ]
    },
    {
        "name": "Equipment ID Required",
        "condition": {},
        "required": [
            "Equipment_ID"
        ]
    },
    {
        "name": "Date Required",
        "condition": {},
        "required": [
            "Date"
        ]
    }
]
