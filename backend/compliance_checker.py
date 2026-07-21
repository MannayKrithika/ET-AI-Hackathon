"""
Compliance Checker — Step 11.

Goal: compare extracted entities from a document/answer against a small set
of compliance rules (compliance_rules.py) and report any violations.

Pipeline:
    Document -> Text Extraction -> Entity Extraction -> Compliance Checker
                                                       -> Compliance Report

This deliberately mirrors the simplicity of Step 9/10: check_compliance()
takes the same entity list shape produced by extract_entities()
(`[{"entity": ..., "type": ...}, ...]`) and returns a structured report —
no external rule engine, no legal database, just a straightforward
condition/required check against a small Python rule list.
"""

from compliance_rules import COMPLIANCE_RULES


def _group_entities_by_type(entities: list[dict]) -> dict:
    """
    Turns [{"entity": "P-101", "type": "Equipment_ID"}, ...] into
    {"Equipment_ID": ["P-101", ...], ...} so rule conditions/requirements
    can be checked with simple dict/list lookups.
    """
    entity_types = {}
    for e in entities:
        entity_types.setdefault(e["type"], []).append(e["entity"])
    return entity_types


def _condition_met(entity_types: dict, entity_type: str, expected_value: str) -> bool:
    """
    Case-insensitive check for whether `expected_value` appears among the
    extracted values for `entity_type`. Gemini's extraction may return
    "Breakdown" where the rule says "breakdown" — comparing case-insensitively
    avoids spurious rule mismatches over capitalization alone.
    """
    values = entity_types.get(entity_type, [])
    return any(str(v).strip().lower() == expected_value.strip().lower() for v in values)


def check_compliance(entities: list[dict]) -> list[dict]:
    """
    Checks a list of extracted entities against COMPLIANCE_RULES.

    Returns a list of report entries, one per rule that applies:
        [
          {"rule": "Breakdown Ticket Required", "status": "PASS", "missing": []},
          {"rule": "Equipment ID Required", "status": "FAIL", "missing": ["Equipment_ID"]},
          ...
        ]

    A rule only appears in the report if its "condition" is satisfied (or
    empty, meaning it always applies). Rules whose condition isn't met are
    simply skipped — they don't apply to this document.
    """
    entity_types = _group_entities_by_type(entities)
    report = []

    for rule in COMPLIANCE_RULES:
        # Check whether this rule's condition applies to these entities.
        condition_applies = True
        for entity_type, expected_value in rule["condition"].items():
            if not _condition_met(entity_types, entity_type, expected_value):
                condition_applies = False
                break

        if not condition_applies:
            continue

        # Condition applies (or there was none) — check required entities.
        missing = [req for req in rule["required"] if req not in entity_types]

        report.append({
            "rule": rule["name"],
            "status": "PASS" if not missing else "FAIL",
            "missing": missing,
        })

    return report
