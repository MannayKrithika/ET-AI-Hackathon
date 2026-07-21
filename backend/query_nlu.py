"""
Query NLU — intent classification + multi-entity extraction for user
*queries* (as opposed to entity_extractor.py, which extracts from free-form
generated *answers* / document text using Gemini for /ask, /build_graph,
/compliance — those callers rely on that module's original vocabulary
("Equipment", "Status", ...) which knowledge_graph.py and
compliance_rules.py hard-code, so that module is intentionally left alone).

This module powers ONLY the /extract_entities endpoint. It is:
  - Rule-based / regex / keyword-dictionary — no LLM call, no transformer.
  - Scoped to the industrial query vocabulary requested for that endpoint.
  - Able to return multiple entities of different types from one query,
    plus a single classified intent.

Flow:
    text -> understand_query(text) -> {"intent": ..., "entities": [...]}
"""

import re

# ---------------------------------------------------------------------------
# Supported vocabulary
# ---------------------------------------------------------------------------

INTENTS = [
    "Equipment Search",
    "Maintenance History",
    "Inspection Search",
    "Inventory Check",
    "Spare Parts Lookup",
    "Risk Assessment",
    "Status Check",
    "Maintenance Schedule",
    "Document Lookup",
    "General Question",
]

ENTITY_TYPES = [
    "Equipment",
    "Equipment Model",
    "Equipment Attribute",  # e.g. "model" used as a search field, not a value
    "Asset ID",
    "Manufacturer",
    "Serial Number",
    "Maintenance ID",
    "Inspection ID",
    "Technician",
    "Engineer",
    "Department",
    "Location",
    "Status",
    "Risk Level",
    "Failure Type",
    "Part SKU",
    "Inventory",
    "Inventory Threshold",
    "Supplier",
    "Maintenance Date",
    "Inspection Date",
    "Temperature",
    "Pressure",
    "Downtime",
    "Maintenance Type",
    "Action Taken",
]

# ---------------------------------------------------------------------------
# ID patterns (checked before keyword dictionaries — most specific first)
# ---------------------------------------------------------------------------

_ID_PATTERNS: dict[str, re.Pattern] = {
    "Asset ID": re.compile(r"\b(?:EQ|AST|ASSET)-?\d{2,6}\b", re.IGNORECASE),
    "Maintenance ID": re.compile(r"\b(?:MNT|WO|MO|WORK-?ORDER)-?\d{2,6}\b", re.IGNORECASE),
    "Inspection ID": re.compile(r"\b(?:INS|INSP)-?\d{2,6}\b", re.IGNORECASE),
    "Schedule ID": re.compile(r"\b(?:SCH|PM)-?\d{2,6}\b", re.IGNORECASE),
    "Part SKU": re.compile(r"\b(?:SKU|PRT|PN)-?[A-Z0-9]{2,10}\b", re.IGNORECASE),
    "Serial Number": re.compile(r"\bSN-?[A-Z0-9]{3,15}\b", re.IGNORECASE),
}

# Fallback for exact codes that don't match any known prefix above — e.g.
# "MTR-AB-38_038". Requires a letter prefix plus at least one dash/underscore
# segment AND at least one digit somewhere, so it catches real identifiers
# without also matching ordinary hyphenated words ("state-of-the-art").
_GENERIC_ID_PATTERN = re.compile(r"\b[A-Za-z]{2,10}(?:[-_][A-Za-z0-9]+){1,6}\b")

_LOCATION_PATTERN = re.compile(
    r"\b(?:Plant|Zone|Building|Warehouse|Site|Unit|Floor)\s+[A-Za-z0-9]+\b",
    re.IGNORECASE,
)

_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

_TEMPERATURE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*°?\s*(?:C|F|celsius|fahrenheit)\b", re.IGNORECASE)
_PRESSURE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:psi|bar|kpa|mpa)\b", re.IGNORECASE)
_DOWNTIME_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?)\b", re.IGNORECASE)

# Two/three consecutive capitalized words -> candidate person name. Only
# matched mid-string (not the leading, always-capitalized first word of a
# sentence) to keep false positives down.
_NAME_PATTERN = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b")

# ---------------------------------------------------------------------------
# Keyword dictionaries: entity_type -> phrases, longest phrase first so a
# multi-word match (e.g. "under maintenance") wins over a shorter overlapping
# one. Matched as whole words/phrases, case-insensitive.
# ---------------------------------------------------------------------------

_KEYWORD_DICTS: dict[str, list[str]] = {
    "Equipment": [
        "hydraulic press", "pump", "compressor", "motor", "valve", "conveyor",
        "turbine", "generator", "boiler", "chiller", "gearbox", "bearing",
        "transformer", "engine", "fan", "blower", "actuator",
    ],
    "Equipment Attribute": ["model", "models"],
    "Manufacturer": [
        "siemens", "abb", "schneider electric", "honeywell", "emerson",
        "rockwell automation", "bosch", "caterpillar", "grundfos",
        "general electric", "mitsubishi electric", "yokogawa", "danfoss",
        "atlas copco",
    ],
    "Status": [
        "under maintenance", "non-functional", "critical", "running",
        "stopped", "operational", "breakdown", "failed", "faulty", "active",
        "inactive", "idle", "offline", "online", "decommissioned",
        "functional",
    ],
    "Risk Level": [
        "critical risk", "high risk", "medium risk", "low risk",
    ],
    "Failure Type": [
        "bearing failure", "motor burnout", "short circuit", "mechanical failure",
        "electrical failure", "overheating", "leakage", "corrosion", "wear and tear",
    ],
    "Maintenance Type": [
        "breakdown maintenance", "condition-based", "condition based",
        "preventive", "corrective", "predictive", "routine", "emergency",
        "scheduled", "unscheduled",
    ],
    "Action Taken": [
        "replaced", "repaired", "serviced", "inspected", "cleaned",
        "lubricated", "adjusted", "recalibrated", "overhauled", "tightened",
        "realigned",
    ],
    "Department": [
        "maintenance department", "mechanical department", "electrical department",
        "production department", "quality department", "safety department",
        "engineering department", "operations department",
    ],
    "Inventory Threshold": ["minimum stock", "below minimum", "reorder level"],
    "Inventory": ["spare parts", "spare part", "inventory", "stock"],
}

# Words a matched "name" must not be, to filter obvious false positives
# picked up by the capitalized-words heuristic.
_NAME_STOPWORDS = {"which", "show", "plant"}

# ---------------------------------------------------------------------------
# Intent patterns — phrase-based, scored by number of distinct phrase hits.
# Order = tie-break priority (first defined wins a tie).
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, list[tuple[re.Pattern, int]]]] = [
    ("Equipment Search", [
        (re.compile(r"\bwhich\s+(?:all\s+)?(?:models?|equipment|assets?|machines?)\b", re.I), 3),
        (re.compile(r"\binstalled in\b", re.I), 2),
        (re.compile(r"\blist\s+(?:of\s+)?(?:equipment|assets?|machines?)\b", re.I), 3),
        (re.compile(r"\bfind\s+(?:equipment|assets?|machines?)\b", re.I), 3),
    ]),
    ("Maintenance History", [
        (re.compile(r"\bmaintenance history\b", re.I), 2),
        (re.compile(r"\bservice history\b", re.I), 2),
        (re.compile(r"\brepair history\b", re.I), 2),
        (re.compile(r"\bpast maintenance\b", re.I), 2),
    ]),
    ("Inspection Search", [
        (re.compile(r"\binspection report(?:s)?\b", re.I), 2),
        (re.compile(r"\binspection (?:history|records?)\b", re.I), 2),
        (re.compile(r"\binspected by\b", re.I), 2),
    ]),
    ("Inventory Check", [
        (re.compile(r"\bminimum stock\b", re.I), 2),
        (re.compile(r"\bbelow minimum\b", re.I), 2),
        (re.compile(r"\breorder level\b", re.I), 2),
        (re.compile(r"\bout of stock\b", re.I), 2),
        (re.compile(r"\bstock\b", re.I), 1),
    ]),
    ("Spare Parts Lookup", [
        (re.compile(r"\bspare parts?\b", re.I), 2),
        (re.compile(r"\bpart (?:number|sku)\b", re.I), 2),
        (re.compile(r"\bcompatible part\b", re.I), 2),
    ]),
    ("Risk Assessment", [
        (re.compile(r"\brisk assessment\b", re.I), 3),
        (re.compile(r"\brisk level\b", re.I), 2),
        (re.compile(r"\bhigh risk\b", re.I), 2),
        (re.compile(r"\brisk\b", re.I), 1),
    ]),
    ("Maintenance Schedule", [
        (re.compile(r"\bnext maintenance\b", re.I), 2),
        (re.compile(r"\bmaintenance due\b", re.I), 2),
        (re.compile(r"\bscheduled maintenance\b", re.I), 2),
        (re.compile(r"\bupcoming maintenance\b", re.I), 2),
        (re.compile(r"\bpm schedule\b", re.I), 2),
    ]),
    ("Status Check", [
        (re.compile(r"\bstatus of\b", re.I), 2),
        (re.compile(r"\bcurrent status\b", re.I), 2),
        (re.compile(r"\bcondition\b", re.I), 1),
        (re.compile(r"\bcritical\b", re.I), 1),
    ]),
    ("Document Lookup", [
        (re.compile(r"\bmanual\b", re.I), 2),
        (re.compile(r"\bsop\b", re.I), 2),
        (re.compile(r"\bprocedure\b", re.I), 2),
        (re.compile(r"\bdatasheet\b", re.I), 2),
        (re.compile(r"\bdocument\b", re.I), 2),
    ]),
]


def classify_intent(text: str) -> str:
    """Scores each intent by weighted phrase-pattern hits, returns the
    highest scorer. Ties go to whichever intent is listed first above.
    Falls back to "General Question" if nothing matches."""
    best_intent, best_score = "General Question", 0
    for intent, patterns in _INTENT_PATTERNS:
        score = sum(weight for pattern, weight in patterns if pattern.search(text))
        if score > best_score:
            best_intent, best_score = intent, score
    return best_intent


def extract_query_entities(text: str, intent: str | None = None) -> list[dict]:
    """Rule-based, regex + keyword-dictionary entity extraction. Returns a
    list of {"entity": ..., "type": ...} dicts, possibly several per query
    and spanning several types — unlike a single-entity match."""
    entities: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, etype: str) -> None:
        value = value.strip()
        if not value:
            return
        key = (value.lower(), etype)
        if key in seen:
            return
        seen.add(key)
        entities.append({"entity": value, "type": etype})

    # 1. Structured IDs first (most specific, least ambiguous).
    for etype, pattern in _ID_PATTERNS.items():
        for match in pattern.finditer(text):
            add(match.group(0).upper(), etype)

    # 2. Locations ("Plant A", "Zone 2", ...).
    for match in _LOCATION_PATTERN.finditer(text):
        add(match.group(0), "Location")

    # 3. Keyword dictionaries, longest phrase first so a multi-word phrase
    #    wins over a shorter, overlapping one. Stop at the first hit per
    #    type — the dict entries are alternative phrasings of one category,
    #    not independent facts, so one canonical match per type is enough.
    for etype, keywords in _KEYWORD_DICTS.items():
        for kw in sorted(keywords, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                add(kw, etype)
                break

    # 4. Dates, disambiguated by nearby context; default to Maintenance
    #    Date if no stronger signal either way.
    for match in _DATE_PATTERN.finditer(text):
        window = text[max(0, match.start() - 20): match.end() + 20].lower()
        if "inspection" in window:
            add(match.group(0), "Inspection Date")
        else:
            add(match.group(0), "Maintenance Date")

    # 5. Numeric readings.
    for match in _TEMPERATURE_PATTERN.finditer(text):
        add(match.group(0), "Temperature")
    for match in _PRESSURE_PATTERN.finditer(text):
        add(match.group(0), "Pressure")
    for match in _DOWNTIME_PATTERN.finditer(text):
        window = text[max(0, match.start() - 20): match.start()].lower()
        if "downtime" in window or "down" in window:
            add(match.group(0), "Downtime")

    # 6. Manufacturer / Supplier via explicit "X by/from ..." phrasing.
    m = re.search(r"(?:manufactured by|manufacturer\s*[:\-])\s*([A-Za-z0-9 &.]+)", text, re.I)
    if m:
        add(m.group(1), "Manufacturer")
    m = re.search(r"(?:supplied by|supplier\s*[:\-])\s*([A-Za-z0-9 &.]+)", text, re.I)
    if m:
        add(m.group(1), "Supplier")

    # 7. Person names -> Technician or Engineer. Default depends on intent:
    #    inspection-flavoured queries default to Engineer, maintenance-
    #    flavoured ones to Technician; an explicit nearby keyword overrides.
    lower_text = text.lower()
    resolved_intent = intent or classify_intent(text)
    for match in _NAME_PATTERN.finditer(text):
        name = match.group(0)
        if name.split()[0].lower() in _NAME_STOPWORDS:
            continue
        window = lower_text[max(0, match.start() - 25): match.start()]
        if "technician" in window:
            add(name, "Technician")
        elif "engineer" in window:
            add(name, "Engineer")
        elif resolved_intent == "Inspection Search":
            add(name, "Engineer")
        elif resolved_intent in ("Maintenance History", "Maintenance Schedule"):
            add(name, "Technician")
        else:
            add(name, "Engineer")

    return entities


def understand_query(text: str) -> dict:
    """Single entry point for the /extract_entities endpoint: classifies
    intent, then extracts entities (passing the resolved intent through so
    name disambiguation can use it), and returns both."""
    intent = classify_intent(text)
    entities = extract_query_entities(text, intent=intent)
    return {"intent": intent, "entities": entities}


def detect_identifiers(text: str) -> list[str]:
    """
    Returns the exact identifier strings (uppercased, deduplicated) found in
    a query — Asset IDs, Maintenance IDs, Inspection IDs, Schedule IDs, Part
    SKUs, Serial Numbers, plus anything else that *looks* like a structured
    code via the generic fallback pattern (e.g. "MTR-AB-38_038").

    This is what hybrid retrieval (rag.py) calls to decide whether a query
    needs exact substring matching over chunk text in addition to — or
    instead of — semantic FAISS search. Pure rule-based/regex, so it's fast
    and has no embedding/LLM dependency, consistent with the rest of this
    module.
    """
    if not text or not text.strip():
        return []

    identifiers: set[str] = set()

    for pattern in _ID_PATTERNS.values():
        for match in pattern.finditer(text):
            identifiers.add(match.group(0).upper())

    for match in _GENERIC_ID_PATTERN.finditer(text):
        token = match.group(0)
        # Require a digit somewhere in the token so plain hyphenated words
        # ("state-of-the-art", "follow-up") aren't mistaken for codes.
        if re.search(r"\d", token):
            identifiers.add(token.upper())

    return sorted(identifiers)