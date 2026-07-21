"""
Entity Extraction — Step 9.

Pulls structured entities (equipment, IDs, status, operators, dates, etc.)
out of free text using Gemini, so Step 10 (Knowledge Graph) has something
to build nodes and edges from.

Deliberately reuses the same lazy-singleton Gemini client pattern as
rag.py (_get_client / google-genai / GOOGLE_API_KEY from the environment)
rather than introducing a second way of talking to Gemini in this project.

Flow:
    text
      -> build_extraction_prompt(text)   (instructs Gemini to return
                                           JSON-only, scoped to ENTITY_TYPES)
      -> Gemini generate_content(...)
      -> _parse_entities_response(...)   (strips ```json fences if present,
                                           json.loads, validates shape)
      -> [ {"entity": ..., "type": ...}, ... ]

Kept independent of rag.py / faiss_index.py: this module only knows how to
turn a string of text into a list of entities. It doesn't care whether that
text came from a chunk, an uploaded document, or a Gemini RAG answer — all
three are handled the same way, since extract_entities() takes plain text.
"""

import json
import os
from unicodedata import category

MODEL_NAME = "gemini-3.1-flash-lite"

# Categories the extractor is scoped to. Kept in one place so the prompt,
# the docstring examples, and (later) the Knowledge Graph in Step 10 all
# agree on the same vocabulary.
ENTITY_TYPES = [
    "Equipment",
    "Equipment_ID",
    "Ticket_ID",
    "Status",
    "Location",
    "Operator",
    "Department",
    "Date",
    "Reading",
]

_client = None


def _get_client():
    """
    Lazy singleton — identical pattern to rag.py's _get_client(), so the
    Gemini client is only constructed the first time extraction is actually
    used, and a missing GOOGLE_API_KEY fails clearly at call time instead of
    at import/startup time.
    """
    global _client

    if _client is None:
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to backend/.env "
                "(see .env.example) before calling /extract_entities."
            )

        _client = genai.Client(api_key=api_key)

    return _client

def build_extraction_prompt(text: str) -> str:
    categories = "\n".join(f"- {t}" for t in ENTITY_TYPES)
    return f"""Extract all industrial entities from the text below.

Categories:
{categories}

Definitions:
- Equipment: machine or asset names (Pump, Compressor, Hydraulic Press)
- Equipment_ID: IDs like EQ-2009, P-101
- Ticket_ID: IDs like BRK-8000, BRK-8012
- Status: operational conditions such as breakdown, failed, fault, maintenance, critical, running, stopped
- Date: any dates
- Location: plant or area names
- Operator: person names
- Department: department names
- Reading: numeric readings with units

Rules:
- Return ONLY a JSON array.
- Do NOT return a JSON object.
- Every entity must be an object like:

[
  {{"entity":"Pump","type":"Equipment"}},
  {{"entity":"P-101","type":"Equipment_ID"}},
  {{"entity":"EQ-2009","type":"Equipment_ID"}},
  {{"entity":"BRK-8012","type":"Ticket_ID"}},
  {{"entity":"breakdown","type":"Status"}},
  {{"entity":"15 June 2026","type":"Date"}}
]

- Do not use keys like "Equipment" or "Status".
- Return JSON only.

Text:
{text}

JSON:
"""
#     categories = "\n".join(f"- {t}" for t in ENTITY_TYPES)

#     return f"""Extract all industrial entities from the text below.

# Categories:
# {categories}

# Examples:
# - Ticket IDs like BRK-8000, BRK-8012, BRK-9999 should be extracted as "Ticket_ID".
# - Equipment IDs like EQ-2009 should be extracted as "Equipment_ID".

# Rules:
# - Return JSON only — no preamble, no explanation, no Markdown code fences.
# - Return a JSON array of objects, each shaped exactly like:
#   {{"entity": "<the exact text found>", "type": "<one of the categories above>"}}
# - Only use the categories listed above for "type".
# - Only extract entities that actually appear in the text below.
# - If nothing relevant is found, return an empty array: []

# Text:
# {text}

# JSON:"""
# def build_extraction_prompt(text: str) -> str:
#     return f"""Extract all industrial entities from the text below.
# Categories:
# {categories}
# Ticket IDs are values like BRK-8000, BRK-8012, BRK-9999.
# Extract them as type "Ticket_ID".

# Rules:
# ...

# Rules:
# - Return JSON only — no preamble, no explanation, no Markdown code fences.
# - Return a JSON array of objects, each shaped exactly like:
#   {{"entity": "<the exact text found>", "type": "<one of the categories above>"}}
# - Only use the categories listed above for "type".
# - Only extract entities that actually appear in the text below.
# - If nothing relevant is found, return an empty array: []

# Text:
# {text}

# JSON:"""


def _strip_code_fences(raw: str) -> str:
    """
    Gemini is instructed to return JSON only, but models sometimes wrap the
    output in ```json ... ``` anyway. Strip that defensively before parsing
    rather than letting json.loads fail on well-formed-but-fenced output.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (with optional language tag) and the
        # closing fence.
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()
    return cleaned


def _parse_entities_response(raw_text: str) -> list[dict]:
    """
    Turns Gemini's raw response text into a validated list of
    {"entity": str, "type": str} dicts. Anything malformed (not a list,
    missing keys, unknown type) is dropped rather than raising — a partial,
    valid list is more useful downstream (Step 10) than a hard failure over
    one bad element.
    """
    cleaned = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
          entities = []
          for entity_type, values in parsed.items():
             if entity_type not in ENTITY_TYPES:
               continue

             if isinstance(values, list):
                for value in values:
                  entities.append({
                    "entity": str(value),
                    "type": entity_type
                })

          return entities
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        entity_type = item.get("type")
        if not entity or not entity_type:
            continue
        if entity_type not in ENTITY_TYPES:
            continue
        entities.append({"entity": str(entity).strip(), "type": entity_type})

    return entities


def extract_entities(text: str) -> list[dict]:
    """
    The Step 9 deliverable: sends text to Gemini and returns a list of
    extracted entities.

        extract_entities("Hydraulic Press (Equipment ID: EQ-2009) is marked Critical.")
        -> [
            {"entity": "Hydraulic Press", "type": "Equipment"},
            {"entity": "EQ-2009", "type": "Equipment_ID"},
            {"entity": "Critical", "type": "Status"},
        ]

    Returns an empty list for blank input, or if the model call / parsing
    fails — callers (the /extract_entities route, the /ask integration)
    should treat an empty list as "no entities found", not as an error.
    """
    if not text or not text.strip():
        return []

    client = _get_client()
    prompt = build_extraction_prompt(text)

    print("\n===== PROMPT =====")
    print(prompt)
    response = client.models.generate_content(
      model=MODEL_NAME,
      contents=prompt
    )
    print("\n===== GEMINI RESPONSE =====")
    print(response.text)
    entities = _parse_entities_response(response.text)
    print("\n===== PARSED ENTITIES =====")
    print(entities)
    return entities
