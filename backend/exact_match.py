"""
Exact identifier match retrieval — the "Exact Match Retrieval" arm of the
hybrid retrieval pipeline (see rag.py for where this plugs in).

Why this exists: embeddings are trained to capture meaning, not to
distinguish one alphanumeric code from another that looks similar. A query
like "Tell me everything about EQ-2001" can end up semantically closest to
chunks that merely share vocabulary ("equipment", "condition") without ever
mentioning EQ-2001 itself, which is exactly the bug this module fixes.

Deliberately does NOT touch faiss_index.py, embeddings.py, or the FAISS
index file itself — this reads the same corpus.meta.json that
faiss_index.load_index() already produces (specifically meta["chunks"]:
{chunk_id: chunk_dict}, where chunk_dict["text"] is the chunk's raw text)
and does a plain case-insensitive substring scan over it. No re-embedding,
no new index, no vector math.
"""

import re

# Safety cap so one identifier that happens to appear across a huge corpus
# can't blow out the Gemini context window. Generous for this project's
# scale (per faiss_index.py's own comments: "thousands of chunks, not
# millions") — in practice a single Asset ID appearing in the Asset
# Register, Maintenance Log, Inspection Report, Breakdown History, and PM
# Schedule is ~5 chunks, well under this cap.
MAX_EXACT_MATCHES = 30

# chunking.py builds chunk_id as f"{asset_id}_{counter:03d}" for tabular
# rows (chunk_csv/chunk_xlsx) — e.g. asset_id "MTR-AB-38" becomes chunk_id
# "MTR-AB-38_038". The frontend surfaces chunk_id as the source label, so
# users end up querying for the internal chunk_id rather than the real
# identifier that's actually written in the chunk text. Rather than fix
# that display issue here (frontend is out of scope), exact match falls
# back to stripping this exact suffix shape and retrying.
_COUNTER_SUFFIX_PATTERN = re.compile(r"_\d{3}$")


def _strip_chunk_counter_suffix(identifier: str) -> str | None:
    """Returns identifier with a trailing '_NNN' (exactly 3 digits) removed,
    or None if it doesn't end that way."""
    if _COUNTER_SUFFIX_PATTERN.search(identifier):
        return _COUNTER_SUFFIX_PATTERN.sub("", identifier)
    return None


def _scan(chunks: dict, identifiers: list[str]) -> list[dict]:
    lowered_ids = [identifier.lower() for identifier in identifiers]
    matches = []
    for chunk in chunks.values():
        text = (chunk.get("text") or "").lower()
        if not text:
            continue
        hit_ids = [orig for orig, low in zip(identifiers, lowered_ids) if low in text]
        if not hit_ids:
            continue
        matches.append({
            **chunk,
            "score": 1.0,
            "match_type": "exact",
            "matched_identifiers": hit_ids,
        })
    return matches


def exact_match_search(meta: dict | None, identifiers: list[str]) -> list[dict]:
    """
    meta: the dict returned by faiss_index.load_index()'s second value
          (corpus.meta.json content).
    identifiers: exact strings to search for, e.g. ["EQ-2001"] — as
                 produced by query_nlu.detect_identifiers().

    Returns every chunk (across every document already in the index) whose
    text contains ANY of the given identifiers as a case-insensitive
    substring — so one EQ-2001 query pulls its chunk from the Asset
    Register, Maintenance Log, Inspection Report, Breakdown History, and PM
    Schedule all in one merged list. Each result is tagged:
      - "score": 1.0            (ranks above any cosine score, which maxes at 1.0)
      - "match_type": "exact"
      - "matched_identifiers": [...]   (which requested identifier(s) hit this chunk)

    If nothing matches on the first pass, retries with any identifier's
    trailing "_NNN" chunk-counter suffix stripped (see
    _strip_chunk_counter_suffix) before giving up — see module docstring.

    Chunks matching more of the requested identifiers are ranked first
    among exact matches (relevant when a query mentions more than one ID).
    """
    if not identifiers or not meta:
        return []

    chunks = meta.get("chunks", {})
    if not chunks:
        return []

    matches = _scan(chunks, identifiers)

    if not matches:
        fallback_ids = [
            stripped
            for identifier in identifiers
            if (stripped := _strip_chunk_counter_suffix(identifier)) is not None
        ]
        if fallback_ids:
            matches = _scan(chunks, fallback_ids)

    matches.sort(key=lambda c: len(c["matched_identifiers"]), reverse=True)
    return matches[:MAX_EXACT_MATCHES]