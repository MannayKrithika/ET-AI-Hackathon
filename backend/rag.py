"""
Gemini + RAG — Step 7.

This is the module that connects retrieval (Steps 5-6: embeddings + FAISS)
to generation. Nothing here re-implements search — it calls straight into
faiss_index.search_index() and embeddings.embed_texts(), the same functions
the /search sanity-check route already uses, so retrieval behaves identically
here and there.

Deliberately NOT using langchain's FAISS wrapper / HuggingFaceEmbeddings:
this project already has its own FAISS index (faiss_index.py) built on
fastembed vectors, with real validation (dimension checks, duplicate
chunk_id detection, NaN/Inf guards) that langchain's wrapper doesn't do for
you. Swapping to langchain here would mean re-embedding everything with a
different model and throwing that validation away for no benefit — the
retrieval side stays exactly as it is; only generation is new in this file.

Uses google-genai (the current, maintained Google SDK) rather than
google-generativeai — the latter is deprecated as of 2026 and prints a
FutureWarning on import.

Flow:
    question
      -> embed_texts([question])          (Step 5's embedder, same model
                                             the corpus was embedded with)
      -> search_index(...)                  (Step 6's FAISS index)
      -> build_context(...)                 (chunks -> a labelled context
                                             block Gemini can cite from)
      -> build_prompt(...)                  (grounding instructions + context
                                             + question)
      -> Gemini generate_content(...)
      -> {"answer", "sources", "chunks_used"}

Grounding is enforced two ways, not just by asking nicely in the prompt:
  1. The prompt explicitly instructs the model to answer only from the
     provided context and to say when something isn't covered.
  2. If retrieval itself comes back empty (no index yet, or zero results),
     Gemini is never called at all — there's no way for it to "answer" a
     question with no context, so we return the not-found message directly
     and save an API call.
"""

import os
from pathlib import Path

from embeddings import embed_texts
from faiss_index import load_index, search_index
from query_nlu import detect_identifiers
from exact_match import exact_match_search

MODEL_NAME = "gemini-3.1-flash-lite"

NOT_FOUND_MESSAGE = "The information is not found in the uploaded documents."

_client = None


def _get_client():
    """
    Lazy singleton, mirroring the pattern in embeddings.get_model() — the
    Gemini client only gets built the first time it's actually needed, not
    at server import time, so a missing/blank API key doesn't crash
    startup; it only fails the first /ask call, with a clear error.

    Uses google-genai (the current, maintained SDK), not the older
    google-generativeai package — that one is deprecated and prints a
    FutureWarning on import as of mid-2026. The client picks up
    GOOGLE_API_KEY from the environment automatically.
    """
    global _client

    if _client is None:
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        print("Using API key:", api_key[:10])
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to backend/.env "
                "(see .env.example) before calling /ask."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def build_context(chunks: list[dict]) -> str:
    """
    Turns retrieved chunks into a labelled context block. Each chunk is
    tagged with its own source line (document + page/section) so Gemini can
    cite a source with the answer, rather than returning a fact with no way
    to trace it back — that traceability was the whole point of adding
    source_page in Step 4.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("document_name", "Unknown document")
        page = chunk.get("source_page")
        section = chunk.get("section_title")

        source_bits = [doc]
        if page is not None:
            source_bits.append(f"page {page}")
        elif section:
            source_bits.append(section)
        source_label = ", ".join(source_bits)

        blocks.append(f"[Source {i}: {source_label}]\n{chunk['text']}")

    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return f"""You are an industrial document assistant.

Answer only using the provided context. Do not use outside knowledge and do not guess.
If the answer is not available in the context, say exactly:
"{NOT_FOUND_MESSAGE}"

When you do answer, mention which source(s) you used, e.g. "(Source 2)".

Context:
{context}

Question:
{question}

Answer:"""


def answer_question(question: str, indexes_dir: Path, top_k: int = 4) -> dict:
    """
    The Step 7 deliverable, now with hybrid retrieval on top: retrieves
    chunks from FAISS (semantic) and, when the question contains an exact
    identifier (Asset ID, Maintenance ID, Inspection ID, Schedule ID, Part
    SKU, Serial Number, ...), also retrieves every chunk containing that
    identifier verbatim, across every uploaded document. Both result sets
    are merged (exact matches first) before being sent to Gemini.

    Returns:
        {
            "question": str,
            "answer": str,
            "grounded": bool,        # False when no chunks existed to answer from
            "chunks_used": [ {chunk_id, document_name, source_page, score,
                               match_type, ...} ],
        }
    """
    index, meta = load_index(indexes_dir)

    if index is None:
        return {
            "question": question,
            "answer": NOT_FOUND_MESSAGE,
            "grounded": False,
            "chunks_used": [],
        }

    # --- Identifier detection + exact match retrieval ----------------------
    # Rule-based (query_nlu.py) — no embeddings involved. An identifier like
    # "EQ-2001" isn't semantic content, so it shouldn't be searched for with
    # vector similarity; it should be searched for verbatim.
    identifiers = detect_identifiers(question)
    exact_matches = exact_match_search(meta, identifiers) if identifiers else []

    # --- Semantic retrieval --------------------------------------------
    # Exactly the same FAISS path as before Step 2's hybrid retrieval, run
    # unconditionally: a no-identifier query behaves identically to the
    # original implementation, and an identifier query still gets
    # supporting semantic context merged in per Step 5.
    query_vector = embed_texts([question])
    semantic_matches = (
        search_index(index, meta, query_vector[0], top_k=top_k)
        if query_vector.shape[0] != 0
        else []
    )

    # --- Merge: exact matches first, then semantic matches, deduped -----
    exact_chunk_ids = {c["chunk_id"] for c in exact_matches}
    merged_chunks = list(exact_matches)
    for chunk in semantic_matches:
        if chunk["chunk_id"] in exact_chunk_ids:
            continue  # already included via the exact-match pass
        merged_chunks.append({**chunk, "match_type": "semantic"})

    if not merged_chunks:
        return {
            "question": question,
            "answer": NOT_FOUND_MESSAGE,
            "grounded": False,
            "chunks_used": [],
        }

    context = build_context(merged_chunks)
    prompt = build_prompt(question, context)

    client = _get_client()
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    return {
        "question": question,
        "answer": response.text.strip(),
        "grounded": True,
        "chunks_used": [
            {
                "chunk_id": c["chunk_id"],
                "document_name": c["document_name"],
                "section_title": c.get("section_title"),
                "source_page": c.get("source_page"),
                "score": round(c["score"], 4),
                "match_type": c.get("match_type", "semantic"),
                **({"matched_identifiers": c["matched_identifiers"]} if "matched_identifiers" in c else {}),
            }
            for c in merged_chunks
        ],
    }