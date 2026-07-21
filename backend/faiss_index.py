"""
FAISS Vector Index — Step 6.

Step 5 (embeddings.py) produces one .npy + .meta.json per document, kept
separate so a single re-upload doesn't force re-embedding everything else.
This step combines all of those into ONE searchable index, since a query
needs to search across the whole corpus, not one document at a time.

Pipeline so far:
    Upload -> Extract -> Chunk -> Embed (.npy, per-document)
    -> [this file] Build FAISS index (combined, whole corpus)
    -> Step 7: Retrieve top-k -> Step 8: Gemini answer

We use IndexFlatIP (inner product), not IndexFlatL2. Reason: embeddings.py
already L2-normalizes every vector at embed time, and for unit-length
vectors, inner product == cosine similarity. Cosine similarity is the
right metric for text embeddings — it compares direction (meaning), not
magnitude — so IP on normalized vectors is the correct choice, not IVF or
L2. "Flat" means exact search (no approximation): correct for a hackathon
corpus (thousands of chunks, not millions), where exact top-k beats a
faster-but-approximate index.

Output:
    indexes/
        corpus.index        <- faiss.IndexFlatIP, all chunks from all documents
        corpus.meta.json     <- {"chunk_ids": [...], "chunks": {chunk_id: full_chunk_dict}}
        corpus.info.json     <- retrieval-level metadata: embedding model, dim,
                                 per-document chunk counts, build timestamp.
                                 Kept separate from corpus.meta.json on purpose —
                                 info.json is small and human-skimmable (good for
                                 a demo/debug check), meta.json holds the bulk
                                 per-chunk data. Step 7 should read info.json
                                 first to confirm the query embedding model
                                 matches what the index was built with.

corpus.meta.json is what turns "FAISS says row 482 is the closest match"
back into an actual chunk of text with its document name, page, etc. FAISS
only ever stores and returns vectors + row numbers — it has no idea what a
"chunk" or "document" is, so this side-file is required, not optional.

Validation done at build time (fail loud rather than silently corrupt the
index):
  - embedding dimension consistency across documents
  - each document's declared model/dim (from its own .meta.json) matches
    what the rest of the corpus was embedded with — catches a partial
    re-embed (e.g. someone changes MODEL_NAME and only re-embeds one file)
  - duplicate chunk_ids across documents (would silently overwrite one
    chunk's text/metadata with another's in chunks_by_id)
  - every chunk_id an embeddings file claims must actually exist in that
    document's chunks.json — a mismatch here means a chunk vector would
    have no way to be turned back into readable text at search time
  - NaN/Inf vectors

Anything that fails one of these checks is skipped, not silently dropped:
every skip is recorded with a reason and returned in the response, so the
caller can tell "10 documents uploaded, 7 indexed" apart from "10 uploaded,
10 indexed" at a glance instead of just seeing a smaller total_chunks and
wondering why.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from embeddings import MODEL_NAME, EMBEDDING_DIM


def build_index(chunks_dir: Path, embeddings_dir: Path, indexes_dir: Path) -> dict:
    """
    Reads every {slug}_embeddings.npy in embeddings_dir, concatenates them
    in a fixed order, builds one IndexFlatIP over all vectors, and saves
    both the index and the row -> chunk lookup needed to make sense of
    search results later.

    Returns a summary dict (never the raw index/vectors — those aren't
    JSON-serializable and aren't useful in an API response). Documents that
    can't be safely included are listed under "skipped" with a reason,
    rather than silently vanishing from the count.
    """
    npy_files = sorted(embeddings_dir.glob("*_embeddings.npy"))

    if not npy_files:
        return {"success": False, "reason": "No embeddings found. Run Step 5 first.", "total_chunks": 0}

    all_vectors = []
    all_chunk_ids = []
    chunks_by_id = {}
    document_stats = {}  # slug -> chunk count, for the hackathon demo / debugging
    documents_order = []  # slugs in the order they were folded into the index
    skipped = []  # [{"document": slug, "reason": "..."}], never silent
    dim = None

    for npy_path in npy_files:
        slug = npy_path.stem.removesuffix("_embeddings")
        meta_path = embeddings_dir / f"{slug}_embeddings.meta.json"
        chunks_path = chunks_dir / f"{slug}_chunks.json"

        if not meta_path.exists() or not chunks_path.exists():
            skipped.append({
                "document": slug,
                "reason": "missing embeddings meta file or chunks file",
            })
            continue

        vectors = np.load(npy_path)
        if vectors.size == 0:
            skipped.append({"document": slug, "reason": "embeddings file is empty"})
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunk_ids = meta["chunk_ids"]

        if len(chunk_ids) != vectors.shape[0]:
            # Row count must match id count 1:1, or row i would map to the
            # wrong chunk_id silently — safer to skip this document's
            # contribution than to build a corrupted index.
            skipped.append({
                "document": slug,
                "reason": f"chunk_ids count ({len(chunk_ids)}) != vector rows ({vectors.shape[0]})",
            })
            continue

        # Cross-check the document's OWN declared model/dim (written by
        # embeddings.py into its .meta.json) against the actual vector
        # shape and against the rest of the corpus. This catches a partial
        # re-embed — e.g. MODEL_NAME changes and only one file gets
        # re-embedded — before it can silently corrupt the combined index.
        declared_dim = meta.get("dim")
        declared_model = meta.get("model")

        if declared_dim != vectors.shape[1]:
            skipped.append({
                "document": slug,
                "reason": f"meta declares dim={declared_dim} but vectors have shape[1]={vectors.shape[1]}",
            })
            continue

        if declared_model != MODEL_NAME:
            skipped.append({
                "document": slug,
                "reason": f"embedded with model '{declared_model}', current model is '{MODEL_NAME}' — re-embed this document",
            })
            continue

        if dim is None:
            dim = vectors.shape[1]
        elif vectors.shape[1] != dim:
            skipped.append({
                "document": slug,
                "reason": f"embedding dimension mismatch (expected {dim}, got {vectors.shape[1]})",
            })
            continue

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks_by_id_for_doc = {c["chunk_id"]: c for c in chunks}

        # Every chunk_id the embeddings file claims must actually resolve to
        # a real chunk. If one doesn't, we can't safely index this document:
        # a vector with no matching chunk means a search hit that can never
        # be turned back into readable text (silently dropped at search
        # time in the old version of this file). Fail this whole document
        # rather than partially include it with a hole in the middle.
        missing_chunk_ids = [cid for cid in chunk_ids if cid not in chunks_by_id_for_doc]
        if missing_chunk_ids:
            skipped.append({
                "document": slug,
                "reason": f"{len(missing_chunk_ids)} chunk_id(s) in embeddings have no matching chunk, e.g. {missing_chunk_ids[0]}",
            })
            continue

        # Duplicate chunk_id across documents would silently overwrite an
        # earlier document's chunk text/metadata in chunks_by_id, corrupting
        # retrieval for both documents without any error. Fail loud instead.
        duplicate_ids = [cid for cid in chunk_ids if cid in chunks_by_id]
        if duplicate_ids:
            skipped.append({
                "document": slug,
                "reason": f"duplicate chunk_id already present in index, e.g. '{duplicate_ids[0]}'",
            })
            continue

        all_vectors.append(vectors)  # already float32 + normalized from embeddings.py
        all_chunk_ids.extend(chunk_ids)
        document_stats[slug] = len(chunk_ids)
        documents_order.append(slug)
        for cid in chunk_ids:
            chunks_by_id[cid] = chunks_by_id_for_doc[cid]

    if not all_vectors:
        return {
            "success": False,
            "reason": "No valid embeddings to index.",
            "total_chunks": 0,
            "skipped": skipped,
        }

    matrix = np.vstack(all_vectors)  # already float32 — vectors loaded via np.load stay float32
    if matrix.dtype != np.float32:
        matrix = matrix.astype("float32")

    # Fail loud rather than build a silently-broken index: a NaN/Inf vector
    # (e.g. from an all-empty chunk, or a corrupted .npy) would still let
    # FAISS build an index, but every distance touching that row afterwards
    # becomes garbage — better to catch it here, at build time.
    if np.isnan(matrix).any():
        raise ValueError("Embedding matrix contains NaN values — check chunks with empty or malformed text.")
    if np.isinf(matrix).any():
        raise ValueError("Embedding matrix contains Inf values — check chunks with empty or malformed text.")

    # Embeddings are already normalized in embeddings.py, but normalizing
    # again here is a cheap no-op for unit vectors and makes this function
    # correct even if it's ever fed vectors from somewhere else that
    # forgot to normalize.
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    indexes_dir.mkdir(exist_ok=True)
    index_path = indexes_dir / "corpus.index"
    meta_out_path = indexes_dir / "corpus.meta.json"
    info_out_path = indexes_dir / "corpus.info.json"

    faiss.write_index(index, str(index_path))
    meta_out_path.write_text(json.dumps({
        "dim": dim,
        "total_chunks": len(all_chunk_ids),
        "chunk_ids": all_chunk_ids,          # row i in the index == chunk_ids[i]
        "chunks": chunks_by_id,               # chunk_id -> full chunk dict
    }, indent=2), encoding="utf-8")

    # Separate, small, human-skimmable file: retrieval-level metadata only.
    # Step 7 can check "embedding_model" here before embedding a query, to
    # catch a model mismatch immediately instead of getting silently wrong
    # (but not obviously wrong) search results.
    info_out_path.write_text(json.dumps({
        "embedding_model": MODEL_NAME,
        "dimension": dim,
        "documents": len(document_stats),
        "document_list": documents_order,   # slugs, in the order folded into the index
        "chunks": len(all_chunk_ids),
        "document_stats": document_stats,   # slug -> chunk count
        "skipped": skipped,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "cosine",                  # IndexFlatIP over L2-normalized vectors
    }, indent=2), encoding="utf-8")

    return {
        "success": True,
        "total_chunks": len(all_chunk_ids),
        "dim": dim,
        "dimension": dim,
        "metric": "cosine",
        "embedding_model": MODEL_NAME,
        "documents_indexed": len(document_stats),
        "documents": documents_order,
        "index_file": index_path.name,
        "skipped": skipped,
    }


def load_index(indexes_dir: Path) -> tuple["faiss.Index | None", dict | None]:
    """
    Loads the saved index + metadata for searching (Step 7 uses this).
    Returns (index, meta_dict) or (None, None) if no index has been built yet.
    """
    index_path = indexes_dir / "corpus.index"
    meta_path = indexes_dir / "corpus.meta.json"

    if not index_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def load_index_info(indexes_dir: Path) -> dict | None:
    """
    Loads corpus.info.json — the small retrieval-metadata file (model,
    dimension, per-document stats, build time). Returns None if the index
    hasn't been built yet. Step 7 can use this to confirm the query is
    being embedded with the same model the index was built with.
    """
    info_path = indexes_dir / "corpus.info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text(encoding="utf-8"))


def search_index(index, meta: dict, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    query_vector: shape (1, dim), float32 — caller (Step 7) is responsible
    for embedding the query text into this shape using the SAME model as
    embeddings.py (BAAI/bge-small-en-v1.5), or scores will be meaningless.

    Returns the top_k chunks as full dicts, each with "score" (cosine
    similarity, roughly -1 to 1, higher = more similar) and "row_id" (the
    raw FAISS index row) added — row_id is redundant with chunk_id for
    normal use but is useful while debugging retrieval, e.g. confirming two
    different queries hit different rows rather than a stale cached result.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    query_vector = query_vector.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # If top_k > index.ntotal, FAISS pads the extra slots with id -1 and a
    # meaningless score rather than erroring — clamping here means the
    # caller gets exactly as many real results as exist, nothing to filter
    # out downstream, and no surprise about why fewer than top_k came back.
    effective_top_k = max(1, min(top_k, index.ntotal))
    if index.ntotal == 0:
        return []

    scores, ids = index.search(query_vector, effective_top_k)

    results = []
    chunk_ids = meta["chunk_ids"]
    chunks = meta["chunks"]

    for score, row_id in zip(scores[0], ids[0]):
        if row_id == -1:
            continue  # defensive: shouldn't happen now that top_k is clamped
        chunk_id = chunk_ids[row_id]
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        results.append({**chunk, "score": float(score), "row_id": int(row_id)})

    return results
