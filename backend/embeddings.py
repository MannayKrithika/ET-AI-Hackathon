"""
Embeddings — Step 5.

Turns each chunk's text into a fixed-length vector so that, in Step 7,
FAISS can compare vectors by similarity instead of comparing raw words.

Model: BAAI/bge-small-en-v1.5, served via fastembed (ONNX Runtime).
  - Free, runs locally, no API key, no per-call cost or rate limit.
  - 384-dimensional vectors — same shape as before, good quality for RAG
    over technical documents.
  - Deliberately NOT sentence-transformers: that library pulls in torch,
    and torch's own package ships license folders nested so deep that
    installing it fails on Windows with "filename or extension is too
    long" (MAX_PATH, 260 chars) unless the machine has long-path support
    turned on. fastembed uses onnxruntime instead, which has no such
    dependency and installs cleanly on a stock Windows setup.
  - Deliberately NOT Gemini here either: Gemini is reserved for Step 8
    (answer generation). Keeping embeddings local means the retrieval
    side of the pipeline still works even if the Gemini API key or quota
    has a problem during a live demo.

For every {slug}_chunks.json in chunks/, this produces two files in
embeddings/:

    {slug}_embeddings.npy        <- float32 array, shape (num_chunks, 384)
    {slug}_embeddings.meta.json  <- {model, dim, count, chunk_ids}

chunk_ids preserves the exact order of rows in the .npy file, so row i's
vector corresponds to chunk_ids[i]. That mapping is what Step 7 (FAISS)
will use to go from "closest vector" back to "which chunk, which document,
which page."

We embed the whole chunks/ directory into one combined index later (Step 7)
by concatenating these per-document arrays — kept per-document here so a
single re-upload doesn't require re-embedding everything else.

Note: the first time this runs, fastembed downloads the ONNX model
(~130MB) from Hugging Face and caches it locally. That first call needs
internet access; every call after that is fully offline.
"""

import json
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model = None  # loaded once, reused across requests


def get_model():
    """Lazy singleton — the model only loads (and, on first run, downloads)
    the first time it's needed, not at server startup, so
    `uvicorn main:app --reload` stays fast."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    texts -> float32 array, shape (len(texts), EMBEDDING_DIM), L2-normalized.

    Normalizing here means a plain dot product equals cosine similarity,
    which is what Step 7's FAISS index (IndexFlatIP) will expect. bge
    models are trained to output near-unit-norm vectors already, but we
    normalize explicitly rather than rely on that.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    model = get_model()
    vectors = np.array(list(model.embed(texts)), dtype="float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on a pathological empty-text chunk
    vectors = vectors / norms

    return vectors.astype("float32")


def embed_chunks_file(chunks_path: Path, embeddings_dir: Path) -> dict:
    """
    Reads a {slug}_chunks.json, embeds every chunk's text, and writes the
    matching {slug}_embeddings.npy + .meta.json. Returns a small summary
    dict for the API response — never the raw vectors, those are large
    and not useful in a JSON response.
    """
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        return {"count": 0, "dim": EMBEDDING_DIM}

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    vectors = embed_texts(texts)

    slug = chunks_path.stem.removesuffix("_chunks")
    embeddings_dir.mkdir(exist_ok=True)

    npy_path = embeddings_dir / f"{slug}_embeddings.npy"
    meta_path = embeddings_dir / f"{slug}_embeddings.meta.json"

    np.save(npy_path, vectors)
    meta_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "dim": vectors.shape[1] if vectors.size else EMBEDDING_DIM,
        "count": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }, indent=2), encoding="utf-8")

    return {
        "count": len(chunk_ids),
        "dim": vectors.shape[1] if vectors.size else EMBEDDING_DIM,
        "embeddings_file": npy_path.name,
    }
