"""
Industrial Knowledge Intelligence - Backend

Pipeline so far:
  1. Receive uploaded files, validate, save into uploads/.
  2. Extract text (PDF, DOCX, XLSX, CSV) into extracted_text/.
  3. Chunk that text using the right strategy per type, save into chunks/.
  4. Embed each chunk into a vector, save into embeddings_store/.
  5. Combine all per-document embeddings into one FAISS index, save into indexes/.

No retrieval-to-answer or Gemini yet. That's next.
"""

import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from extraction import extract_text
from chunking import create_chunks, save_chunks, slugify
from embeddings import embed_chunks_file, embed_texts
from faiss_index import build_index, load_index, search_index
from rag import answer_question
from industrial_entity_extractor import (
    extract_industrial_entities,
    save_entities,
    load_all_entities,
)

load_dotenv()  # reads backend/.env, so GOOGLE_API_KEY doesn't need to be
               # exported in the shell every time — see .env.example

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

EXTRACTED_DIR = BASE_DIR / "extracted_text"
EXTRACTED_DIR.mkdir(exist_ok=True)

CHUNKS_DIR = BASE_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

EMBEDDINGS_DIR = BASE_DIR / "embeddings_store"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

INDEXES_DIR = BASE_DIR / "indexes"
INDEXES_DIR.mkdir(exist_ok=True)

ENTITIES_DIR = BASE_DIR / "entities"
ENTITIES_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_MB = 50

app = FastAPI(title="Industrial Knowledge Intelligence - Upload Service")

# Allow the React dev server (default port 5173 for Vite, 3000 for CRA) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "upload-service"}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Accepts one or more files, saves the valid ones into uploads/,
    and reports per-file success or failure. Nothing else happens here yet.
    """
    results = []

    for file in files:
        original_name = file.filename or "unnamed_file"
        extension = Path(original_name).suffix.lower()

        # Reject unsupported file types
        if extension not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": original_name,
                "status": "rejected",
                "reason": f"Unsupported file type '{extension}'",
            })
            continue

        # Read content and enforce a basic size limit
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            results.append({
                "filename": original_name,
                "status": "rejected",
                "reason": f"File exceeds {MAX_FILE_SIZE_MB}MB limit",
            })
            continue

        # Avoid overwriting files with the same name: prefix with a short uuid
        safe_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
        save_path = UPLOAD_DIR / safe_name

        with open(save_path, "wb") as f:
            f.write(content)

        result = {
            "filename": original_name,
            "saved_as": safe_name,
            "status": "uploaded",
            "size_kb": round(len(content) / 1024, 1),
        }

        # Attempt text extraction right away (PDF, DOCX, XLSX, CSV —
        # PNG/JPG are saved but not yet extracted; that's OCR, later).
        extraction_result = extract_text(save_path)

        txt_path = None
        page_word_counts = None
        if extraction_result is None:
            result["extraction"] = "not_supported_yet"
        elif extraction_result["success"]:
            txt_filename = save_path.stem + ".txt"
            txt_path = EXTRACTED_DIR / txt_filename
            txt_path.write_text(extraction_result["text"], encoding="utf-8")

            page_word_counts = extraction_result.get("page_word_counts")
            if page_word_counts:
                pagemap_path = EXTRACTED_DIR / (save_path.stem + ".pagemap.json")
                pagemap_path.write_text(json.dumps(page_word_counts), encoding="utf-8")

            result["extraction"] = "success"
            result["pages_processed"] = extraction_result["pages_processed"]
            result["char_count"] = extraction_result["char_count"]
            result["extracted_text_file"] = txt_filename
        else:
            result["extraction"] = "failed"
            result["extraction_error"] = extraction_result["error"]

        # Industrial Entity Extraction: runs right after text extraction and
        # before chunking, for every file type it supports (xlsx/csv always;
        # pdf/docx once extraction has produced a .txt to read). Regex/
        # column-based, no LLM call, so it doesn't add meaningful latency to
        # the upload. A failure here never blocks the rest of the pipeline —
        # chunking/embeddings/FAISS/Gemini all continue exactly as before.
        if extension in (".xlsx", ".csv") or txt_path is not None:
            try:
                entity_result = extract_industrial_entities(save_path, txt_path)
                if entity_result:
                    entities_path = save_entities(entity_result, save_path, ENTITIES_DIR)
                    result["entity_extraction"] = "success"
                    result["document_type_detected"] = entity_result["document_type"]
                    result["entities_file"] = entities_path.name
                else:
                    result["entity_extraction"] = "not_supported_yet"
            except Exception as e:
                result["entity_extraction"] = "failed"
                result["entity_extraction_error"] = str(e)

        # Chunking: only makes sense once extraction succeeded (or, for
        # CSV/XLSX, regardless — those chunk straight from the original
        # rows rather than the flattened extracted text).
        chunkable = extension in (".csv", ".xlsx") or (extraction_result and extraction_result["success"])
        if chunkable:
            try:
                chunks, num_filtered = create_chunks(save_path, txt_path, page_word_counts)
                if chunks:
                    chunks_path = save_chunks(chunks, save_path, CHUNKS_DIR)
                    result["chunking"] = "success"
                    result["chunk_count"] = len(chunks)
                    result["chunks_filtered_out"] = num_filtered
                    result["chunks_file"] = chunks_path.name
                    print(f"{original_name} -> {len(chunks)} chunks created ({num_filtered} filtered out)")

                    # Embedding: only makes sense once chunks exist.
                    try:
                        embed_summary = embed_chunks_file(chunks_path, EMBEDDINGS_DIR)
                        result["embedding"] = "success"
                        result["embedding_count"] = embed_summary["count"]
                        result["embedding_dim"] = embed_summary["dim"]
                        print(f"{original_name} -> {embed_summary['count']} chunks embedded ({embed_summary['dim']}-dim)")
                    except Exception as e:
                        result["embedding"] = "failed"
                        result["embedding_error"] = str(e)
                else:
                    result["chunking"] = "no_chunks_produced"
            except Exception as e:
                result["chunking"] = "failed"
                result["chunking_error"] = str(e)

        results.append(result)

    uploaded_count = sum(1 for r in results if r["status"] == "uploaded")
    any_embedded = any(r.get("embedding") == "success" for r in results)

    index_summary = None
    if any_embedded:
        # Rebuild once per batch (not once per file) — a query needs to search
        # across every document, so the index always reflects the full corpus,
        # not just whatever was just uploaded.
        try:
            index_summary = build_index(CHUNKS_DIR, EMBEDDINGS_DIR, INDEXES_DIR)
            print(f"FAISS index rebuilt -> {index_summary.get('total_chunks', 0)} chunks total")
        except Exception as e:
            index_summary = {"success": False, "reason": str(e)}

    return {
        "message": f"{uploaded_count}/{len(files)} file(s) uploaded successfully.",
        "results": results,
        "index": index_summary,
    }

from entity_extractor import extract_entities
from knowledge_graph import build_knowledge_graph
from compliance_checker import check_compliance

@app.get("/entities")
def list_entities():
    """
    Industrial Entity Extraction deliverable: returns every document's
    extracted entities, grouped by document. Reads whatever is currently
    saved under backend/entities/ — one file per uploaded document, written
    automatically during /upload (and /rechunk-all, for files uploaded
    before this feature existed).
    """
    results = load_all_entities(ENTITIES_DIR)
    return {"count": len(results), "documents": results}


@app.get("/files")
def list_uploaded_files():
    """Simple listing of what's currently in uploads/ — useful for sanity-checking."""
    files = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.is_file():
            files.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
    return {"count": len(files), "files": files}


@app.get("/documents")
def list_documents():
    """
    Powers the "Uploaded Documents" page: every saved file, plus whether
    extracted text and chunks exist for it yet.
    """
    documents = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        txt_path = EXTRACTED_DIR / (f.stem + ".txt")
        clean_stem = re.sub(r"^[0-9a-f]{8}_", "", f.stem)
        slug = slugify(clean_stem)
        chunks_path = CHUNKS_DIR / f"{slug}_chunks.json"
        embed_meta_path = EMBEDDINGS_DIR / f"{slug}_embeddings.meta.json"
        documents.append({
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "has_extracted_text": txt_path.exists(),
            "has_chunks": chunks_path.exists(),
            "chunk_count": len(json.loads(chunks_path.read_text())) if chunks_path.exists() else 0,
            "has_embeddings": embed_meta_path.exists(),
            "embedding_count": json.loads(embed_meta_path.read_text())["count"] if embed_meta_path.exists() else 0,
        })
    return {"count": len(documents), "documents": documents}


@app.get("/documents/{filename}/text")
def get_document_text(filename: str):
    """Returns the extracted text for a given uploaded filename, if available."""
    source_path = UPLOAD_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File not found in uploads/")

    txt_path = EXTRACTED_DIR / (source_path.stem + ".txt")
    if not txt_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No extracted text available for this file yet.",
        )

    text = txt_path.read_text(encoding="utf-8")
    return {
        "filename": filename,
        "char_count": len(text),
        "text": text,
    }


@app.get("/documents/{filename}/chunks")
def get_document_chunks(filename: str):
    """Returns the chunks generated for a given uploaded filename, if available."""
    source_path = UPLOAD_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File not found in uploads/")

    clean_stem = re.sub(r"^[0-9a-f]{8}_", "", source_path.stem)
    chunks_path = CHUNKS_DIR / f"{slugify(clean_stem)}_chunks.json"
    if not chunks_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No chunks available for this file yet.",
        )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return {"filename": filename, "chunk_count": len(chunks), "chunks": chunks}


@app.get("/documents/{filename}/embeddings")
def get_document_embeddings(filename: str):
    """
    Stats only — never the raw vectors, they're not useful in a JSON
    response and would be huge. Confirms the embedding step actually ran
    and gives dimension/count for sanity-checking.
    """
    source_path = UPLOAD_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File not found in uploads/")

    clean_stem = re.sub(r"^[0-9a-f]{8}_", "", source_path.stem)
    meta_path = EMBEDDINGS_DIR / f"{slugify(clean_stem)}_embeddings.meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No embeddings available for this file yet.",
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "filename": filename,
        "model": meta["model"],
        "dim": meta["dim"],
        "count": meta["count"],
    }


@app.post("/reembed-all")
def reembed_all():
    """
    Utility route: re-embeds every chunks file already sitting in chunks/.
    Useful after changing the embedding model, or for chunks created before
    embeddings existed — no need to re-upload or re-chunk anything.
    """
    summary = []
    for chunks_path in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        try:
            result = embed_chunks_file(chunks_path, EMBEDDINGS_DIR)
            print(f"{chunks_path.name} -> {result['count']} chunks embedded ({result['dim']}-dim)")
            summary.append({"chunks_file": chunks_path.name, **result})
        except Exception as e:
            summary.append({"chunks_file": chunks_path.name, "error": str(e)})

    return {"processed": len(summary), "results": summary}


@app.post("/rechunk-all")
def rechunk_all():
    """
    Utility route: re-runs chunking for every file already sitting in
    uploads/. Useful for files uploaded before chunking existed, or after
    tweaking the chunking strategy — no need to re-upload anything.
    """
    summary = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue

        extension = f.suffix.lower()
        txt_path = EXTRACTED_DIR / (f.stem + ".txt")
        chunkable = extension in (".csv", ".xlsx") or txt_path.exists()
        if not chunkable:
            continue

        pagemap_path = EXTRACTED_DIR / (f.stem + ".pagemap.json")
        page_word_counts = (
            json.loads(pagemap_path.read_text()) if pagemap_path.exists() else None
        )

        if extension in (".xlsx", ".csv") or txt_path.exists():
            try:
                entity_result = extract_industrial_entities(
                    f, txt_path if txt_path.exists() else None
                )
                if entity_result:
                    save_entities(entity_result, f, ENTITIES_DIR)
            except Exception:
                pass  # best-effort backfill; doesn't block re-chunking

        try:
            chunks, num_filtered = create_chunks(
                f, txt_path if txt_path.exists() else None, page_word_counts
            )
            if chunks:
                chunks_path = save_chunks(chunks, f, CHUNKS_DIR)
                print(f"{f.name} -> {len(chunks)} chunks created ({num_filtered} filtered out)")
                entry = {
                    "filename": f.name,
                    "chunk_count": len(chunks),
                    "chunks_filtered_out": num_filtered,
                }
                try:
                    embed_summary = embed_chunks_file(chunks_path, EMBEDDINGS_DIR)
                    entry["embedding_count"] = embed_summary["count"]
                    entry["embedding_dim"] = embed_summary["dim"]
                except Exception as e:
                    entry["embedding_error"] = str(e)
                summary.append(entry)
        except Exception as e:
            summary.append({"filename": f.name, "error": str(e)})

    index_summary = None
    if any("embedding_count" in s for s in summary):
        try:
            index_summary = build_index(CHUNKS_DIR, EMBEDDINGS_DIR, INDEXES_DIR)
        except Exception as e:
            index_summary = {"success": False, "reason": str(e)}

    return {"processed": len(summary), "results": summary, "index": index_summary}


@app.post("/build-index")
def rebuild_index():
    """
    Step 6: manually (re)builds the combined FAISS index from whatever is
    currently in embeddings_store/. Normally this happens automatically
    after every upload batch — this route exists for cases where you've
    changed embeddings out from under the server (e.g. ran /reembed-all)
    and want the index to reflect that immediately, without re-uploading.
    """
    result = build_index(CHUNKS_DIR, EMBEDDINGS_DIR, INDEXES_DIR)
    return result


@app.get("/index/status")
def index_status():
    """Quick sanity check: does an index exist, and how big is it."""
    index, meta = load_index(INDEXES_DIR)
    if index is None:
        return {"exists": False, "total_chunks": 0}
    return {
        "exists": True,
        "total_chunks": meta["total_chunks"],
        "dim": meta["dim"],
    }


@app.post("/search")
def search(query: str, top_k: int = 5):
    """
    Step 6 sanity-check route — NOT the final /ask endpoint (that's Step 8,
    once Gemini is wired in). This just proves retrieval works: embed the
    query with the same model used for chunks, search the FAISS index,
    return the raw top-k chunks with similarity scores. Useful for
    confirming good chunks actually surface before adding an LLM on top.
    """
    index, meta = load_index(INDEXES_DIR)
    if index is None:
        raise HTTPException(status_code=404, detail="No index built yet. Upload documents or call /build-index first.")

    query_vector = embed_texts([query])
    if query_vector.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Query could not be embedded.")

    results = search_index(index, meta, query_vector[0], top_k=top_k)
    return {"query": query, "top_k": top_k, "results": results}


@app.post("/ask")
def ask(question: str, top_k: int = 4):
    """
    Step 7: the actual RAG deliverable. Retrieves the top-k relevant chunks
    from FAISS and sends them to Gemini as context, so the answer comes from
    your documents rather than Gemini's own general knowledge.

    Unlike /search, this returns a natural-language answer plus the sources
    it was grounded in — this is what Step 8's chat interface will call.
    """
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    try:
        result = answer_question(question, INDEXES_DIR, top_k=top_k)
    except RuntimeError as e:
        # Missing/misconfigured API key — a config problem, not a bad
        # request, so 500 rather than 400.
        raise HTTPException(status_code=500, detail=str(e))

    # Step 9: run entity extraction over the generated answer, so the
    # frontend gets structured entities alongside the natural-language
    # answer without a second round-trip. Only bother when the answer was
    # actually grounded in something — an ungrounded "not found in the
    # documents" message has nothing worth extracting.
    entities = []
    if result.get("grounded"):
        try:
            entities = extract_entities(result["answer"])
        except RuntimeError:
            # Same missing-API-key case as above, but entity extraction is
            # a bonus on top of the answer, not the main deliverable — don't
            # fail the whole /ask call over it, just return no entities.
            entities = []

    result["entities"] = entities
    return result


@app.post("/extract_entities")
def extract_entities_route(text: str):
    """
    Step 9: standalone entity extraction endpoint. Takes arbitrary text
    (a document excerpt, a chunk, a previous answer, anything) and returns
    the structured entities Gemini finds in it, scoped to ENTITY_TYPES.

    This is the building block Step 10 (Knowledge Graph) will call
    repeatedly to turn documents/answers into graph nodes and edges.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    try:
        entities = extract_entities(text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"entities": entities}


@app.post("/build_graph")
def build_graph_route(text: str):
    """
    Step 10: takes arbitrary text, runs Step 9's entity extraction over it,
    then builds a knowledge graph (nodes + edges) from the result.

    Returns the raw entities too, alongside the graph, so the frontend/
    Swagger can see exactly what the graph was built from without a
    separate call to /extract_entities.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    try:
        entities = extract_entities(text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    graph = build_knowledge_graph(entities)
    return {"entities": entities, "graph": graph}


@app.post("/compliance")
def compliance(text: str):
    """
    Step 11: the Compliance Checker deliverable. Takes arbitrary industrial
    text, runs Step 9's entity extraction over it, then checks the resulting
    entities against the small rule set in compliance_rules.py.

    Returns the raw entities too, alongside the compliance report, so the
    frontend/Swagger can see exactly what the report was built from without
    a separate call to /extract_entities.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    try:
        entities = extract_entities(text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    report = check_compliance(entities)
    return {"entities": entities, "compliance": report}
