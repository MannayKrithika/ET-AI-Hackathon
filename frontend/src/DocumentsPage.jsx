import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [view, setView] = useState("text"); // text | chunks | embeddings
  const [extractedText, setExtractedText] = useState("");
  const [chunks, setChunks] = useState([]);
  const [embeddingInfo, setEmbeddingInfo] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | unavailable
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      const data = await res.json();
      setDocuments(data.documents);
    } catch {
      setLoadError("Could not reach the backend at " + API_BASE);
    }
  }

  async function handleSelectDoc(doc, requestedView) {
    setSelectedDoc(doc.name);
    setView(requestedView);
    setExtractedText("");
    setChunks([]);
    setEmbeddingInfo(null);

    if (requestedView === "text") {
      if (!doc.has_extracted_text) {
        setStatus("unavailable");
        return;
      }
      setStatus("loading");
      try {
        const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(doc.name)}/text`);
        if (!res.ok) return setStatus("unavailable");
        const data = await res.json();
        setExtractedText(data.text);
        setStatus("ready");
      } catch {
        setStatus("unavailable");
      }
    } else if (requestedView === "chunks") {
      if (!doc.has_chunks) {
        setStatus("unavailable");
        return;
      }
      setStatus("loading");
      try {
        const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(doc.name)}/chunks`);
        if (!res.ok) return setStatus("unavailable");
        const data = await res.json();
        setChunks(data.chunks);
        setStatus("ready");
      } catch {
        setStatus("unavailable");
      }
    } else {
      if (!doc.has_embeddings) {
        setStatus("unavailable");
        return;
      }
      setStatus("loading");
      try {
        const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(doc.name)}/embeddings`);
        if (!res.ok) return setStatus("unavailable");
        const data = await res.json();
        setEmbeddingInfo(data);
        setStatus("ready");
      } catch {
        setStatus("unavailable");
      }
    }
  }

  return (
    <div className="docs-page">
      <h2>Uploaded Documents</h2>

      {loadError && <p className="error">{loadError}</p>}

      <div className="docs-layout">
        <ul className="docs-list">
          {documents.length === 0 && !loadError && (
            <li className="docs-empty">No documents uploaded yet.</li>
          )}
          {documents.map((doc) => (
            <li key={doc.name} className={doc.name === selectedDoc ? "docs-item active" : "docs-item"}>
              <span className="docs-name">{doc.name}</span>
              <div className="docs-actions">
                <button
                  className={doc.name === selectedDoc && view === "text" ? "docs-tab active" : "docs-tab"}
                  onClick={() => handleSelectDoc(doc, "text")}
                  disabled={!doc.has_extracted_text}
                >
                  Text
                </button>
                <button
                  className={doc.name === selectedDoc && view === "chunks" ? "docs-tab active" : "docs-tab"}
                  onClick={() => handleSelectDoc(doc, "chunks")}
                  disabled={!doc.has_chunks}
                >
                  Chunks {doc.has_chunks ? `(${doc.chunk_count})` : ""}
                </button>
                <button
                  className={doc.name === selectedDoc && view === "embeddings" ? "docs-tab active" : "docs-tab"}
                  onClick={() => handleSelectDoc(doc, "embeddings")}
                  disabled={!doc.has_embeddings}
                >
                  Embeddings {doc.has_embeddings ? `(${doc.embedding_count})` : ""}
                </button>
              </div>
            </li>
          ))}
        </ul>

        <div className="docs-preview">
          {!selectedDoc && <p className="docs-hint">Select a document to view its extracted text or chunks.</p>}

          {selectedDoc && status === "loading" && <p className="docs-hint">Loading…</p>}

          {selectedDoc && status === "unavailable" && (
            <p className="docs-hint">
              No {view === "text" ? "extracted text" : view === "chunks" ? "chunks" : "embeddings"} available for{" "}
              <strong>{selectedDoc}</strong> yet.
            </p>
          )}

          {selectedDoc && status === "ready" && view === "text" && (
            <>
              <h3>{selectedDoc}</h3>
              <pre className="extracted-text">{extractedText}</pre>
            </>
          )}

          {selectedDoc && status === "ready" && view === "chunks" && (
            <>
              <h3>{selectedDoc} — {chunks.length} chunks</h3>
              <div className="chunk-list">
                {chunks.map((c) => (
                  <div key={c.chunk_id} className="chunk-card">
                    <div className="chunk-meta">
                      <span className="chunk-id">{c.chunk_id}</span>
                      {c.section_title && <span className="chunk-section">{c.section_title}</span>}
                      <span className="chunk-words">{c.word_count} words</span>
                    </div>
                    <pre className="chunk-text">{c.text}</pre>
                  </div>
                ))}
              </div>
            </>
          )}

          {selectedDoc && status === "ready" && view === "embeddings" && embeddingInfo && (
            <>
              <h3>{selectedDoc} — embeddings</h3>
              <div className="embedding-stats">
                <div><strong>Model:</strong> {embeddingInfo.model}</div>
                <div><strong>Dimensions:</strong> {embeddingInfo.dim}</div>
                <div><strong>Vectors:</strong> {embeddingInfo.count}</div>
              </div>
              <p className="docs-hint">
                Raw vectors aren't shown here — they're not human-readable. This confirms
                every chunk has a corresponding {embeddingInfo.dim}-dimensional vector, ready for
                FAISS indexing in the next step.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
