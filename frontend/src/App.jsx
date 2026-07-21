import { useState, useRef } from "react";
import DocumentsPage from "./DocumentsPage.jsx";
import ChatPage from "./ChatPage.jsx";

const API_BASE = "http://localhost:8000";
const ACCEPTED = ".pdf,.docx,.xlsx,.csv,.png,.jpg,.jpeg";

const NAV_ITEMS = [
  { id: "upload", label: "Upload" },
  { id: "chat", label: "Chat" },
  { id: "documents", label: "Documents" },
];

function Shell({ page, setPage, children }) {
  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <div className="brand-text">
            <span className="brand-name">Industrial Knowledge Intelligence</span>
            <span className="brand-sub">Unified asset &amp; operations brain</span>
          </div>
        </div>
        <nav className="toplinks">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "toplink active" : "toplink"}
              onClick={() => setPage(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="shell-main">{children}</main>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("upload"); // upload | documents | chat
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadResults, setUploadResults] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | uploading | done | error
  const [errorMessage, setErrorMessage] = useState("");
  const fileInputRef = useRef(null);

  function handleChooseFiles(e) {
    setSelectedFiles(Array.from(e.target.files));
    setUploadResults([]);
    setStatus("idle");
    setErrorMessage("");
  }

  async function handleUpload() {
    if (selectedFiles.length === 0) return;

    setStatus("uploading");
    setErrorMessage("");

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      setUploadResults(data.results);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setErrorMessage(
        "Could not reach the backend. Is it running at " + API_BASE + "?"
      );
    }
  }

  function resetForm() {
    setSelectedFiles([]);
    setUploadResults([]);
    setStatus("idle");
    setErrorMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  if (page === "documents") {
    return (
      <Shell page={page} setPage={setPage}>
        <div className="page-panel">
          <DocumentsPage />
        </div>
      </Shell>
    );
  }

  if (page === "chat") {
    return (
      <Shell page={page} setPage={setPage}>
        <div className="page-panel page-panel-chat">
          <ChatPage />
        </div>
      </Shell>
    );
  }

  return (
    <Shell page={page} setPage={setPage}>
      <div className="page-panel page-panel-narrow">
        <p className="eyebrow">Step 01 — Ingest</p>
        <h1 className="page-title">Upload plant documents</h1>

        <div className="upload-box">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPTED}
            onChange={handleChooseFiles}
            id="file-input"
          />
          <label htmlFor="file-input" className="choose-btn">
            Choose files
          </label>

          {selectedFiles.length > 0 && (
            <ul className="selected-list">
              {selectedFiles.map((f, i) => (
                <li key={i}>{f.name}</li>
              ))}
            </ul>
          )}

          <p className="supported">Supported: PDF · DOCX · XLSX · CSV · PNG · JPG</p>

          <div className="upload-actions">
            <button
              className="upload-btn"
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || status === "uploading"}
            >
              {status === "uploading" ? "Uploading…" : "Upload"}
            </button>

            {selectedFiles.length > 0 && status !== "uploading" && (
              <button className="reset-btn" onClick={resetForm}>
                Clear
              </button>
            )}
          </div>
        </div>

        {status === "error" && <p className="error">{errorMessage}</p>}

        {uploadResults.length > 0 && (
          <div className="results">
            {uploadResults.map((r, i) => (
              <div key={i} className="result-block">
                <div className={r.status === "uploaded" ? "result-ok" : "result-fail"}>
                  {r.status === "uploaded" ? "✓" : "✗"} {r.filename}
                  {r.status !== "uploaded" && <span className="reason"> — {r.reason}</span>}
                </div>

                {r.status === "uploaded" && r.extraction === "success" && (
                  <div className="result-sub result-ok">
                    ✓ Text extracted
                    <div className="extract-stats">
                      {r.pages_processed != null && <>Pages processed: {r.pages_processed} · </>}
                      Characters extracted: {r.char_count.toLocaleString()}
                    </div>
                  </div>
                )}

                {r.status === "uploaded" && r.chunking === "success" && (
                  <div className="result-sub result-ok">
                    ✓ {r.chunk_count} chunk{r.chunk_count === 1 ? "" : "s"} created
                  </div>
                )}

                {r.status === "uploaded" && r.extraction === "failed" && (
                  <div className="result-sub result-fail">
                    ✗ Text extraction failed — {r.extraction_error}
                  </div>
                )}

                {r.status === "uploaded" && r.extraction === "not_supported_yet" && (
                  <div className="result-sub result-pending">
                    Saved, but text extraction for this file type isn't built yet.
                  </div>
                )}

                {r.status === "uploaded" && r.embedding === "success" && (
                  <div className="result-sub result-ok">
                    ✓ {r.embedding_count} chunk{r.embedding_count === 1 ? "" : "s"} embedded ({r.embedding_dim}-dim)
                  </div>
                )}

                {r.status === "uploaded" && r.embedding === "failed" && (
                  <div className="result-sub result-fail">
                    ✗ Embedding failed — {r.embedding_error}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </Shell>
  );
}
