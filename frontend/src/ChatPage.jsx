import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

function scorePercent(score) {
  if (typeof score !== "number") return 0;
  const pct = score <= 1 ? score * 100 : score;
  return Math.max(2, Math.min(100, Math.round(pct)));
}

export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // {question, answer, grounded, chunks_used, error}
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend() {
    const q = question.trim();
    if (!q || loading) return;

    setLoading(true);
    setQuestion("");

    try {
      const url = `${API_BASE}/ask?question=${encodeURIComponent(q)}&top_k=4`;
      const res = await fetch(url, { method: "POST" });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          question: data.question ?? q,
          answer: data.answer,
          grounded: data.grounded,
          chunks_used: data.chunks_used || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          question: q,
          error: "Unable to reach the model. Check that the backend is running, then try again.",
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const suggestions = [
    "Which equipment is marked Critical?",
    "Summarize the last inspection on Pump P-204",
    "What's the lockout-tagout procedure for the boiler?",
  ];

  return (
    <div className="chat-page">
      <div className="chat-header">
        <p className="eyebrow">Query the knowledge base</p>
        <h1 className="page-title">Ask your documents</h1>
      </div>

      <div className="chat-window">
        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <span className="chat-empty-mark" aria-hidden="true" />
            <p className="chat-empty-title">No questions asked yet</p>
            <p className="chat-empty-copy">
              Ask about anything in the uploaded manuals, drawings, or logs. Answers are
              grounded in retrieved passages, with sources cited below each reply.
            </p>
            <div className="chat-suggestions">
              {suggestions.map((s) => (
                <button
                  key={s}
                  className="chat-suggestion"
                  onClick={() => setQuestion(s)}
                  type="button"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className="chat-turn">
            <div className="chat-query-row">
              <span className="chat-query-tag">Q</span>
              <span className="chat-query-text">{m.question}</span>
            </div>

            {m.error ? (
              <div className="chat-error">
                <span className="chat-error-mark">!</span>
                {m.error}
              </div>
            ) : (
              <div className="chat-response">
                <div className="chat-response-head">
                  <span
                    className={
                      m.grounded ? "grounding-pill grounded" : "grounding-pill ungrounded"
                    }
                  >
                    <span className="grounding-dot" />
                    {m.grounded ? "Grounded in documents" : "General knowledge"}
                  </span>
                </div>

                <p className="chat-answer">{m.answer}</p>

                {m.chunks_used.length > 0 && (
                  <div className="chat-sources">
                    <div className="chat-sources-title">
                      Sources <span className="chat-sources-count">{m.chunks_used.length}</span>
                    </div>
                    <ul className="chat-sources-list">
                      {m.chunks_used.map((c) => (
                        <li key={c.chunk_id} className="chat-source-item">
                          <span className="chunk-id">{c.chunk_id}</span>
                          <span className="chat-source-doc">{c.document_name}</span>
                          <span className="chat-source-loc">
                            {c.source_page != null
                              ? `page ${c.source_page}`
                              : c.section_title || ""}
                          </span>
                          <span className="chat-source-meter" title={`score ${c.score}`}>
                            <span
                              className="chat-source-meter-fill"
                              style={{ width: `${scorePercent(c.score)}%` }}
                            />
                          </span>
                          <span className="chat-source-score">{c.score}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-turn">
            <div className="chat-loading">
              <span className="chat-loading-dot" />
              <span className="chat-loading-dot" />
              <span className="chat-loading-dot" />
              Retrieving and reasoning…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          placeholder="Ask a question about your documents…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={loading || !question.trim()}
        >
          {loading ? "Asking…" : "Send"}
        </button>
      </div>
    </div>
  );
}
