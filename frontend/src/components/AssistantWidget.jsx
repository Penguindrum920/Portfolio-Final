import { useMemo, useRef, useState } from "react";

const SUGGESTIONS = [
  "about",
  "projects",
  "skills",
  "contact",
  "open projects",
  "play minesweeper"
];

export default function AssistantWidget({ onAction }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([
    {
      role: "assistant",
      text: "Aditya AI online. Ask about projects, skills, or say what to open."
    }
  ]);
  const scrollRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  function pushMessage(role, text) {
    setHistory((prev) => [...prev, { role, text }]);
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  }

  async function sendMessage(message) {
    const query = message.trim();
    if (!query || loading) return;

    setInput("");
    pushMessage("user", query);
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query })
      });

      if (!response.ok) {
        throw new Error("Assistant request failed");
      }

      const data = await response.json();
      pushMessage("assistant", data.message || "I don't have that information yet.");

      if (data.action && data.action !== "NONE" && data.target) {
        onAction?.(data.action, data.target);
      }
    } catch (error) {
      pushMessage("assistant", "Assistant backend is unreachable right now.");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    sendMessage(input);
  }

  return (
    <div className="assistant-root">
      {!open && (
        <button
          type="button"
          className="assistant-fab"
          aria-label="Open Aditya AI"
          onClick={() => setOpen(true)}
        >
          AI
        </button>
      )}

      {open && (
        <section className="assistant-window" role="dialog" aria-label="Aditya AI Assistant">
          <header className="assistant-head">
            <strong>Aditya AI</strong>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close assistant">
              ×
            </button>
          </header>

          <div className="assistant-suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => sendMessage(suggestion)}
                disabled={loading}
              >
                {suggestion}
              </button>
            ))}
          </div>

          <div className="assistant-log" ref={scrollRef}>
            {history.map((entry, index) => (
              <div key={`${entry.role}-${index}`} className={`bubble ${entry.role}`}>
                {entry.text}
              </div>
            ))}
            {loading && <div className="bubble assistant typing">Typing...</div>}
          </div>

          <form className="assistant-form" onSubmit={onSubmit}>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask or command..."
              autoComplete="off"
            />
            <button type="submit" disabled={!canSend}>
              Send
            </button>
          </form>
        </section>
      )}
    </div>
  );
}
