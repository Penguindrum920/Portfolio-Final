import { useMemo, useState } from "react";
import AssistantWidget from "./components/AssistantWidget";

const APPS = [
  { id: "about", label: "About" },
  { id: "projects", label: "Projects" },
  { id: "skills", label: "Skills" },
  { id: "contact", label: "Contact" },
  { id: "resume", label: "Resume" },
  { id: "games", label: "Games" },
  { id: "snake", label: "Snake" },
  { id: "minesweeper", label: "Minesweeper" }
];

function WindowPanel({ appId, onClose }) {
  const title = useMemo(() => APPS.find((x) => x.id === appId)?.label || appId, [appId]);

  return (
    <section className="os-window">
      <header>
        <strong>{title}</strong>
        <button type="button" onClick={() => onClose(appId)} aria-label={`Close ${title}`}>
          ×
        </button>
      </header>
      <div className="os-window-body">
        {appId === "about" && <p>Aditya builds polished, interactive full-stack experiences with a strong frontend focus.</p>}
        {appId === "projects" && <p>Projects include AniVerse, Pegasus, and healthcare-focused portal builds.</p>}
        {appId === "skills" && <p>Core stack: React, Node.js, Python, ML tooling, and strong UI systems work.</p>}
        {appId === "contact" && <p>Use socials or email links to connect. Quickest is GitHub + LinkedIn.</p>}
        {appId === "resume" && <p>Resume module opened. You can wire this panel to an actual PDF viewer.</p>}
        {appId === "games" && <p>Games launcher: ask "play snake" or "play minesweeper".</p>}
        {appId === "snake" && <p>Snake launched. In your current HTML desktop, this routes to the real game page.</p>}
        {appId === "minesweeper" && <p>Minesweeper launched. In your current HTML desktop, this routes to the real game page.</p>}
      </div>
    </section>
  );
}

export default function App() {
  const [openWindows, setOpenWindows] = useState(["about"]);

  function openWindow(target) {
    setOpenWindows((prev) => (prev.includes(target) ? prev : [...prev, target]));
  }

  function closeWindow(target) {
    setOpenWindows((prev) => prev.filter((id) => id !== target));
  }

  function handleAssistantAction(action, target) {
    if (action !== "OPEN_WINDOW") return;
    openWindow(target);
  }

  return (
    <main className="desktop-root">
      <aside className="desktop-icons">
        {APPS.slice(0, 6).map((app) => (
          <button key={app.id} type="button" onClick={() => openWindow(app.id)}>
            {app.label}
          </button>
        ))}
      </aside>

      <section className="desktop-windows">
        {openWindows.map((appId) => (
          <WindowPanel key={appId} appId={appId} onClose={closeWindow} />
        ))}
      </section>

      <footer className="taskbar">
        <span>Portfolio OS</span>
        <span>Aditya AI Ready</span>
      </footer>

      <AssistantWidget onAction={handleAssistantAction} />
    </main>
  );
}
