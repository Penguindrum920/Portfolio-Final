# Aditya AI (RAG + Agentic Assistant)

This workspace now includes a full-stack implementation with the requested structure:

- frontend: React desktop-style UI with floating Aditya AI widget
- backend: FastAPI API with RAG + action detection
- data: portfolio knowledge documents for grounding

## Important Security Note

Your OpenAI key was shared in chat. Rotate/revoke it immediately in your OpenAI dashboard and use a new key in environment variables. Do not hardcode keys in files.

## Architecture

1. User sends message from frontend assistant widget.
2. Backend checks command intent first:
   - open/show/launch/play/go to + target
   - returns structured action payload
3. If not a command, backend runs RAG:
   - lazy-loads index from data files
   - chunks text
   - computes embeddings
   - stores vectors in local FAISS cache
   - retrieves top-k chunks
   - answers using only retrieved context
4. Frontend consumes:
   {
     "message": "Opening projects...",
     "action": "OPEN_WINDOW",
     "target": "projects"
   }

## Setup

## 1) Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Update backend/.env with your fresh key:

```env
OPENAI_API_KEY=your_new_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
TOP_K=4
MIN_SIMILARITY=0.18
CACHE_SIZE=100
FRONTEND_ORIGIN=http://localhost:5173
```

Run backend:

```bash
uvicorn main:app --reload --port 8000
```

## 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API Contract

POST /chat

Request:

```json
{ "message": "open projects" }
```

Response (action):

```json
{
  "message": "Opening projects...",
  "action": "OPEN_WINDOW",
  "target": "projects",
  "sources": []
}
```

Response (RAG answer):

```json
{
  "message": "Aditya has built AniVerse, Pegasus, and a Parkinson Disease Assessment Portal.",
  "action": "NONE",
  "target": null,
  "sources": ["data/projects.json", "data/resume.md"]
}
```

## Notes

- RAG is strict-grounded. If context is missing, response is:
  I don't have that information yet.
- Query results are cached (LRU) for speed.
- Embeddings/index are lazy-loaded and persisted in backend/cache/rag_index.pkl.

## Integrating with Existing code.html Desktop

You already have native open-window functions in code.html.
Map backend action payloads to your existing methods:

- target: projects -> openExplorer("portfolio")
- target: resume -> openItem(resumeItem)
- target: games -> openGamesLauncher()
- target: minesweeper -> openGameWindow(minesweeperItem)
- target: snake -> openGameWindow(snakeItem)
- target: contact -> openExplorer("socials")

If you want, this can be wired directly into your current HTML desktop as a second step.
