import hashlib
import json
import os
import pickle
import re
from difflib import get_close_matches
from collections import OrderedDict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = CACHE_DIR / "rag_index.pkl"
PROJECTS_FILE = DATA_DIR / "projects.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.18"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

APP_SYSTEM_PROMPT = (
    "You are Aditya Kaushik's personal assistant embedded in his portfolio OS. "
    "Answer only questions about Aditya, his portfolio, projects, skills, contact details, "
    "and the portfolio OS interface. You are concise, direct, and naturally confident. "
    "Do not say 'As an AI'. Do not invent dates, rankings, credentials, private details, "
    "or implementation claims that are not in the provided context. Format responses as short "
    "paragraphs or clean bullets. Do not dump raw source text, OCR junk, or markdown heading syntax. "
    "If the answer is missing or unrelated to the portfolio, reply exactly: I don't have that information yet."
)

ACTION_TARGETS = {
    "about": "about",
    "projects": "projects",
    "project": "projects",
    "skills": "skills",
    "contact": "contact",
    "socials": "contact",
    "resume": "resume",
    "minesweeper": "minesweeper",
    "snake": "snake",
    "games": "games"
}

ACTION_ALIASES = {
    "about": ["about", "about me", "profile"],
    "projects": ["projects", "project", "portfolio", "work"],
    "skills": ["skills", "tech stack", "stack", "technologies"],
    "contact": ["contact", "socials", "linkedin", "github", "mail", "email", "phone"],
    "resume": ["resume", "cv"],
    "games": ["games", "game"],
    "snake": ["snake"],
    "minesweeper": ["minesweeper", "mine sweeper"]
}

ACTION_VERBS = [
    "open", "show", "launch", "play", "go to", "start", "pull up", "bring up", "take me", "navigate"
]

STOP_WORDS = {
    "the", "a", "an", "is", "are", "am", "to", "for", "and", "or", "of", "in", "on", "at", "about",
    "what", "which", "tell", "me", "you", "your", "can", "do", "have", "with", "that", "this", "it"
}

PROJECT_FIELD_ALIASES = {
    "web": ["web", "frontend", "front end", "backend", "full stack", "react", "javascript", "html", "css", "node", "express"],
    "ai": ["ai", "ml", "machine learning", "nlp", "llm", "rag", "tensorflow", "pytorch", "opencv"],
    "data": ["data", "database", "db", "sql", "postgres", "mongodb", "analytics"],
    "health": ["health", "medical", "healthcare", "disease", "clinical"],
    "creative": ["creative", "design", "three", "three.js", "music", "animation", "motion", "gsap", "particles"]
}

SKILL_QUERY_TERMS = {
    "skill", "skills", "stack", "tech", "technology", "technologies", "language", "languages",
    "framework", "frameworks", "tool", "tools", "database", "databases", "ml", "ai", "web"
}

CONTACT_QUERY_TERMS = {
    "contact", "email", "mail", "gmail", "github", "linkedin", "phone", "whatsapp", "social", "socials"
}

PORTFOLIO_QUERY_TERMS = {
    "portfolio", "website", "desktop", "windows", "os", "interface", "ui", "feature", "features",
    "taskbar", "start", "terminal", "notepad", "browser", "chrome", "store", "recycle", "wallpaper",
    "calendar", "dark", "light", "mobile", "phone", "games", "snake", "minesweeper", "aditya ai",
    "assistant", "chatbot"
}

RAG_SCOPE_TERMS = (
    SKILL_QUERY_TERMS
    | CONTACT_QUERY_TERMS
    | PORTFOLIO_QUERY_TERMS
    | {
        "aditya", "kaushik", "about", "resume", "cv", "education", "btech", "college",
        "student", "bengaluru", "location", "based", "background", "experience", "paper",
        "publication", "hobby", "hobbies", "anime", "manga", "project", "projects",
        "aniverse", "pegasus", "parkinson", "healthcare", "medical", "three.js", "fastapi"
    }
)

SESSION_STATE: dict[str, Any] = {
    "last_project_list": []
}


class ChatRequest(BaseModel):
    message: str


class Chunk(BaseModel):
    text: str
    source: str


class QueryCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        if key not in self.items:
            return None
        value = self.items.pop(key)
        self.items[key] = value
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        if key in self.items:
            self.items.pop(key)
        self.items[key] = value
        if len(self.items) > self.max_size:
            self.items.popitem(last=False)


cache = QueryCache(CACHE_SIZE)


def safe_read_projects() -> list[dict[str, Any]]:
    if not PROJECTS_FILE.exists():
        return []
    try:
        payload = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        return []
    except Exception:
        return []


def pretty_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.replace("## ", "")
    cleaned = cleaned.replace("### ", "")
    return cleaned.strip()


def format_sectioned(title: str, sections: list[tuple[str, str | list[str]]]) -> str:
    lines = [title.strip()]
    for heading, content in sections:
        if not content:
            continue
        lines.extend(["", heading.strip()])
        if isinstance(content, list):
            lines.extend(f"- {str(item).strip()}" for item in content if str(item).strip())
        else:
            lines.append(str(content).strip())
    return pretty_text("\n".join(lines))


def format_project_line(project: dict[str, Any], index: int) -> str:
    name = str(project.get("name", f"Project {index}"))
    desc = str(project.get("description", "No description yet.")).strip()
    short_desc = desc[:150].rstrip() + ("..." if len(desc) > 150 else "")
    return f"{index}. {name}: {short_desc}"


def safe_read_text(path: Path) -> str:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def contains_any(message: str, terms: set[str] | list[str]) -> bool:
    lower = message.lower()
    return any(term in lower for term in terms)


def in_portfolio_scope(message: str) -> bool:
    return contains_any(message, RAG_SCOPE_TERMS)


def markdown_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current and line:
            sections[current].append(line)
    return sections


def clean_bullet(line: str) -> str:
    return re.sub(r"^[-*]\s*", "", line).strip()


def format_project_detail(project: dict[str, Any]) -> str:
    name = str(project.get("name", "Project"))
    desc = str(project.get("description", "No description yet.")).strip()
    problem = str(project.get("problem", "")).strip()
    role = str(project.get("role", "Not specified")).strip()
    stack = project.get("stack", [])
    stack_text = ", ".join(stack) if isinstance(stack, list) and stack else str(stack or "Not listed")
    features = project.get("features", [])
    links = project.get("links", {})

    sections: list[tuple[str, str | list[str]]] = [
        ("Overview", desc),
        ("Why It Matters", problem),
    ]
    if isinstance(features, list) and features:
        sections.append(("Key Features", [str(feature) for feature in features[:5]]))
    sections.extend([
        ("Tech Stack", stack_text),
        ("Aditya's Role", role),
    ])
    if isinstance(links, dict) and links:
        link_text = ", ".join(f"{key}: {value}" for key, value in links.items())
        sections.append(("Links", link_text))
    return format_sectioned(name, sections)


def project_names(project: dict[str, Any]) -> list[str]:
    names = [str(project.get("name", ""))]
    aliases = project.get("aliases", [])
    if isinstance(aliases, list):
        names.extend(str(alias) for alias in aliases)
    return [name for name in names if name]


def find_project_in_message(message: str, projects: list[dict[str, Any]], allow_fuzzy: bool = False) -> dict[str, Any] | None:
    lower = message.lower()
    for project in projects:
        for name in project_names(project):
            if name.lower() in lower:
                return project

    if not allow_fuzzy:
        return None

    tokens = set(tokenize(message))
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for project in projects:
        haystack = " ".join([
            " ".join(project_names(project)),
            str(project.get("description", "")),
            str(project.get("problem", "")),
            str(project.get("role", "")),
            " ".join(project.get("category", [])) if isinstance(project.get("category"), list) else "",
            " ".join(project.get("stack", [])) if isinstance(project.get("stack"), list) else "",
            " ".join(project.get("features", [])) if isinstance(project.get("features"), list) else ""
        ])
        overlap = len(tokens & set(tokenize(haystack)))
        if overlap > best[0]:
            best = (float(overlap), project)
    return best[1] if best[0] >= 2 else None


def about_response(message: str) -> str | None:
    lower = message.lower()
    about_triggers = [
        "about me",
        "about aditya",
        "who are you",
        "who is aditya",
        "introduce yourself",
        "introduce aditya",
        "tell me about yourself",
        "aditya kaushik",
        "background",
        "education",
        "study",
        "college",
        "where is aditya",
        "location",
        "based",
        "hobby",
        "hobbies"
    ]
    is_about_prompt = lower.strip() in {"about", "about me", "who are you"} or any(trigger in lower for trigger in about_triggers)
    if not is_about_prompt:
        return None

    resume_text = safe_read_text(DATA_DIR / "resume.md")
    contact_text = safe_read_text(DATA_DIR / "contact.md")

    summary = "Computer Science student focused on full-stack development, machine learning applications, and polished interactive experiences."
    summary_match = re.search(r"## Summary\s*(.+?)(?:##|$)", resume_text, flags=re.IGNORECASE | re.DOTALL)
    if summary_match:
        candidate = " ".join(summary_match.group(1).split())
        if candidate:
            summary = candidate

    github = re.search(r"GitHub:\s*(.+)", contact_text)
    linkedin = re.search(r"LinkedIn:\s*(.+)", contact_text)

    highlights = [
        "Core focus: full-stack builds, machine learning applications, UI systems, and practical AI integration.",
        "Current interest: combining ML projects with creative web technologies like Three.js.",
        "Projects include AniVerse, Pegasus, and a Parkinson Disease Assessment Portal."
    ]
    if "education" in lower or "study" in lower or "college" in lower:
        highlights.append("Education listed in the portfolio: B.Tech in Computer Science, ongoing.")
    if "where" in lower or "location" in lower or "based" in lower:
        highlights.append("Location listed in the portfolio: Bengaluru.")
    if "hobb" in lower or "anime" in lower or "manga" in lower or "game" in lower:
        highlights.append("Hobbies listed in the portfolio: anime, manga, and video games.")

    contact = []
    if github:
        contact.append(f"GitHub: {github.group(1).strip()}")
    if linkedin:
        contact.append(f"LinkedIn: {linkedin.group(1).strip()}")

    return format_sectioned(
        "About Aditya",
        [
            ("Summary", summary),
            ("Highlights", highlights),
            ("Links", contact),
        ],
    )


def resolve_followup_project_reference(message: str, projects: list[dict[str, Any]]) -> dict[str, Any] | None:
    lower = message.lower()
    remembered = SESSION_STATE.get("last_project_list") or []
    if not remembered:
        return None

    ordinal_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5
    }

    idx = None
    for word, value in ordinal_words.items():
        if re.search(rf"\b{word}\b", lower):
            idx = value
            break

    if idx is None:
        numeric = re.search(r"\b(\d+)\b", lower)
        if numeric:
            idx = int(numeric.group(1))

    if idx is not None and 1 <= idx <= len(remembered):
        target_name = remembered[idx - 1]
        for project in projects:
            if str(project.get("name", "")).lower() == str(target_name).lower():
                return project

    # Name-based follow-up in remembered list.
    for name in remembered:
        if name.lower() in lower:
            for project in projects:
                if str(project.get("name", "")).lower() == name.lower():
                    return project
    return None


def classify_requested_field(message: str) -> str | None:
    lower = message.lower()
    for field, aliases in PROJECT_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                return field
    return None


def score_project_for_field(project: dict[str, Any], field: str, query_tokens: set[str]) -> float:
    text = " ".join([
        str(project.get("name", "")),
        str(project.get("description", "")),
        str(project.get("problem", "")),
        str(project.get("role", "")),
        " ".join(project.get("category", [])) if isinstance(project.get("category"), list) else str(project.get("category", "")),
        " ".join(project.get("features", [])) if isinstance(project.get("features"), list) else str(project.get("features", "")),
        " ".join(project.get("stack", [])) if isinstance(project.get("stack"), list) else str(project.get("stack", ""))
    ]).lower()
    aliases = PROJECT_FIELD_ALIASES.get(field, [])
    alias_hits = sum(1 for alias in aliases if alias in text)
    token_hits = sum(1 for token in query_tokens if token in text)
    return float(alias_hits * 2 + token_hits)


def projects_response(message: str, projects: list[dict[str, Any]]) -> str | None:
    lower = message.lower()
    project_terms = {"project", "projects", "work", "built", "made", "created", "aniverse", "pegasus", "parkinson"}
    has_project_term = contains_any(message, project_terms)
    project = find_project_in_message(message, projects, allow_fuzzy=has_project_term)
    if not has_project_term and project is None:
        return None

    if not projects:
        return "I don't have that information yet."

    is_detail_ask = (
        "elaborate" in lower
        or "detail" in lower
        or "tell me about" in lower
        or "more about" in lower
        or "explain" in lower
        or "tech" in lower
        or "stack" in lower
        or "feature" in lower
        or project is not None
    )

    # Follow-up detail ask by remembered index/name.
    if is_detail_ask:
        followup = resolve_followup_project_reference(message, projects)
        if followup is not None:
            return format_project_detail(followup)

    if project is not None:
        return format_project_detail(project)

    # Detail ask: "elaborate on X" or "tell me about X"
    for project in projects:
        name = str(project.get("name", "")).lower()
        if not name:
            continue
        if is_detail_ask and name in lower:
            return format_project_detail(project)

    field = classify_requested_field(message)
    query_tokens = set(tokenize(message))
    if field or "related" in lower or "specific" in lower or "domain" in lower or "field" in lower:
        ranked = []
        inferred_field = field or "web"
        for project in projects:
            score = score_project_for_field(project, inferred_field, query_tokens)
            if score > 0:
                ranked.append((score, project))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        picks = [entry for _, entry in ranked[:5]]
        if not picks:
            return "I don't have that information yet."
        SESSION_STATE["last_project_list"] = [str(project.get("name", "")) for project in picks if project.get("name")]
        project_lines = [format_project_line(project, index) for index, project in enumerate(picks, start=1)]
        return format_sectioned(
            f"Projects Related To {inferred_field.title()}",
            [
                ("Matches", project_lines),
                ("Next Step", "Ask for a project name if you want the stack, features, and role."),
            ],
        )

    SESSION_STATE["last_project_list"] = [str(project.get("name", "")) for project in projects if project.get("name")]
    project_lines = [format_project_line(project, index) for index, project in enumerate(projects, start=1)]
    return format_sectioned(
        "Aditya's Projects",
        [
            ("Overview", project_lines),
            ("Next Step", "Ask for AniVerse, Pegasus, or Parkinson Disease Assessment Portal for details."),
        ],
    )


def skills_response(message: str) -> str | None:
    lower = message.lower()
    skills_text = safe_read_text(DATA_DIR / "skills.md")
    if not skills_text:
        return None

    sections = markdown_sections(skills_text)
    skill_to_section: dict[str, tuple[str, str]] = {}
    for section, lines in sections.items():
        for line in lines:
            item = clean_bullet(line)
            if not item or ":" in item:
                continue
            skill_to_section[item.lower()] = (item, section)

    explicit_skill = None
    for skill in sorted(skill_to_section, key=len, reverse=True):
        if re.search(r"\W", skill):
            matched = skill in lower
        else:
            matched = bool(re.search(rf"\b{re.escape(skill)}\b", lower))
        if matched:
            explicit_skill = skill
            break

    if not contains_any(message, SKILL_QUERY_TERMS) and explicit_skill is None:
        return None

    if explicit_skill:
        original, section = skill_to_section[explicit_skill]
        return format_sectioned(
            "Skill Match",
            [
                ("Result", f"Yes. {original} is listed under {section} in Aditya's portfolio."),
            ],
        )

    category_aliases = {
        "Programming Languages": ["programming", "language", "languages", "code", "coding"],
        "Machine Learning and AI": ["machine learning", "ml", "ai", "llm", "nlp", "rag", "model"],
        "Web Development": ["web", "frontend", "front end", "backend", "full stack", "react"],
        "Data, Database, and Analysis": ["database", "data", "db", "sql", "postgres", "mongodb", "analysis"],
        "Tools and Technologies": ["tool", "tools", "technology", "technologies", "docker", "git", "deployment"]
    }

    selected = [
        section for section, aliases in category_aliases.items()
        if section in sections and any(alias in lower for alias in aliases)
    ]
    if not selected:
        selected = [
            "Programming Languages",
            "Web Development",
            "Machine Learning and AI",
            "Data, Database, and Analysis",
            "Tools and Technologies"
        ]

    formatted_sections: list[tuple[str, str | list[str]]] = []
    for section in selected:
        entries = [clean_bullet(line) for line in sections.get(section, [])]
        entries = [entry for entry in entries if entry and ":" not in entry][:10]
        if entries:
            formatted_sections.append((section, entries))
    return format_sectioned("Aditya's Skills", formatted_sections) if formatted_sections else "I don't have that information yet."


def contact_response(message: str) -> str | None:
    if not contains_any(message, CONTACT_QUERY_TERMS):
        return None

    contact_text = safe_read_text(DATA_DIR / "contact.md")
    if not contact_text:
        return "I don't have that information yet."

    items = []
    for line in contact_text.splitlines():
        cleaned = clean_bullet(line)
        if ":" in cleaned and not cleaned.startswith("#"):
            items.append(cleaned)

    lower = message.lower()
    if "github" in lower:
        items = [item for item in items if item.lower().startswith("github")]
    elif "linkedin" in lower:
        items = [item for item in items if item.lower().startswith("linkedin")]
    elif "email" in lower or "mail" in lower or "gmail" in lower:
        items = [item for item in items if "email" in item.lower()]
    elif "phone" in lower or "whatsapp" in lower:
        items = [item for item in items if "phone" in item.lower() or "whatsapp" in item.lower()]

    if not items:
        return "I don't have that information yet."
    return format_sectioned(
        "Contact",
        [
            ("Portfolio Links", items),
        ],
    )


def portfolio_response(message: str) -> str | None:
    lower = message.lower()
    if not contains_any(message, PORTFOLIO_QUERY_TERMS):
        return None

    if "aditya ai" in lower or "assistant" in lower or "chatbot" in lower:
        return format_sectioned(
            "Aditya AI",
            [
                ("Purpose", "Aditya AI is the portfolio assistant. It is meant to answer only from Aditya's portfolio knowledge: about, skills, projects, contact details, and the portfolio UI."),
                ("Action Commands", [
                    "open projects",
                    "open resume",
                    "open skills",
                    "play snake",
                    "play minesweeper",
                ]),
            ],
        )

    if "terminal" in lower or "command" in lower:
        return format_sectioned(
            "Portfolio Terminal",
            [
                ("Supported Commands", [
                    "help",
                    "open about",
                    "open projects",
                    "open games",
                    "open skills",
                    "open socials",
                    "open store",
                    "open recyclebin",
                    "open chrome",
                    "open terminal",
                    "open notepad",
                    "open adityaai",
                    "theme dark / theme light",
                    "time / date",
                    "clear",
                    "shutdown",
                ]),
            ],
        )

    if "game" in lower or "snake" in lower or "minesweeper" in lower:
        return format_sectioned(
            "Games",
            [
                ("Available Games", ["Snake", "Minesweeper"]),
                ("How They Open", "Both run inside app-style windows through the Games Launcher."),
            ],
        )

    if "store" in lower or "recycle" in lower:
        return format_sectioned(
            "Store And Recycle Bin",
            [
                ("Store", "The portfolio has a Microsoft Store-style app for installing project shortcuts."),
                ("Recycle Bin", "Installed project apps can be deleted into the Recycle Bin, restored, or permanently emptied."),
            ],
        )

    return format_sectioned(
        "Portfolio OS",
        [
            ("Overview", "Aditya's portfolio is a Windows-style desktop simulation in a web page."),
            ("Main Features", [
                "Desktop icons and File Explorer windows",
                "Taskbar and Start menu",
                "Chrome-like browser",
                "Terminal and Notepad",
                "Project showcase",
                "Games Launcher with Snake and Minesweeper",
                "Store and Recycle Bin behavior",
                "Wallpaper personalization",
                "Calendar flyout",
                "Dark mode",
                "Aditya AI",
            ]),
        ],
    )


class RagStore:
    def __init__(self):
        self.ready = False
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict[str, str]] = []
        self.doc_hash = ""

    def _hash_docs(self) -> str:
        h = hashlib.sha256()
        paths = list(DATA_DIR.glob("**/*")) + list((ROOT / "About Me").glob("**/*.pdf"))
        for path in sorted(paths):
            if not path.is_file():
                continue
            h.update(str(path.relative_to(ROOT)).encode("utf-8"))
            h.update(path.read_bytes())
        return h.hexdigest()

    def _chunk_text(self, text: str, source: str, size: int = 560, overlap: int = 120) -> list[dict[str, str]]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        out: list[dict[str, str]] = []
        step = max(1, size - overlap)
        for start in range(0, len(cleaned), step):
            segment = cleaned[start:start + size]
            if len(segment) < 60 and start != 0:
                continue
            out.append({"text": segment, "source": source})
        return out

    def _json_to_text(self, value: Any, prefix: str = "") -> list[str]:
        lines: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}{key}." if prefix else f"{key}."
                lines.extend(self._json_to_text(child, next_prefix))
        elif isinstance(value, list):
            for index, child in enumerate(value, start=1):
                next_prefix = f"{prefix}{index}."
                lines.extend(self._json_to_text(child, next_prefix))
        else:
            head = prefix[:-1] if prefix.endswith(".") else prefix
            lines.append(f"{head}: {value}")
        return lines

    def _read_pdf(self, path: Path) -> str:
        if not PdfReader:
            return ""
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""

    def _read_data_docs(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []

        for path in sorted(DATA_DIR.glob("**/*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".md", ".txt"}:
                docs.extend(self._chunk_text(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))))
            elif path.suffix.lower() == ".json":
                raw = path.read_text(encoding="utf-8")
                try:
                    parsed = json.loads(raw)
                    as_text = "\n".join(self._json_to_text(parsed))
                    docs.extend(self._chunk_text(as_text, str(path.relative_to(ROOT))))
                except Exception:
                    docs.extend(self._chunk_text(raw, str(path.relative_to(ROOT))))
            elif path.suffix.lower() == ".pdf":
                text = self._read_pdf(path)
                if text:
                    docs.extend(self._chunk_text(text, str(path.relative_to(ROOT))))

        # Also include resume-like files outside data for convenience.
        for path in sorted((ROOT / "About Me").glob("**/*.pdf")):
            if not path.is_file():
                continue
            text = self._read_pdf(path)
            if text:
                docs.extend(self._chunk_text(text, str(path.relative_to(ROOT))))

        return docs

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not client:
            raise RuntimeError("OPENAI_API_KEY missing")
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def ensure(self) -> None:
        if self.ready:
            return

        current_hash = self._hash_docs()
        if INDEX_FILE.exists():
            saved = pickle.loads(INDEX_FILE.read_bytes())
            if saved.get("doc_hash") == current_hash:
                self.doc_hash = current_hash
                self.chunks = saved["chunks"]
                vectors = saved["vectors"]
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                self.index = index
                self.ready = True
                return

        self.chunks = self._read_data_docs()
        if not self.chunks:
            self.index = None
            self.ready = True
            return

        if not client:
            self.index = None
            self.ready = True
            return

        try:
            vectors = self._embed_texts([chunk["text"] for chunk in self.chunks])
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self.index = index
            self.doc_hash = current_hash
            self.ready = True

            INDEX_FILE.write_bytes(pickle.dumps({
                "doc_hash": current_hash,
                "chunks": self.chunks,
                "vectors": vectors
            }))
        except Exception:
            # If embeddings fail (quota/network/model issues), keep lexical retrieval available.
            self.index = None
            self.doc_hash = current_hash
            self.ready = True

    def query(self, text: str, k: int) -> list[dict[str, Any]]:
        self.ensure()
        if not self.chunks:
            return []

        semantic: list[dict[str, Any]] = []
        if self.index is not None and client is not None:
            try:
                q = self._embed_texts([text])
                scores, idx = self.index.search(q, min(max(k * 2, 8), len(self.chunks)))
                for score, i in zip(scores[0], idx[0]):
                    if i < 0:
                        continue
                    if float(score) < MIN_SIMILARITY:
                        continue
                    adjusted = float(score) * source_bias(self.chunks[i]["source"])
                    semantic.append({
                        "score": adjusted,
                        "text": self.chunks[i]["text"],
                        "source": self.chunks[i]["source"]
                    })
            except Exception:
                semantic = []

        lexical = lexical_query(self.chunks, text, max(k * 2, 8))

        merged: list[dict[str, Any]] = []
        seen = set()
        for item in semantic + lexical:
            key = (item["source"], item["text"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return merged[:k]


store = RagStore()
app = FastAPI(title="Aditya AI Backend", version="1.0.0")


def parse_allowed_origins() -> list[str]:
    defaults = [
        FRONTEND_ORIGIN,
        "null",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]

    dynamic = [origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()]
    merged: list[str] = []
    for origin in defaults + dynamic:
        if origin and origin not in merged:
            merged.append(origin)
    return merged


ALLOWED_ORIGINS = parse_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def detect_action(message: str) -> dict[str, str] | None:
    lower = message.lower().strip()
    has_verb = any(verb in lower for verb in ACTION_VERBS)
    if not has_verb:
        return None

    alias_to_target = {alias: target for target, aliases in ACTION_ALIASES.items() for alias in aliases}

    for alias, target in alias_to_target.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return {
                "action": "OPEN_WINDOW",
                "target": target,
                "message": f"Opening {target}..."
            }

    # Fuzzy single-token fallback for small typos like "minesweper".
    tokens = re.findall(r"[a-z0-9+]+", lower)
    alias_tokens = {alias: alias for alias in alias_to_target if " " not in alias}
    for token in tokens:
        match = get_close_matches(token, list(alias_tokens.keys()), n=1, cutoff=0.82)
        if match:
            target = alias_to_target[match[0]]
            return {
                "action": "OPEN_WINDOW",
                "target": target,
                "message": f"Opening {target}..."
            }
    return None


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9+]+", text.lower()) if word not in STOP_WORDS and len(word) > 1]


def source_bias(source: str) -> float:
    lower = source.lower()
    if lower.endswith(".md") or lower.endswith(".json") or lower.endswith(".txt"):
        return 1.15
    if lower.endswith(".pdf"):
        return 0.75
    return 1.0


def lexical_query(chunks: list[dict[str, str]], query: str, k: int) -> list[dict[str, Any]]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scored: list[tuple[float, dict[str, str]]] = []
    q_set = set(q_tokens)
    lower_query = query.lower()
    for chunk in chunks:
        chunk_text = chunk["text"].lower()
        c_tokens = set(tokenize(chunk_text))
        if not c_tokens:
            continue
        overlap = len(q_set & c_tokens)
        if overlap == 0:
            continue
        phrase_hits = sum(1 for token in q_set if token in chunk_text)
        source_bonus = 0.12 if any(token in chunk["source"].lower() for token in q_set) else 0.0
        exact_bonus = 0.25 if lower_query and lower_query in chunk_text else 0.0
        score = ((overlap / max(1, len(q_set))) + phrase_hits * 0.03 + source_bonus + exact_bonus) * source_bias(chunk["source"])
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"score": float(score), "text": item["text"], "source": item["source"]}
        for score, item in scored[:k]
    ]


def local_grounded_answer(user_message: str, retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "I don't have that information yet."

    q_tokens = set(tokenize(user_message))
    ranked_sentences: list[tuple[float, str]] = []
    for item in retrieved:
        sentences = re.split(r"(?<=[.!?])\s+", item["text"])
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 20:
                continue
            s_tokens = set(tokenize(cleaned))
            overlap = len(q_tokens & s_tokens)
            ranked_sentences.append((float(overlap), cleaned))

    ranked_sentences.sort(key=lambda pair: pair[0], reverse=True)
    picks = [text for score, text in ranked_sentences if score > 0][:2]
    if not picks:
        picks = [retrieved[0]["text"][:260].strip()]

    answer = " ".join(picks)
    return pretty_text(answer) if answer else "I don't have that information yet."


def grounded_answer(user_message: str, retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "I don't have that information yet."

    if not client:
        return local_grounded_answer(user_message, retrieved)

    context_blocks = []
    for idx, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Chunk {idx} | source: {item['source']} | score: {item['score']:.3f}]\n{item['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = (
        "Use only the context below. If answer is not explicitly present, reply exactly: "
        "I don't have that information yet.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User query: {user_message}"
    )

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": APP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return pretty_text(completion.choices[0].message.content.strip())
    except Exception:
        # Fall back to local extraction-based response when chat completion fails.
        return local_grounded_answer(user_message, retrieved)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    message = req.message.strip()
    if not message:
        return {
            "message": "Say something and I will handle it.",
            "action": "NONE",
            "target": None,
            "sources": []
        }

    action = detect_action(message)
    if action:
        return {
            "message": action["message"],
            "action": action["action"],
            "target": action["target"],
            "sources": []
        }

    about = about_response(message)
    if about is not None:
        return {
            "message": about,
            "action": "NONE",
            "target": None,
            "sources": ["data/resume.md", "data/contact.md"]
        }

    projects = safe_read_projects()
    project_answer = projects_response(message, projects)
    if project_answer is not None:
        return {
            "message": project_answer,
            "action": "NONE",
            "target": None,
            "sources": ["data/projects.json"]
        }

    skills = skills_response(message)
    if skills is not None:
        return {
            "message": skills,
            "action": "NONE",
            "target": None,
            "sources": ["data/skills.md"]
        }

    contact = contact_response(message)
    if contact is not None:
        return {
            "message": contact,
            "action": "NONE",
            "target": None,
            "sources": ["data/contact.md"]
        }

    portfolio = portfolio_response(message)
    if portfolio is not None:
        return {
            "message": portfolio,
            "action": "NONE",
            "target": None,
            "sources": ["data/portfolio.md"]
        }

    if not in_portfolio_scope(message):
        return {
            "message": "I don't have that information yet.",
            "action": "NONE",
            "target": None,
            "sources": []
        }

    key = message.lower()
    cached = cache.get(key)
    if cached:
        return cached

    try:
        retrieved = store.query(message, TOP_K)
    except Exception:
        # Keep the endpoint alive even if semantic retrieval initialization fails.
        retrieved = lexical_query(store.chunks, message, TOP_K) if store.chunks else []
    answer = grounded_answer(message, retrieved)

    payload = {
        "message": pretty_text(answer),
        "action": "NONE",
        "target": None,
        "sources": [item["source"] for item in retrieved]
    }
    cache.put(key, payload)
    return payload
