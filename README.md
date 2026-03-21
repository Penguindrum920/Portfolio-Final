---
title: Aditya AI Backend
emoji: robot
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Aditya AI Backend (Hugging Face Spaces)

This repository supports:
- Frontend deployment on Vercel
- Backend deployment on Hugging Face Spaces (Docker)

## Spaces Runtime

The Space starts FastAPI using:
- uvicorn backend.main:app --host 0.0.0.0 --port 7860

Health endpoint:
- /health

Chat endpoint:
- /chat

## Required Space Variables

Set these in Space Settings -> Variables:
- OPENAI_API_KEY
- OPENAI_CHAT_MODEL (optional, default gpt-4o-mini)
- OPENAI_EMBED_MODEL (optional, default text-embedding-3-small)
- TOP_K (optional)
- MIN_SIMILARITY (optional)
- CACHE_SIZE (optional)
- FRONTEND_ORIGIN (optional)
- ALLOWED_ORIGINS (required for production domains)

Example ALLOWED_ORIGINS:
https://adityakaushik.in,https://www.adityakaushik.in,https://your-vercel-project.vercel.app

## Frontend Endpoint

In code.html, set:
https://REPLACE_WITH_YOUR_SPACE.hf.space/chat

to your actual Space URL.
