### backend-rag

A small RAG (Retrieval-Augmented Generation) microservice used by the project to answer user queries with contextual product data.

- What it does: loads a local Chroma vector DB (stored in `chroma_db/`), performs similarity search, then calls the local LLM endpoint (via `main.py`) to compose user-facing answers that include retrieved product sources.
- Key files:
  - `main.py` — service entrypoint / API handler
  - `requirements.txt` — Python dependencies
  - `chroma_db/` — Chroma SQLite DB and stored vectors
  - `.env.example` — environment variables to configure the service
- Quick usage:
  - Copy `.env.example` → `.env` and set variables.
  - Install deps: `python -m pip install -r requirements.txt`
  - Run: `python main.py`
- Notes: The service persists vector data under `chroma_db/` and should not be edited while the service is running. Do not change `main.py` unless you need to adjust prompt/context handling or LLM endpoint configuration.