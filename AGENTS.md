# Repository Guidelines

## Project Structure & Module Organization

Knowledge Base Assistant is a general-purpose RAG and direct-chat application
built with FastAPI, Qdrant, SentenceTransformers, PostgreSQL, and a
configurable LLM provider.

- `app/`: application code. `main.py` exposes the FastAPI app and routes,
  `static/` contains the browser UI, `intent_router.py` chooses RAG vs direct
  chat, `rag.py` coordinates recall, reranking, prompts, generation, and
  history compaction, `reranker.py` loads and runs the Jina cross-encoder,
  `vector_store.py` manages Markdown chunking and Qdrant, `session_store.py`
  manages PostgreSQL-backed users, auth sessions, chat sessions, messages, and
  summaries, `llm_client.py` calls cloud or local LLM APIs, and `config.py`
  reads environment settings.
- `scripts/`: operational and smoke-test scripts for local services,
  ingestion, retrieval, routing, prompt budgeting, settings, and session
  helpers.
- `data/docs/`: replaceable Markdown source documents ingested into Qdrant.
  The committed corpus is a synthetic Chinese restaurant/business case,
  including the 巨大历史机遇/巨大历史鲫鱼 meme topic.
- `docker/`, `Dockerfile`, `compose.yaml`: container entrypoint and Docker
  Compose deployment.
- Runtime state such as `postgres_data/`, `qdrant_storage/`, `models/`, caches,
  logs, virtual environments, and `.env` must stay uncommitted.

## Build, Test, and Development Commands

- `cp .env.example .env`: create local deployment settings.
- `docker compose up --build`: build and start PostgreSQL, Qdrant, document
  ingest, and FastAPI using the configured LLM API.
- `APP_PORT=9000 docker compose up --build`: run the frontend/API on a
  different host port.
- `docker compose up -d --build --force-recreate`: recreate containers after
  changing `.env` values such as LLM settings, admin bootstrap values, or
  retrieval configuration.
- `docker compose down`: stop the stack.
- `docker compose down -v`: stop the stack and reset persisted PostgreSQL,
  Qdrant, and Hugging Face cache volumes.
- `env UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.api.txt`:
  set up a local API development environment with uv.
- `conda activate rag_llm && pip install -r requirements.txt`: set up full
  manual local development, including optional local vLLM dependencies.
- `python scripts/ingest_docs.py --recreate`: rebuild the Qdrant collection
  from `data/docs/**/*.md`.
- `python scripts/test_retrieve.py`: smoke-test vector retrieval against a
  running Qdrant collection.
- `python scripts/test_chunking.py`: validate Markdown-aware chunk boundaries.
- `python scripts/test_vector_store.py`: validate lazy loading, recursive
  ingest, payload indexes, score thresholds, and incremental document
  replacement.
- `python scripts/test_session_store.py`: validate login/session helper
  behavior without a live PostgreSQL service.
- `python scripts/test_intent_router.py`: smoke-test routing for RAG vs
  direct-chat questions.
- `python scripts/test_reranker.py`: smoke-test cross-encoder reranker ordering
  with a fake model.
- `python scripts/test_prompt_budget.py`: validate prompt trimming, query
  budgets, and history compaction behavior.
- `python scripts/test_settings.py`: validate configuration wiring, auth gates,
  CSV parsing, LLM helpers, retries, and API request models.
- `uvicorn app.main:app --host 0.0.0.0 --port 8080`: run the API manually.
- `python -m compileall app scripts`: quick syntax check before pushing.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, type hints for public
helpers, and concise comments only where behavior is not obvious. Keep module
names lowercase with underscores. Prefer environment-driven configuration
through `Settings` in `app/config.py` instead of hard-coded service URLs,
provider names, model names, or limits.

## Testing Guidelines

There is no formal pytest suite yet. Use the script-style smoke tests above and
at minimum run `python -m compileall app scripts` for code changes. For
retrieval or deployment changes, also validate with `python scripts/test_retrieve.py`,
`curl http://localhost:8080/health`, login through `/auth/login`, and a
token-authenticated `POST /rag` request. Add future automated tests under
`tests/` using `test_*.py` naming.

## Commit & Pull Request Guidelines

Git history uses short descriptive commits. Keep commits focused and
imperative, for example `Add Docker Compose deployment` or `Fix Qdrant ingest
retry`.

PRs should include a short summary, commands run, configuration changes, and
deployment notes when relevant. Include screenshots only when the web UI
changes.

## Security & Configuration Tips

Do not commit `.env`, API keys, Hugging Face tokens, model weights, PostgreSQL
data, Qdrant storage, cache directories, logs, or local editor files.

Account login and server-side chat sessions are always enabled. PostgreSQL runs
as the `postgres` Docker Compose service and persists users, auth sessions,
chat sessions, messages, retrieved references, route metadata, and compacted
conversation summaries in the `postgres_data` volume. No host PostgreSQL
install is required.

Startup creates the default administrator when enabled:

```text
username: admin
password: 123456
```

Change `AUTH_DEFAULT_ADMIN_PASSWORD` before exposing the app beyond local
development. If the PostgreSQL volume already contains that user, startup keeps
the existing password. Admin-created and CSV-imported users default to
`must_change_password`, so they must update their password before using app
features.

For restricted networks, configure runtime proxy or mirror values in `.env`.
Docker daemon proxy settings only affect image pulls, not running containers.

RAG retrieval is two-stage when reranking is enabled: Qdrant recalls
`RECALL_TOP_K` candidates, `jinaai/jina-reranker-v3` reranks them through its
native `AutoModel.rerank()` interface, and only `RETRIEVE_TOP_K` chunks enter
the prompt. The API preloads and warms the reranker during startup by default
with `RERANKER_PRELOAD=1`; keep Hugging Face cache, mirror, or proxy settings
ready before rebuilding containers.
