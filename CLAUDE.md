# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge Base Assistant is a RAG-and-direct-chat web application. It indexes a
local Markdown corpus into Qdrant for vector recall and into an in-process BM25
index for keyword recall, routes each user question to either RAG or direct chat
via `IntentRouter`, fuses and reranks relevant chunks, then generates answers
with a configurable LLM. The browser UI, auth layer, and session persistence
are all served by the same FastAPI process.

## Development Commands

### Environment Setup

```bash
# Preferred: uv-based venv (API dependencies only)
env UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.api.txt

# Alternative: conda with full dependencies (includes vLLM extras)
conda activate rag_llm && pip install -r requirements.txt
```

### Docker Compose (primary workflow)

```bash
cp .env.example .env          # then set LLM_API_KEY and LLM_MODEL
docker compose up --build     # start PostgreSQL + Qdrant + API
APP_PORT=9000 docker compose up --build   # custom host port
scripts/compose_up.sh up --build          # GPU-with-CPU-fallback wrapper
CUDA=FALSE docker compose -f compose.yaml -f compose.cpu.yaml up --build  # force CPU
docker compose up -d --build --force-recreate  # after .env changes
docker compose down -v        # reset all persisted volumes
```

### Local Development (without Docker)

```bash
bash scripts/start_qdrant.sh       # start Qdrant
uvicorn app.main:app --host 0.0.0.0 --port 8080  # run FastAPI
```

### Corpus Ingestion

```bash
python scripts/ingest_docs.py            # incremental (replaces existing by source)
python scripts/ingest_docs.py --recreate # full rebuild (required when changing embedding model)
```

### Smoke Tests (no pytest; run these before pushing)

```bash
python -m compileall app scripts          # syntax check
python scripts/test_chunking.py           # Markdown-aware chunk boundaries
python scripts/test_vector_store.py       # ingest, payload indexes, score thresholds
python scripts/test_intent_router.py      # RAG vs direct-chat routing
python scripts/test_reranker.py           # cross-encoder ordering
python scripts/test_prompt_budget.py      # prompt trimming and history compaction
python scripts/test_settings.py           # config wiring, CSV parsing, LLM helpers
python scripts/test_session_store.py      # login/session helpers (no live DB required)
python scripts/test_retrieve.py           # end-to-end retrieval (requires running Qdrant)
docker compose config                     # validate compose files
```

### Pre-Push Checklist

```bash
git status --short --ignored
python -m compileall app scripts
python scripts/test_chunking.py
python scripts/test_vector_store.py
python scripts/test_intent_router.py
python scripts/test_prompt_budget.py
python scripts/test_settings.py
docker compose config
```

## Architecture

### Request Flow

```
Browser UI  →  POST /rag
               ↓
           IntentRouter (keyword → embedding → LLM fallback)
               ↓ use_rag?
         [RAG path]                    [Direct path]
     VectorStore.hybrid_search()   build_direct_messages()
       |-- BM25 recall
       |-- Qdrant vector recall
       |-- RRF fusion
     Reranker.rerank()
     build_rag_messages()
               ↓
           LLMClient.chat()
               ↓
       SessionStore.append_message()  (persist to PostgreSQL)
```

### Key Modules

- **`app/config.py`** — Frozen `Settings` dataclass; every tunable parameter is read from env vars here. Adding new config always goes through this class; never hard-code URLs, model names, or limits elsewhere.
- **`app/main.py`** — FastAPI lifespan (constructs all singletons: `VectorStore`, `LLMClient`, `SessionStore`, `Reranker`, `RAGPipeline`), route definitions, Pydantic request/response models, and auth dependency chain (`require_login_auth` → `require_password_ready_user` → `require_admin_auth`). Serves the UI shell at `/` and mounts static assets at `/static`.
- **`app/intent_router.py`** — Three-pass classifier: (1) keyword exact/regex match against `FORCE_*`, `DOMAIN_RAG_*`, and `DIRECT_TASK_PATTERNS`; (2) cosine similarity against `RAG_ANCHORS`/`DIRECT_ANCHORS` using the same BGE embedding model; (3) LLM zero-shot fallback. **When swapping the corpus, update `DOMAIN_RAG_PHRASES`, `DOMAIN_RAG_PATTERNS`, and the LLM fallback prompt string inside `_route_with_llm`.**
- **`app/rag.py`** — Orchestrates the full answer pipeline: history normalization, history compaction (rolling summary via LLM when turn count exceeds `HISTORY_COMPACT_AFTER_TURNS`), intent routing, BM25 + vector recall, RRF fusion, reranking, and prompt construction. `RAGPipeline.answer()` is the single entry point from the API layer.
- **`app/vector_store.py`** — Markdown-aware chunking (text/code/table separately, heading metadata preserved as `h1`/`h2`/`h3` payload), SentenceTransformers BGE embedding, BM25 keyword indexing over local Markdown chunks, RRF fusion, and Qdrant collection management. Incremental ingest replaces all chunks by `source` filename so no stale chunks accumulate on edits.
- **`app/reranker.py`** — Jina `jina-reranker-v3` cross-encoder, loaded via `AutoModel.rerank()` (requires `trust_remote_code=True`). Pre-warmed at startup when `RERANKER_PRELOAD=1`. Batched when recall exceeds `RERANKER_MAX_DOCUMENTS_PER_CALL`.
- **`app/llm_client.py`** — Provider-aware HTTP client supporting `openai_compatible` and `anthropic`. Retries `429`/`5xx` with exponential backoff. The `LLMClient` instance is shared across RAG, history compaction, and intent routing calls.
- **`app/session_store.py`** — Raw `psycopg2` PostgreSQL (no ORM). Manages users, PBKDF2-SHA256 passwords, SHA-256 bearer tokens, chat sessions, messages with retrieved contexts as JSON, retrieval scores (`vector_score`, `bm25_score`, `rrf_score`, `rerank_score`), route metadata, and compacted conversation summaries.
- **`app/prompt_budget.py`** — Text-trimming utilities used by both `RAGPipeline` and `IntentRouter` to enforce character budgets before building prompts.
- **`app/security.py`** — `bearer_token()` extracts the raw token string from the `Authorization` header.

### Frontend (`app/static/`)

Plain ES modules served directly by FastAPI — **no build step, no npm, no CDN** (keeps the restricted-network story intact). Icons are an inline SVG sprite in `index.html`.

- `index.html` is structure-only and references `/static/css/styles.css` + `/static/js/main.js` (`type="module"`).
- `js/store.js` holds the single mutable `state` object plus a tiny pub/sub bus (`on`/`emit`). Views never call each other's render functions directly; they `emit` a named event (`auth`, `sessions`, `messages`, `references`, `meta`, `admin`, `authenticated`, `unauthorized`, `password-required`, `open-admin`, `session-selected`, `focus-prompt`) and the owning view re-renders.
- `js/api.js` is the only place that calls `fetch`; it attaches the bearer token and turns `401`/`403` into `unauthorized`/`password-required` events.
- `js/views/{auth,sessions,chat,meta,admin}.js` each own one region: `mountX()` caches elements + wires listeners + subscribes to events. `main.js` mounts all views, wires global events, and manages the responsive drawers.
- Cross-view imports are one-directional (`chat`/`admin` import data actions from `sessions`); navigation and DB-side effects are bridged through events to avoid cycles.
- `styles.css` uses design tokens with a `prefers-color-scheme: dark` block. The three-column `.app-shell` is `height: 100dvh; overflow: hidden` so the sidebar and references panel stay pinned while only the session list / chat log / reference list scroll internally.
- Retrieval controls live in the meta/settings view. `/health/details` initializes `BM25 K`, `Cosine K`, `RRF K`, and `Final K`; `/rag` sends them as `bm25_top_k`, `recall_top_k`, `rrf_top_k`, and `top_k`.

### LLM Provider Configuration

| Provider | `LLM_PROVIDER` | `LLM_BASE_URL` |
|---|---|---|
| OpenAI-compatible (default) | `openai_compatible` | `https://api.openai.com/v1` |
| Anthropic | `anthropic` | `https://api.anthropic.com/v1` |
| Local vLLM | `openai_compatible` | `http://vllm:8000/v1` |

For Anthropic, the `LLM_ANTHROPIC_VERSION` env var pins the API version header. Health check endpoint defaults: `GET /models` for OpenAI-compatible, `POST /messages/count_tokens` for Anthropic. Override with `LLM_HEALTH_PATH`.

### GPU / CUDA Behavior

`CUDA=TRUE` (default) makes both the BGE embedding model and Jina reranker prefer CUDA. If PyTorch cannot allocate GPU (old architecture, OOM, no NVIDIA toolkit), both fall back to CPU with a log message. `compose.cpu.yaml` overrides the GPU device spec for forced-CPU deployments. The `scripts/compose_up.sh` wrapper tries GPU first and retries on CPU if Docker rejects the device allocation.

### Retrieval Observability

The `/rag` response and stored assistant `contexts` are the source of truth for
which retrieval stages ran. `retrieval_source=hybrid` means a chunk appeared in
both BM25 and vector recall; pure `bm25` or `vector` sources carry only the
matching score. `rrf_score` confirms fusion, and `rerank_score` confirms the
Jina reranker ran. Docker logs may omit the retrieval `logger.info()` lines
because the container's Python root logger can run at `WARNING`.

### Replacing the Corpus

1. Replace/edit `data/docs/**/*.md`.
2. Run `python scripts/ingest_docs.py --recreate`.
3. Update `DOMAIN_RAG_PHRASES`, `DOMAIN_RAG_PATTERNS`, and the corpus description string in `IntentRouter._route_with_llm()`.

### Coding Conventions

- Python 3.12. 4-space indentation. Type hints on public helpers.
- All configuration through `Settings` in `app/config.py`; never hard-code service URLs, model names, or limits in module bodies.
- `auth_enabled` is always `True` — the `Settings.__post_init__` enforces this; do not attempt to disable it.
- `DEBUG=0` in production — error details are suppressed from HTTP responses; full tracebacks go to server logs only.
- Commit messages are short and imperative: `Add Docker Compose deployment`, `Fix Qdrant ingest retry`.
- Future automated tests belong under `tests/test_*.py`.
