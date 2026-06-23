# Repository Guidelines

## Project Structure & Module Organization

Knowledge Base Assistant is a general-purpose RAG and direct-chat application
built with FastAPI, Qdrant, SentenceTransformers, PostgreSQL, and a
configurable LLM provider.

- `app/`: application code. `main.py` exposes the FastAPI app and routes,
  `static/` contains the browser UI, `intent_router.py` chooses RAG vs direct
  chat through keyword/state, Jina-embedding, and LLM-classifier passes, `rag.py`
  coordinates BM25/vector recall, RRF fusion, reranking, prompts, generation,
  and context-budget-aware history compaction, `reranker.py` loads and runs the
  Jina cross-encoder, `vector_store.py` manages Markdown chunking, Jina
  embeddings v3 task routing, BM25 indexing, Qdrant vector search, and RRF
  fusion, `session_store.py` manages PostgreSQL-backed users, auth sessions,
  chat sessions, messages, retrieved references, route metadata, runtime LLM
  settings, and summaries, `llm_client.py` calls cloud or local LLM APIs, and
  `config.py` reads environment settings.
- `scripts/`: operational, smoke-test, and evaluation scripts for local
  services, ingestion, retrieval, routing, prompt budgeting, settings, session
  helpers, and offline intent-router A/B checks.
- `data/docs/`: replaceable Markdown source documents ingested into Qdrant.
  The committed corpus is a synthetic Chinese restaurant/business case,
  including the 巨大历史机遇/巨大历史鲫鱼 meme topic.
- `data/eval/`: labeled intent-routing evaluation cases used by
  `scripts/intent_router_ab.py`, including bilingual and cross-lingual slices
  for comparing Jina against other encoders.
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
- `python scripts/test_retrieve.py`: smoke-test retrieval against a running
  Qdrant collection.
- `python scripts/test_chunking.py`: validate Markdown-aware chunk boundaries.
- `python scripts/test_vector_store.py`: validate lazy loading, recursive
  ingest, payload indexes, score thresholds, BM25 keyword recall, RRF fusion,
  and incremental document replacement.
- `python scripts/test_session_store.py`: validate login/session helper
  behavior without a live PostgreSQL service.
- `python scripts/test_intent_router.py`: smoke-test routing for RAG vs
  direct-chat questions.
- `python scripts/intent_router_ab.py --fake-embedder`: smoke-test the offline
  intent-router A/B harness without downloading encoder weights.
- `python scripts/intent_router_ab.py --model-variant old_bge=BAAI/bge-small-en-v1.5,,0`:
  compare current Jina intent embeddings against the old BGE small encoder on
  the labeled routing set; this may download model weights.
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
retrieval or deployment changes, also validate with
`python scripts/test_retrieve.py`, `curl http://localhost:8080/health`, login
through `/auth/login`, and a token-authenticated `POST /rag` request. Inspect
the returned or stored `contexts` for `retrieval_source`, `vector_score`,
`bm25_score`, `rrf_score`, and `rerank_score` when checking hybrid retrieval.
Add future automated tests under `tests/` using `test_*.py` naming.

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
the existing password. The startup-created administrator is the only
superuser. Superuser-only Admin UI controls can update the global LLM provider
format, API base URL, model name, context-window size, and API key at runtime;
those values are stored in PostgreSQL `app_settings` and override the `.env`
LLM defaults. API keys are write-only from the browser's perspective.
Admin-created and CSV-imported users default to `must_change_password`, so they
must update their password before using app features.

For restricted networks, configure runtime proxy or mirror values in `.env`.
Docker daemon proxy settings only affect image pulls, not running containers.

RAG retrieval uses Jina embeddings v5 text small by default:
`EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small`,
`EMBEDDING_TRUST_REMOTE_CODE=1`, `CHUNK_SIZE=2000`,
`CHUNK_OVERLAP=300`, query embeddings use `retrieval` with prompt `query`,
document chunks use `retrieval` with prompt `document`, and intent-routing
embeddings use `classification`.
Hybrid recall runs before reranking: BM25 recalls `BM25_TOP_K` keyword
candidates from Markdown chunks, Qdrant recalls `RECALL_TOP_K`
cosine-similarity candidates, reciprocal rank fusion keeps `RRF_TOP_K`
candidates, `jinaai/jina-reranker-v3` reranks them through its native
`AutoModel.rerank()` interface, and only `RETRIEVE_TOP_K` chunks enter the
prompt. The API preloads and warms the reranker during startup by default with
`RERANKER_PRELOAD=1`; warmup failure is logged as
`retrieval_degraded=True`/`reranker_degraded=True` but does not prevent startup.
Keep Hugging Face cache, mirror, or proxy settings ready before rebuilding
containers. The browser exposes these limits as `BM25 K`, `Cosine K`, `RRF K`,
and `Final K`, defaulting to `100`, `100`, `100`, and `5`.

Retrieval degradation must be explicit. If Qdrant/vector recall fails, RAG falls
back to BM25-only recall from local Markdown. If reranking fails, the pipeline
uses the unre-ranked RRF/BM25 coarse results capped by `Final K`. Responses and
stored assistant messages include `retrieval_degraded`, `qdrant_degraded`,
`reranker_degraded`, and `degradation_reason`; the frontend shows the same
warning, and server logs write the degradation booleans.

Conversation memory defaults to API-scale context windows:
`LLM_CONTEXT_MAX_TOKENS=256000`, safety margin `8192`, prompt overhead `2048`,
`CONVERSATION_SUMMARY_MAX_CHARS=256000`, `SUMMARY_HISTORY_MAX_CHARS=200000`,
and `SUMMARY_MAX_TOKENS=4096`. History is not count-truncated before
compaction (`HISTORY_MAX_MESSAGES=0`); compaction runs only when estimated
summary + uncompressed history would exceed the active context window after
reserving output, question, safety, prompt overhead, and expected retrieved
reference tokens.

The intent router is layered: keyword rules short-circuit explicit/domain
cases, general technical/database questions outside the local corpus, and
short, strong referential follow-ups after the previous assistant answer
actually used retrieved contexts; the second layer compares the current
question plus bounded recent conversation and compact memory against
single-intent RAG/direct anchor queries with Jina classification embeddings;
ambiguous cases fall through to an LLM classifier that receives structured
previous-route state. The intent embedding budgets are character-based
safeguards, not a tokenizer hard limit; if the encoder raises, the second layer
returns no decision and the router continues to the LLM classifier.
The LLM classifier is prompted to return
`<think>THINK_AND_JUDGEMENT</think><answer>JSON_ANS</answer>`, and only the
JSON inside `<answer>` is parsed for routing. Use
`scripts/intent_router_ab.py` to compare keyword-only routing, current Jina
embedding routing, threshold variants, and alternative encoder models before
changing thresholds or encoders. If Docker logs do not show INFO-level
retrieval messages, use the response or PostgreSQL-stored `contexts` scores to
confirm which retrieval stages ran.

All users can enable the browser `RAG-only` switch. That sends `rag_only=true`
to `/rag`, bypasses the intent router for that request, and stores route
metadata as `rag_only`.
