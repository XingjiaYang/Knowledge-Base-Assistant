# Knowledge Base Assistant

中文文档见 [readme.zh.md](readme.zh.md).

Knowledge Base Assistant is a general-purpose chat application for exploring a
replaceable local Markdown knowledge base. It uses intent routing to choose
between retrieval-augmented answers grounded in the indexed corpus and direct
chat for requests that do not need retrieval.

Qdrant provides vector search, PostgreSQL stores required user accounts and
account-scoped chat sessions, SentenceTransformers generates embeddings, and
FastAPI serves both the API and browser UI. Generation is
provider-configurable: the default setup targets an OpenAI-compatible cloud API,
Anthropic is supported, and a local vLLM service remains available as an
optional Docker Compose profile.

The repository currently ships with a synthetic restaurant/business case corpus
about 家是本, 朱剑秋, the Yongge livestream incident, menu pricing, customer
reviews, social-media reactions, and financial simulation. The corpus is
intentionally niche so retrieval grounding matters more than pretrained model
memory. The intent router keeps its embedding layer domain-general, while the
keyword layer and LLM fallback prompt include lightweight corpus-specific
boundaries that should be updated when `data/docs/` is replaced.

## Highlights

- Docker Compose deployment for PostgreSQL, Qdrant, document ingestion, and FastAPI.
- Configurable cloud LLM providers with an optional local vLLM profile.
- Replaceable Markdown corpus with isolated keyword hints for the bundled
  domain.
- Chat-style web UI at `/` with adjustable recall and reranked context counts.
- Required account login with PostgreSQL-backed chat sessions per user and an
  admin UI for user/data management.
- Admin CSV import for batch user creation with strict `email,passwd` format
  validation.
- Intent routing avoids vector search for clear direct-chat questions.
- Source references are shown when retrieval is used.
- Conversation history is stored server-side and older turns are compacted into
  a reusable summary.
- Runtime configuration is environment-driven through `.env`.
- Supports Hugging Face cache volumes, local model directories, mirrors, and
  host-side HTTP proxy settings for restricted networks.

## Architecture

```text
Browser UI
   |
FastAPI /auth, /sessions, /rag
   |-- PostgreSQL users, login tokens, chat sessions, messages, summaries
   |
IntentRouter
   |-- keyword rules
   |-- embedding similarity
   |-- configured LLM zero-shot fallback for ambiguous cases
   |
RAGPipeline or Direct Chat
   |-- optional SentenceTransformers -> Qdrant vector recall
   |-- optional Jina cross-encoder reranking
   |-- OpenAI-compatible or Anthropic LLM API
   |
Answer + retrieved references + compacted conversation memory
```

Main modules:

- `app/main.py`: FastAPI routes, health check, and static UI serving.
- `app/static/`: browser chat UI — `index.html` shell, `css/styles.css`, and
  ES modules under `js/` (`store`, `api`, `dom`, `markdown`, and `views/`).
  Served directly by FastAPI with no build step.
- `app/session_store.py`: PostgreSQL-backed users, login tokens, chat sessions,
  messages, and compacted conversation summaries.
- `app/intent_router.py`: keyword, embedding, and LLM fallback routing.
- `app/rag.py`: recall, reranking, prompt construction, and history compaction.
- `app/reranker.py`: startup-preloaded Jina cross-encoder reranking for
  recalled chunks.
- `app/vector_store.py`: Markdown chunking, embeddings, Qdrant collection
  management, and search.
- `app/llm_client.py`: provider-aware client for cloud APIs or local vLLM.
- `scripts/`: manual service, ingest, and retrieval smoke-test commands.
- `data/docs/`: replaceable Markdown documents ingested into Qdrant.

## Repository Contents

This repository is intended to be pushed without local runtime state. The
committed project should include:

- Source code under `app/`, `scripts/`, and `docker/`.
- Deployment files: `Dockerfile`, `compose.yaml`, `compose.cpu.yaml`,
  `.env.example`, and dependency files.
- Markdown corpus files under `data/docs/`.
- Contributor/project docs such as `readme.md` and `AGENTS.md`.

The repository should not include `.env`, model weights, Hugging Face caches,
PostgreSQL data, Qdrant storage, logs, virtual environments, or local editor
files. These are covered by `.gitignore`.

## Quick Start With Docker Compose

Prerequisites:

- Docker with Compose v2.
- A cloud LLM API key for the default cloud-backed setup.
- Optional: a compatible NVIDIA GPU, current NVIDIA driver, and NVIDIA
  Container Toolkit if you want Docker containers to use CUDA. Very old GPUs
  may not support the PyTorch/Transformers CUDA build used by the embedding
  model, reranker, or local vLLM profile.

Create local settings:

```bash
cp .env.example .env
```

Set `LLM_API_KEY` and, if needed, `LLM_MODEL` in `.env`. Account login is
always required and chat sessions are persisted per user in PostgreSQL. The
default administrator is created on startup:

```text
username: admin
password: 123456
```

Change `AUTH_DEFAULT_ADMIN_PASSWORD` before exposing the deployment beyond
local development. If the PostgreSQL volume already contains an `admin` user,
startup keeps that user's existing password. PostgreSQL runs inside Docker
Compose; you do not need to install PostgreSQL on the host.

Start PostgreSQL, Qdrant, and the API:

```bash
docker compose up --build
```

Open the app:

```text
http://localhost:8080
```

Use a different host port:

```bash
APP_PORT=9000 docker compose up --build
```

Then open `http://localhost:9000`.

After changing `.env` values, recreate the containers so the API sees the new
environment:

```bash
docker compose up -d --build --force-recreate
```

Stop services:

```bash
docker compose down
```

Reset persisted PostgreSQL, Qdrant, and Hugging Face cache volumes:

```bash
docker compose down -v
```

## Startup Behavior

`compose.yaml` starts three services by default:

- `postgres`: runs PostgreSQL inside Docker and stores users, login tokens, and
  chat sessions in the `postgres_data` Docker volume.
- `qdrant`: stores vectors in the `qdrant_storage` Docker volume.
- `api`: waits for Qdrant and PostgreSQL, optionally ingests Markdown under
  `data/docs/`, and starts FastAPI on container port `8080`.

The `vllm` service is optional and only starts under the `local-llm` profile:

```bash
LLM_BASE_URL=http://vllm:8000/v1 \
LLM_API_KEY=token \
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
VLLM_SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
WAIT_FOR_LLM=1 \
LLM_HEALTH_CHECK_ENABLED=1 \
docker compose --profile local-llm up --build
```

Important startup flags:

```bash
INGEST_ON_STARTUP=1      # ingest docs before API starts
RECREATE_COLLECTION=0    # set to 1 to rebuild the collection during startup
WAIT_FOR_LLM=0           # set to 1 when a local LLM service must be ready first
APP_PORT=8080            # host port mapped to FastAPI/UI
QDRANT_IMAGE=qdrant/qdrant:v1.18.1
POSTGRES_IMAGE=postgres:17-alpine
```

For fast restarts after the image has already been built:

```bash
docker compose up -d
```

## Configuration

Common settings from `.env.example`:

```bash
DEBUG=0
CUDA=TRUE
QDRANT_IMAGE=qdrant/qdrant:v1.18.1

LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.9
LLM_MAX_TOKENS=4096
LLM_TIMEOUT_SECONDS=300
LLM_RETRY_ATTEMPTS=3
LLM_RETRY_BACKOFF_SECONDS=1
LLM_RETRY_BACKOFF_MAX_SECONDS=10
LLM_HEALTH_CHECK_ENABLED=0
LLM_HEALTH_PATH=

API_TOP_K_MAX=20
API_RECALL_TOP_K_MAX=1000
API_MESSAGE_MAX_CHARS=16000
API_QUESTION_MAX_CHARS=16000
API_SUMMARY_MAX_CHARS=12000
API_HISTORY_MAX_MESSAGES=120

POSTGRES_USER=kba
POSTGRES_PASSWORD=kba_password
POSTGRES_DB=kba
DATABASE_CONNECT_TIMEOUT_SECONDS=5
AUTH_DEFAULT_ADMIN_ENABLED=1
AUTH_DEFAULT_ADMIN_USERNAME=admin
AUTH_DEFAULT_ADMIN_PASSWORD=123456
AUTH_BOOTSTRAP_USERS=
AUTH_SESSION_TTL_SECONDS=604800
SESSION_LIST_LIMIT=50
SESSION_TITLE_MAX_CHARS=80

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
QDRANT_COLLECTION=tech_docs
RECALL_TOP_K=200
RETRIEVE_TOP_K=5
RETRIEVE_SCORE_THRESHOLD=0
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RERANKER_ENABLED=1
RERANKER_MODEL=jinaai/jina-reranker-v3
RERANKER_PRELOAD=1
RERANKER_TRUST_REMOTE_CODE=1
RERANKER_DTYPE=auto
RERANKER_MAX_DOCUMENTS_PER_CALL=64

HISTORY_RECENT_TURNS=16
HISTORY_COMPACT_AFTER_TURNS=40
HISTORY_MAX_MESSAGES=120
MESSAGE_MAX_CHARS=8000
CONVERSATION_SUMMARY_MAX_CHARS=6000
SUMMARY_HISTORY_MAX_CHARS=20000
SUMMARY_MAX_TOKENS=1200
SEARCH_QUERY_MAX_CHARS=3000

INTENT_ROUTER_ENABLED=1
INTENT_LLM_FALLBACK=1
INTENT_LLM_HISTORY_MAX_CHARS=4000
INTENT_LLM_SUMMARY_MAX_CHARS=2500
INTENT_LLM_MAX_TOKENS=120
INTENT_EMBEDDING_HISTORY_MAX_CHARS=5000
INTENT_EMBEDDING_SUMMARY_MAX_CHARS=2500
INTENT_EMBEDDING_TEXT_MAX_CHARS=7000
INTENT_EMBEDDING_RAG_THRESHOLD=0.38
INTENT_EMBEDDING_DIRECT_THRESHOLD=0.40
INTENT_EMBEDDING_MARGIN=0.06
```

`LLM_PROVIDER=openai_compatible` works with OpenAI-compatible cloud APIs and
local vLLM. For Anthropic Claude, use `LLM_PROVIDER=anthropic` and
`LLM_BASE_URL=https://api.anthropic.com/v1`. Embeddings remain local through
SentenceTransformers.

Leave `LLM_HEALTH_PATH` blank to use provider-specific health defaults:
OpenAI-compatible providers use `GET /models`, while Anthropic uses
`POST /messages/count_tokens`.

LLM chat requests retry transient provider errors (`429`, `502`, `503`, `504`)
with exponential backoff. Keep `DEBUG=0` outside local development so API errors
return generic messages while details stay in server logs.

When intent routing chooses RAG, the pipeline first recalls `RECALL_TOP_K`
chunks from Qdrant, reranks those candidates with the multilingual
`jinaai/jina-reranker-v3` cross-encoder, then keeps `RETRIEVE_TOP_K` chunks for
the LLM prompt and response references. The browser UI exposes both values as
`Recall K` and `Rerank K`; defaults are `200` and `5`. Set
`RETRIEVE_SCORE_THRESHOLD` above `0` to drop low-scoring vector results before
reranking. With `CUDA=TRUE` (the default), the BGE embedding model and Jina
reranker prefer CUDA when PyTorch can see a compatible NVIDIA GPU; if CUDA is
not visible or model placement fails, they log the fallback and continue on
CPU. Set `CUDA=FALSE` to force CPU. With `RERANKER_PRELOAD=1`, the API loads
and warms the reranker during startup through Jina's native
`AutoModel.rerank()` interface, so model download or load failures happen
before the service accepts requests.
`jina-reranker-v3` is a listwise reranker, so candidates are reranked in
batches of `RERANKER_MAX_DOCUMENTS_PER_CALL` when recall returns more than the
model should process in one call.

The default Compose configuration exposes all visible NVIDIA GPUs to the API
container with `gpus: all`, so a plain startup uses CUDA when Docker can provide
GPU devices:

```bash
docker compose up --build
```

For a portable startup that tries GPU first and retries on CPU if Docker rejects
GPU device allocation, use the wrapper:

```bash
scripts/compose_up.sh up --build
```

On machines where you want to force CPU from the start, use the CPU override:

```bash
CUDA=FALSE docker compose -f compose.yaml -f compose.cpu.yaml up --build
```

If Docker exposes GPU devices but PyTorch cannot use them, or the GPU
architecture is too old for the installed CUDA/PyTorch build, the application
falls back to CPU after it starts.

Account login and PostgreSQL-backed sessions are always enabled. By default,
startup creates an administrator account if it does not already exist:

```text
username: admin
password: 123456
```

Change `AUTH_DEFAULT_ADMIN_PASSWORD` before exposing the app beyond local
development. Existing users are not overwritten on restart. You can also add
non-admin initial users through
`AUTH_BOOTSTRAP_USERS`, for example:

```bash
AUTH_BOOTSTRAP_USERS=analyst:change-me
```

Bootstrap users are inserted only when they do not already exist. Passwords are
stored as PBKDF2-SHA256 hashes, login bearer tokens are stored as SHA-256
hashes, and each user's chat sessions, messages, retrieved references, route
metadata, and compacted summary are stored in PostgreSQL. The browser opens to
the sign-in form and only shows the RAG workspace after a valid login.
Administrators see an Admin panel for creating users, importing users from CSV,
deleting users, resetting passwords, toggling admin access, and clearing a
user's chat data. CSV imports must contain exactly two columns named `email` and
`passwd`; rows with missing values, wrong headers, or extra columns are
rejected. `/rag` accepts a `session_id` and manages history server-side.

## Replace the Knowledge Base

The Markdown files under `data/docs/` provide the runtime domain content.
Ingestion scans that tree recursively, so nested topic folders are supported.
The committed files cover a generated restaurant/business case corpus about
家是本 and 朱剑秋. Because these topics are unlikely to be memorized well by
general models, they are a better fit for RAG evaluation than common SQL or
database-system facts.

To use another subject:

1. Replace or edit the Markdown files under `data/docs/`.
2. Rebuild the Qdrant collection with `python scripts/ingest_docs.py --recreate`.
3. Update the corpus keyword hints in `app/intent_router.py`
   (`DOMAIN_RAG_PHRASES` and `DOMAIN_RAG_PATTERNS`) and the LLM fallback
   corpus description if first-pass and fallback intent routing should
   recognize the new topic.
4. Ask questions about the new corpus through the same UI or `/rag` API.

## Restricted Network Setup

Docker daemon proxy settings only help image pulls. Runtime containers need
their own proxy or mirror settings in `.env`.

For a Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com
```

For host-side Mihomo, enable Allow LAN / bind to `0.0.0.0`, then set:

```bash
DOCKER_HTTP_PROXY=http://host.docker.internal:7890
DOCKER_HTTPS_PROXY=http://host.docker.internal:7890
DOCKER_NO_PROXY=postgres,qdrant,vllm,api,localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
```

Do not use `127.0.0.1` for the proxy host inside containers; it points to the
container itself.

## Offline Models

You can mount local model directories through `./models:/models:ro`:

```text
models/
  bge-small-en-v1.5/
  qwen2.5-7b-instruct/
```

Then configure:

```bash
EMBEDDING_MODEL=/models/bge-small-en-v1.5
VLLM_MODEL=/models/qwen2.5-7b-instruct
LLM_MODEL=qwen2.5-7b-instruct
```

## API Usage

Health check:

```bash
curl http://localhost:8080/health
```

Log in first and use the returned bearer token:

```bash
TOKEN="$(
  curl -s http://localhost:8080/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"123456"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
)"
```

Authenticated health details:

```bash
curl http://localhost:8080/health/details \
  -H "Authorization: Bearer ${TOKEN}"
```

RAG request:

```bash
SESSION_ID="$(
  curl -s http://localhost:8080/sessions \
    -H "Authorization: Bearer ${TOKEN}" \
    -X POST \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'
)"

curl http://localhost:8080/rag \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"question\": \"When should I choose DuckDB over ClickHouse?\",
    \"recall_top_k\": 200,
    \"top_k\": 5
  }"
```

Response fields:

- `answer`: generated response.
- `contexts`: reranked chunks with `source`, `chunk_id`, vector `score`,
  optional `rerank_score`, `content_type`, `headings`, line bounds, and
  `h1`/`h2`/`h3` metadata.
- `conversation_summary`: compact memory for future turns.
- `compacted_history_messages`: number of old messages merged into memory.
- `used_rag`: whether Qdrant retrieval was used for this answer.
- `route` and `route_reason`: intent-router decision metadata.

## Manual Development

Set up Python dependencies with uv:

```bash
env UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.api.txt
```

Or use an existing Conda environment:

```bash
conda activate rag_llm
pip install -r requirements.txt
```

`requirements.txt` lists direct development dependencies only; transitive pins
are intentionally not committed as a `pip freeze` dump.

For API development with a configured LLM API and Docker-managed Qdrant, the
smaller runtime dependency set is:

```bash
pip install -r requirements.api.txt
```

Start local services manually:

```bash
bash scripts/start_qdrant.sh
bash scripts/start_vllm.sh
```

Ingest Markdown:

```bash
python scripts/ingest_docs.py --recreate
```

Smoke-test retrieval:

```bash
python scripts/test_retrieve.py
```

Smoke-test Markdown chunking:

```bash
python scripts/test_chunking.py
```

Smoke-test incremental document replacement:

```bash
python scripts/test_vector_store.py
```

Smoke-test intent routing:

```bash
python scripts/test_intent_router.py
```

Smoke-test cross-encoder reranker ordering with a fake model:

```bash
python scripts/test_reranker.py
```

Smoke-test prompt budgeting and history trimming:

```bash
python scripts/test_prompt_budget.py
```

Smoke-test configuration wiring:

```bash
python scripts/test_settings.py
```

Run FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Run a quick syntax check before pushing:

```bash
python -m compileall app scripts
```

## Document Corpus

Markdown files in `data/docs/` are the RAG source of truth. The bundled sample
files cover a generated Chinese restaurant/business case: company overview,
FAQ, menu and pricing, customer reviews, Bilibili comments, social-media
archives, financial simulation, a timeline, a profile of 朱剑秋, the Yongge
livestream incident, and a song document. Retrieval uses vector recall plus
optional reranking; no BM25 keyword index is currently included.

The keyword intent layer contains domain hints for this bundled corpus in
`app/intent_router.py`. Those hints only decide whether to use RAG; they do not
rank documents. Replace them when swapping in a different corpus.

Chunking is Markdown-aware and metadata-driven: Markdown blocks are parsed,
headings are stored as `h1`/`h2`/`h3` payload metadata, and text, code, and
tables are chunked separately. Heading context is included in embedding input
but kept separate from stored chunk text to avoid duplicating titles in every
chunk. Oversized text chunks use an effective chunk budget that leaves room for
overlap, while fenced code chunks preserve complete fences.

After adding or editing documents, run an incremental ingest:

```bash
python scripts/ingest_docs.py
```

Each current Markdown file replaces all previously indexed chunks with the same
`source`, so edited or shortened files do not leave stale chunks behind. Use
`--recreate` when deleting documents, replacing the whole corpus, or changing
the embedding model's vector size:

```bash
python scripts/ingest_docs.py --recreate
```

The equivalent Compose setting is `RECREATE_COLLECTION=1`.

## Before Pushing to GitHub

Run these checks from the repository root:

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

Expected ignored local paths may include `.env`, `.vscode/`, `qdrant_storage/`,
`models/`, and `__pycache__/`. Do not add those files. New source files such as
`app/intent_router.py`, `app/static/index.html`, data docs, and scripts should
be tracked.

## Git Hygiene

Do not commit `.env`, API keys, Hugging Face tokens, model weights, Qdrant
storage, cache directories, virtual environments, or logs. Runtime state such as
`qdrant_storage/`, `models/`, `.cache/`, local database files, and `.env` is
intentionally ignored.
