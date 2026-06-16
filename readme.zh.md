# Knowledge Base Assistant

Knowledge Base Assistant 是一个通用知识库问答应用，用于查询可替换的本地
Markdown 文档库。系统会通过意图路由判断问题是否需要检索：需要文档依据时走
RAG，不需要检索的请求走直接对话。

Qdrant 负责向量检索，PostgreSQL 负责必需的用户账号、登录 token、用户会话、
消息和压缩后的对话记忆，SentenceTransformers 负责本地 embedding，FastAPI
同时提供 API 和浏览器界面。生成模型通过环境变量配置，默认使用
OpenAI-compatible API，也支持 Anthropic；本地 vLLM 作为可选 Docker Compose
profile 保留。

仓库自带一组数据库系统 Markdown 文档作为示例语料。应用和意图路由不绑定数据
库领域；替换 `data/docs/` 下的 Markdown 并重建 Qdrant collection 后即可用于
其他知识域。

## 功能概览

- Docker Compose 一键启动 PostgreSQL、Qdrant、文档 ingest 和 FastAPI。
- 必须登录账号后才能使用 RAG 系统，没有匿名访问模式。
- 默认管理员账号为 `admin / 123456`，可通过环境变量改默认密码。
- 管理员可以在前端创建用户、删除用户、重置密码、切换管理员权限、清空用户会话。
- 管理员可以上传 CSV 批量创建用户；CSV 必须只有两列，表头必须为 `email,passwd`。
- 支持 OpenAI-compatible、Anthropic 和可选本地 vLLM。
- Markdown 语料可替换，检索逻辑不依赖具体业务领域。
- 前端支持多会话、召回数量和重排后引用数量调整、引用展示和路由结果展示。
- 对话历史保存在后端，旧消息会压缩成 summary 后继续参与后续回答。
- 支持 Hugging Face cache、离线模型目录、镜像和容器运行时代理配置。

## 架构

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

主要模块：

- `app/main.py`：FastAPI 路由、健康检查、静态前端。
- `app/static/index.html`：浏览器聊天界面和管理员界面。
- `app/session_store.py`：PostgreSQL 用户、登录 token、会话、消息和 summary。
- `app/intent_router.py`：关键词、embedding 和 LLM fallback 意图路由。
- `app/rag.py`：召回、重排、prompt 构造、历史压缩。
- `app/reranker.py`：启动时预加载 Jina cross-encoder，并对召回 chunk 重排。
- `app/vector_store.py`：Markdown 切块、embedding、Qdrant collection 管理和搜索。
- `app/llm_client.py`：不同 LLM provider 的请求封装。
- `scripts/`：ingest、服务启动和 smoke test 脚本。
- `data/docs/`：被 ingest 到 Qdrant 的 Markdown 语料。

## 快速启动

准备 Docker Compose v2 和一个云端 LLM API key。复制配置：

```bash
cp .env.example .env
```

在 `.env` 中设置 `LLM_API_KEY`，必要时设置 `LLM_MODEL`。默认启动时会创建管理
员账号：

```text
username: admin
password: 123456
```

生产或公网环境必须修改 `AUTH_DEFAULT_ADMIN_PASSWORD`。如果 PostgreSQL volume 中
已经存在 `admin` 用户，启动时不会覆盖该用户当前密码。

启动服务：

```bash
docker compose up --build
```

打开：

```text
http://localhost:8080
```

使用其他宿主机端口：

```bash
APP_PORT=9000 docker compose up --build
```

修改 `.env` 后重建容器：

```bash
docker compose up -d --build --force-recreate
```

停止服务：

```bash
docker compose down
```

清空 PostgreSQL、Qdrant 和 Hugging Face cache volume：

```bash
docker compose down -v
```

## 登录和用户管理

应用现在始终要求账号登录。浏览器打开后只显示 `Account` 和 `Password` 登录框；
登录成功后才显示 RAG 工作区、会话列表和引用面板。

管理员 CSV 批量导入格式必须严格如下：

```csv
email,passwd
alice@example.com,secret
bob@example.com,another-secret
```

规则：

- 表头必须精确为 `email,passwd`。
- 每行必须只有两列。
- `email` 和 `passwd` 都不能为空。
- 表头错误、缺值、多列、空文件都会被后端拒绝。
- 导入的账号为普通用户，不会自动授予管理员权限。
- 用户名会按现有规则标准化为小写；允许字母、数字、下划线、短横线、点和 `@`。

也可以通过 `.env` 预置普通用户：

```bash
AUTH_BOOTSTRAP_USERS=analyst:change-me,viewer:change-me-too
```

这些用户只会在不存在时创建，重启不会覆盖已有用户密码。

## 配置要点

常用配置见 `.env.example`：

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=

POSTGRES_USER=kba
POSTGRES_PASSWORD=kba_password
POSTGRES_DB=kba
AUTH_DEFAULT_ADMIN_ENABLED=1
AUTH_DEFAULT_ADMIN_USERNAME=admin
AUTH_DEFAULT_ADMIN_PASSWORD=123456
AUTH_BOOTSTRAP_USERS=

QDRANT_COLLECTION=tech_docs
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RECALL_TOP_K=200
RETRIEVE_TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RERANKER_ENABLED=1
RERANKER_MODEL=jinaai/jina-reranker-v3
RERANKER_PRELOAD=1
RERANKER_TRUST_REMOTE_CODE=1
RERANKER_DTYPE=auto
RERANKER_MAX_DOCUMENTS_PER_CALL=64

INGEST_ON_STARTUP=1
RECREATE_COLLECTION=0
WAIT_FOR_LLM=0
```

当意图路由判断需要 RAG 时，系统会先从 Qdrant 召回 `RECALL_TOP_K`
个候选 chunk，再使用多语言 `jinaai/jina-reranker-v3` cross-encoder
重排，最后只保留 `RETRIEVE_TOP_K` 个 chunk 进入 LLM prompt 和引用列表。
前端对应控件为 `Recall K` 和 `Rerank K`，默认分别为 `200` 和 `5`。
`RERANKER_PRELOAD=1` 时 API 会在启动阶段加载并预热 reranker，模型下载或加载
错误会在服务接受请求前暴露，而不是第一次用户提问时才失败。代码使用 Jina
模型原生的 `AutoModel.rerank()` 接口执行重排。
`jina-reranker-v3` 是 listwise reranker；召回数量超过
`RERANKER_MAX_DOCUMENTS_PER_CALL` 时会分批重排，再按分数全局排序。

`LLM_PROVIDER=openai_compatible` 适用于 OpenAI-compatible 云端 API 和本地 vLLM。
Anthropic 使用：

```bash
LLM_PROVIDER=anthropic
LLM_BASE_URL=https://api.anthropic.com/v1
```

`LLM_HEALTH_PATH` 留空时会使用 provider 默认健康检查：
OpenAI-compatible 使用 `GET /models`，Anthropic 使用
`POST /messages/count_tokens`。

## 替换知识库

`data/docs/` 下的 Markdown 文件是 RAG 语料来源。替换知识库步骤：

1. 替换或编辑 `data/docs/` 下的 Markdown 文件。
2. 重建 Qdrant collection：

   ```bash
   python scripts/ingest_docs.py --recreate
   ```

3. 通过浏览器或 `/rag` API 提问。

Ingest 会递归扫描子目录。增量 ingest 会用当前 Markdown 文件替换同一 source 的旧
chunk，避免修改或缩短文件后留下陈旧 chunk。删除文档、整体替换语料或修改
embedding 模型维度时，请使用 `--recreate`。

## API 使用

公开健康检查：

```bash
curl http://localhost:8080/health
```

先登录获取 token：

```bash
TOKEN="$(
  curl -s http://localhost:8080/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"123456"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
)"
```

带登录 token 的详细健康检查：

```bash
curl http://localhost:8080/health/details \
  -H "Authorization: Bearer ${TOKEN}"
```

创建会话并调用 RAG：

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

`/rag` 的历史由服务端根据 `session_id` 从 PostgreSQL 管理，客户端不需要传完整
history。

## 本地开发

推荐使用 uv 创建 Python 3.12 环境：

```bash
env UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.api.txt
```

也可以使用已有 Conda 环境：

```bash
conda activate rag_llm
pip install -r requirements.txt
```

常用检查：

```bash
python -m compileall app scripts
python scripts/test_settings.py
python scripts/test_session_store.py
python scripts/test_prompt_budget.py
python scripts/test_chunking.py
python scripts/test_intent_router.py
python scripts/test_reranker.py
python scripts/test_vector_store.py
```

手动运行 FastAPI：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 受限网络和离线模型

Docker daemon proxy 只影响镜像拉取，运行中的容器需要在 `.env` 中配置运行时代理。

Hugging Face 镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

宿主机 Mihomo 示例：

```bash
DOCKER_HTTP_PROXY=http://host.docker.internal:7890
DOCKER_HTTPS_PROXY=http://host.docker.internal:7890
DOCKER_NO_PROXY=postgres,qdrant,vllm,api,localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
```

离线模型可挂载到 `./models:/models:ro`：

```text
models/
  bge-small-en-v1.5/
  qwen2.5-7b-instruct/
```

然后配置：

```bash
EMBEDDING_MODEL=/models/bge-small-en-v1.5
VLLM_MODEL=/models/qwen2.5-7b-instruct
LLM_MODEL=qwen2.5-7b-instruct
```

## 仓库内容和安全提示

应提交源码、脚本、Docker 配置、`.env.example`、依赖文件和 Markdown 语料。不要
提交 `.env`、API key、Hugging Face token、模型权重、PostgreSQL 数据、Qdrant
存储、cache、日志或虚拟环境。
