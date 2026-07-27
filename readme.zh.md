# Knowledge Base Assistant

Knowledge Base Assistant 是一个通用知识库问答应用，用于查询可替换的本地
Markdown 文档库。系统会通过意图路由判断问题是否需要检索：需要文档依据时走
RAG，不需要检索的请求走直接对话。

Qdrant 负责向量检索，本地 BM25 索引负责关键词召回，Redis 负责活跃Session热路径，
PostgreSQL 负责用户账号、设置和完整对话归档，SentenceTransformers 负责
本地 embedding，FastAPI 同时提供 API 和浏览器界面。生成模型通过环境变量配置，
默认使用 OpenAI-compatible API，也支持 Anthropic；本地 vLLM 作为可选 Docker
Compose profile 保留。

仓库当前在`data/docs/RAGBench/`中提供RAGBench `cuad`测试集的102份长合同文档；
对应的原始记录和500条标注query位于`RAGBench_Eval/`。做文档召回评测时应打开
`RAG-only`，把意图路由准确率与检索召回率分开测量。

## 功能概览

- Docker Compose 一键启动 Redis、PostgreSQL、PgBouncer、Qdrant、本机 MinIO S3
  对象存储、Ray model worker 和 FastAPI。
- 必须登录账号后才能使用 RAG 系统，没有匿名访问模式。
- 默认管理员账号为 `admin / 123456`，可通过环境变量改默认密码。
- 管理员可以在前端创建用户、删除用户、重置密码、切换管理员权限、清空用户会话。
- 管理员可以上传 CSV 批量创建用户；CSV 必须只有两列，表头必须为 `email,passwd`。
- 支持 OpenAI-compatible、Anthropic 和可选本地 vLLM。
- 默认使用 `jinaai/jina-embeddings-v5-text-small`，并为查询、文档和意图分类分别使用
  `retrieval` task + `query`/`document` prompt，以及 `classification` task。
- Markdown语料可替换；`RAG-only`评测模式可将检索效果与应用专用意图提示分开测量。
- 前端支持多会话、BM25 K、Cosine K、RRF K、Final K 调整、引用展示和路由结果展示。
- 对话历史保存在后端，只在估算 prompt 接近当前 LLM 上下文窗口时才压缩成
  summary。
- 唯一 superuser 可在 Admin UI 中修改全局 LLM provider、API URL、模型名、
  API key 和上下文窗口大小，保存后不需要 rebuild。
- 支持 Hugging Face cache、离线模型目录、镜像和容器运行时代理配置。
- 支持本地源码 Code Search：Python/C++ AST 符号抽取、CodeBERT 文件/函数向量、
  PostgreSQL 原文存储、Qdrant 代码 collection 和静态调用图。

## 架构

```text
Browser UI
   |
FastAPI /auth, /sessions, /rag
   |-- Redis active sessions + pending archive stream
   |-- PostgreSQL users, settings, and complete session archive
   |
IntentRouter
   |-- keyword rules and previous-route strong follow-up state
   |-- Jina classification similarity over single-intent anchors
   |-- configured LLM fallback with previous route state
   |
RAGPipeline or Direct Chat
   |-- BM25 keyword recall + Qdrant vector recall -> RRF fusion
   |-- optional Jina cross-encoder reranking
   |-- OpenAI-compatible or Anthropic LLM API
   |
Answer + retrieved references + compacted conversation memory
```

主要模块：

- `app/main.py`：FastAPI 路由、健康检查、静态前端。
- `app/static/index.html`：浏览器聊天界面和管理员界面。
- `app/session_store.py`：PostgreSQL 用户、登录 token、会话、消息、运行时
  LLM 设置、路由元数据和 summary。
- `app/redis_session_store.py`：活跃Session、auth token缓存、Pending overlay、
  Archive Stream原子写入、一致性冷加载，以及Redis不可用时严格返回HTTP 503。
- `app/session_archive.py`：独立Redis Stream consumer；PG幂等提交成功后再清理
  Pending并确认Stream事件。
- `app/intent_router.py`：关键词/状态机、Jina classification embedding 和
  tagged LLM fallback 意图路由。
- `app/rag.py`：混合召回、RRF 融合、重排、prompt 构造、基于上下文预算的历史压缩。
- `app/reranker.py`：Jina cross-encoder 重排；启用 Ray 时通过 actor 调用。
- `app/vector_store.py`：Markdown 切块、Jina embeddings v5 task 路由、BM25
  索引、Qdrant collection 管理、向量搜索和 RRF 融合。
- `app/code_indexer.py`：基于 tree-sitter 的 Python/C++ 源码解析、CodeBERT
  embedding、Qdrant 代码索引和 PostgreSQL 代码文件/函数持久化。
- `app/code_retrieval.py`：CodeBERT-first 的源码检索、repo-wide 函数召回、
  lexical 候选扩展和显式 text-only 降级报告。
- `app/call_graph.py`：AST 调用边抽取和 networkx 有向调用图查询。
- `app/llm_client.py`：不同 LLM provider 的请求封装。
- `scripts/`：ingest、服务启动、smoke test 和 intent-router A/B 评估脚本。
- `data/docs/`：被 ingest 到 Qdrant 的 Markdown 语料。
- `data/eval/`：带标签的 intent-router 评估样本，包含双语/跨语种切片。

## 快速启动

准备 Docker Compose v2 和一个云端 LLM API key。如果希望 Docker 容器使用
CUDA，还需要兼容的 NVIDIA GPU、较新的 NVIDIA 驱动和 NVIDIA Container
Toolkit；过老的显卡可能不支持当前 PyTorch/Transformers CUDA 构建。复制配置：

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
已经存在 `admin` 用户，启动时不会覆盖该用户当前密码。启动时创建的默认
administrator 同时是唯一 superuser；只有这个 superuser 能在前端 Admin 面板修改
全局 LLM 配置，包括 API 格式（`openai_compatible` 或 `anthropic`）、API URL、
模型名、上下文窗口大小和 API key。前端保存后配置写入 PostgreSQL 并覆盖 `.env`
中的 LLM 默认值，不需要 rebuild 容器；`.env` 仍作为首次启动和未配置时的
fallback。API key 不会回显给浏览器，保存时 key 输入框留空表示保留当前 key。

启动 Redis、Session Archiver、PostgreSQL、PgBouncer、Qdrant、本机 MinIO S3
存储和 API：

```bash
docker compose up --build
```

打开：

```text
http://localhost:8080
```

本机 S3 控制台：

```text
http://localhost:9001
```

默认 MinIO 账号密码是 `minioadmin` / `minioadmin`；离开本机 demo 前需要修改。

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

清空 Redis、PostgreSQL、Qdrant、MinIO 和 Hugging Face cache volume：

```bash
docker compose down -v
```

默认 `compose.yaml` 会启动 Redis、Session Archiver、PostgreSQL、PgBouncer、Qdrant、
MinIO、Ray head、在线三个Ray model worker和`main` API服务，并增加三个有序的
startup服务：`minio-init`、`ray-worker-embedding-bootstrap`和`docs-indexer`。
bootstrap worker只服务独立的`kba_embedding_bootstrap`离线Actor；`docs-indexer`负责同步MinIO、Qwen3
tokenizer切分、跨文档64条批量embedding、候选collection验证和alias原子切换。
成功后`docs-indexer`显式销毁detached离线Actor，bootstrap worker再退出并销毁
CUDA context，随后同时启动在线embedding worker和两个reranker replica。`main`
提前启动，但必须同时通过本轮UUID marker和在线模型readiness门槛后才完成FastAPI
lifespan，因此不会暴露半完成索引或未预热模型。

`docs-update` profile另外提供按需启动的`ray-worker-embedding-ingest-cpu`和
`docs-updater`。它们只负责运行中手动增量更新，使用独立的CPU embedding actor，
不会停止或复用在线GPU embedding/reranker actor。

`main`由显式安装的Uvicorn ASGI server启动，
所有应用数据库连接先进入 transaction-mode PgBouncer，再复用少量 PostgreSQL
server connection。在线`ray-worker-embedding-1`带Ray resource
`ray_worker_embedding`，在actor内执行最多16条、等待10ms的跨请求query
micro-batching；`ray-worker-reranker-1` 和
`ray-worker-reranker-2` 分别带 `ray_worker_reranker_1` 和
`ray_worker_reranker_2`，用于两个 reranker actor replica。`main` 容器只连接
`ray://ray-head:10001`，默认 `RAY_LOCAL_FALLBACK=0`，不会把 embedding/reranker
模型加载回 API 进程。Compose 还会启动 `autoheal`，它根据 Docker healthcheck
重启适合单容器自愈的服务，例如`redis`、`session-archiver`、`pgbouncer`、
`qdrant`和`main`；Ray head/worker不交给
autoheal 单独重启，避免把已有 Ray cluster 连接关系打散。

Docker 服务架构：

```mermaid
flowchart TD
    User["浏览器 / curl<br/>localhost:8080"] --> Main["main<br/>FastAPI + UI + RAG 编排<br/>仅作为 Ray client"]

    subgraph Compose["Docker Compose 网络"]
        Main
        PG["postgres<br/>PostgreSQL<br/>用户 / 会话 / 消息 / 设置"]
        PgBouncer["pgbouncer<br/>transaction connection pool<br/>复用 PostgreSQL server connections"]
        Redis["redis<br/>32 GB volatile-lru热Session<br/>AOF everysec + 不可淘汰Pending/Stream"]
        Archiver["session-archiver<br/>Redis consumer group<br/>PG幂等批量归档"]
        Qdrant["qdrant<br/>Qdrant<br/>向量 collection + alias"]
        MinIO["minio<br/>S3 兼容对象存储<br/>版本化文档"]
        MinIOInit["minio-init<br/>一次性 bucket 初始化<br/>开启 versioning"]
        RayHead["ray-head<br/>Ray 控制面<br/>ray://ray-head:10001"]
        BootstrapEmbed["ray-worker-embedding-bootstrap<br/>临时离线GPU worker<br/>embedding batch: 64"]
        DocsIndexer["docs-indexer<br/>一次性Qwen3 chunker<br/>S3同步 + 候选索引 + alias切换"]
        CpuIngest["ray-worker-embedding-ingest-cpu<br/>docs-update profile<br/>CPU-only Ray worker"]
        DocsUpdater["docs-updater<br/>docs-update profile<br/>manifest diff + 候选协调"]
        EmbedWorker["ray-worker-embedding-1<br/>Ray worker: ray_worker_embedding<br/>kba_embedding actor<br/>query batch: 最大16 / 等待10 ms"]
        RerankWorker1["ray-worker-reranker-1<br/>Ray worker<br/>resource: ray_worker_reranker_1<br/>承载 kba_reranker_1 actor"]
        RerankWorker2["ray-worker-reranker-2<br/>Ray worker<br/>resource: ray_worker_reranker_2<br/>承载 kba_reranker_2 actor"]
        BM25Manager["main BM25S manager<br/>active + candidate双缓冲<br/>原子引用交换"]
        HealthSupervisor["main health supervisor<br/>redis / embedding / qdrant / reranker<br/>独立轻量探测"]
        Autoheal["autoheal<br/>监控 Docker healthcheck<br/>重启 unhealthy 容器"]
        VLLM["vllm<br/>可选 local-llm profile<br/>OpenAI-compatible LLM 服务"]
    end

    Main --> Redis
    Main --> PgBouncer
    Redis --> Archiver
    Archiver --> PgBouncer
    PgBouncer --> PG
    Main --> Qdrant
    Main --> MinIO
    Main --> RayHead
    Main --> BM25Manager
    MinIOInit --> MinIO
    MinIOInit --> DocsIndexer
    RayHead --> BootstrapEmbed
    DocsIndexer --> MinIO
    DocsIndexer --> BootstrapEmbed
    DocsIndexer --> Qdrant
    DocsIndexer --> StartupState["startup_state<br/>docs-ready / docs-failed / offline-stopped UUID"]
    StartupState --> Main
    StartupState --> BootstrapEmbed
    BootstrapEmbed -- "退出并释放CUDA cache" --> EmbedWorker
    BootstrapEmbed -- "退出并释放GPU" --> RerankWorker1
    BootstrapEmbed -- "退出并释放GPU" --> RerankWorker2
    RayHead --> EmbedWorker
    RayHead --> RerankWorker1
    RayHead --> RerankWorker2
    RayHead -. "docs-update profile" .-> CpuIngest
    DocsUpdater -. "docs-update profile" .-> CpuIngest
    DocsUpdater -. "同步 + 版本manifest" .-> MinIO
    DocsUpdater -. "候选向量" .-> Qdrant
    DocsUpdater -. "prepare / commit" .-> Main
    DocsUpdater -. "本轮ready / failed" .-> StartupState
    Main --> HealthSupervisor
    HealthSupervisor -. "collection / alias 元信息" .-> Qdrant
    HealthSupervisor -. "PING" .-> Redis
    HealthSupervisor -. "embedding actor readiness" .-> EmbedWorker
    HealthSupervisor -. "并行 reranker readiness<br/>任一 replica 健康即可" .-> RerankWorker1
    HealthSupervisor -. "并行 reranker readiness<br/>任一 replica 健康即可" .-> RerankWorker2
    Autoheal -. healthcheck .-> Main
    Autoheal -. healthcheck .-> Redis
    Autoheal -. healthcheck .-> Archiver
    Autoheal -. healthcheck .-> PgBouncer
    Autoheal -. healthcheck .-> Qdrant
    Main -. 可选 .-> VLLM
    Main --> CloudLLM["云端 LLM API<br/>OpenAI-compatible / Anthropic"]

    subgraph Volumes["Docker volume 和 bind mount"]
        PGVol["postgres_data"]
        RedisVol["redis_data<br/>AOF"]
        QVol["qdrant_storage"]
        S3Vol["minio_data"]
        HFVol["huggingface_cache"]
        StartupVol["startup_state<br/>本轮启动握手"]
        DataBind["./data:/app/data:ro"]
        ModelsBind["./models:/models:ro"]
    end

    PG --> PGVol
    Redis --> RedisVol
    Qdrant --> QVol
    MinIO --> S3Vol
    Main --> DataBind
    Main --> HFVol
    Main --> StartupVol
    Main --> ModelsBind
    EmbedWorker --> HFVol
    EmbedWorker --> ModelsBind
    DocsIndexer --> DataBind
    DocsIndexer --> HFVol
    DocsIndexer --> StartupVol
    BootstrapEmbed --> HFVol
    BootstrapEmbed --> StartupVol
    RerankWorker1 --> HFVol
    RerankWorker1 --> ModelsBind
    RerankWorker2 --> HFVol
    RerankWorker2 --> ModelsBind
    CpuIngest --> HFVol
    CpuIngest --> ModelsBind
    CpuIngest --> StartupVol
    DocsUpdater --> DataBind
    DocsUpdater --> HFVol
    DocsUpdater --> StartupVol
```

默认Compose两阶段启动和镜像/配置绑定的文档初始化流程：

```mermaid
flowchart TD
    A["docker compose up --build"] --> B["构建 main 镜像"]
    B --> C["COPY data/docs并写入/app/.image_build_id"]
    A --> Infra["并行启动main、PostgreSQL/Redis、qdrant、minio和ray-head"]
    Infra --> Bucket["minio-init创建版本化bucket"]
    Infra --> Offline["启动ray-worker-embedding-bootstrap<br/>写入本轮run UUID"]
    Offline --> MainWait["main已启动，等待匹配的offline-stopped UUID"]
    Bucket --> Indexer["启动一次性docs-indexer"]
    Offline --> Indexer
    Indexer --> Marker{"image build id和索引配置指纹都未变化?"}
    Marker -- "是" --> Ready["写入匹配的ready UUID"]
    Marker -- "否" --> Sync["同步当前Markdown集合到MinIO<br/>包含删除检测"]
    Indexer -. "错误" .-> Failed["写入匹配的docs-failed UUID<br/>main立即终止启动等待"]
    Sync --> Chunk["CPU Qwen3 tokenizer切分<br/>body 1600 + overlap 100/100"]
    Chunk --> Batch["跨文档离线embedding<br/>batch size 64"]
    Batch --> Candidate["写入并验证Qdrant候选collection"]
    Candidate --> Alias["原子切换collection alias<br/>写S3 manifest"]
    Alias --> Ready
    Ready --> Stop["bootstrap embedding worker退出<br/>销毁CUDA context/cache"]
    Stop --> Online["启动在线embedding worker<br/>动态batch 16 / 10ms"]
    Stop --> RR1["启动reranker replica 1"]
    Stop --> RR2["启动reranker replica 2"]
    Stop --> LoadR1["加载reranker-1权重"]
    RR1 --> LoadR1
    LoadR1 --> LoadR2["加载reranker-2权重"]
    RR2 --> LoadR2
    LoadR2 --> LoadE["加载online embedding权重"]
    Online --> LoadE
    LoadE --> Capacity["三路并发最大容量验证<br/>Embedding: 16 x 3000 tokens<br/>每个Reranker: 64 x 1800-token文档 + 512-token Query"]
    Capacity -. "任一路失败" .-> NotReady["启动失败<br/>Retrieval不进入Ready"]
    Capacity --> Empty["释放容量测试产生的临时CUDA cache"]
    Empty --> PerfE["Embedding代表性性能预热<br/>动态batch 16 x 256 tokens，共2轮"]
    PerfE --> PerfR1["Reranker-1代表性性能预热<br/>64 x 1800-token文档 + 256-token Query"]
    PerfR1 --> PerfR2["Reranker-2代表性性能预热<br/>64 x 1800-token文档 + 256-token Query"]
    PerfR2 --> API["Uvicorn/FastAPI ready"]
```

容量阶段会先提交Embedding和两个Reranker的三路最大工作负载，再等待结果，用于覆盖
不同请求导致三路GPU推理同时运行的真实峰值。每个Actor都会报告allocated、
reserved和peak CUDA显存。容量验证结束后释放其临时allocator cache，再串行执行
代表性性能预热，仅保留生产shape对应的缓存。Reranker性能预热会让全部64个RRF
候选使用assembled Chunk的1,800-token硬上限，并搭配256-token Query。加载、容量
验证或代表性预热任一失败时，FastAPI lifespan不会完成，也不会对外宣称
Retrieval Ready。

普通`docker compose restart`不会重新扫描本地Markdown。新的image build id或索引
配置指纹变化会让`docs-indexer`执行同步和manifest diff；只有语料或索引配置变化时
才重建。镜像会COPY `data/docs`，因此语料变化会使构建层失效并产生新build id；
同镜像重启不会同步或diff本地Markdown。BM25S只存在于进程内存，因此单独重启
`main`会从已经提交的S3 active manifest重建active BM25S，但不会重新embedding、
创建collection或移动Qdrant alias。

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
DEBUG=0
CUDA=TRUE
DOCS_INIT_BUILD_ID=auto

LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=
LLM_MAX_TOKENS=4096
LLM_CONTEXT_MAX_TOKENS=256000
LLM_CONTEXT_SAFETY_MARGIN_TOKENS=8192
LLM_CONTEXT_PROMPT_OVERHEAD_TOKENS=2048

POSTGRES_USER=kba
POSTGRES_PASSWORD=kba_password
POSTGRES_DB=kba
PGBOUNCER_IMAGE=edoburu/pgbouncer:v1.25.2-p0
REDIS_IMAGE=redis:8.2.7-alpine3.22
REDIS_MAXMEMORY=32gb
REDIS_CONTAINER_MEMORY_LIMIT=48g
REDIS_CONTAINER_MEMORY_RESERVATION=36g
PGBOUNCER_POOL_MODE=transaction
PGBOUNCER_MAX_CLIENT_CONN=200
PGBOUNCER_DEFAULT_POOL_SIZE=20
PGBOUNCER_MIN_POOL_SIZE=2
PGBOUNCER_RESERVE_POOL_SIZE=5
PGBOUNCER_MAX_PREPARED_STATEMENTS=100
REDIS_SESSION_ENABLED=1
REDIS_SESSION_TTL_SECONDS=604800
REDIS_ARCHIVE_BATCH_SIZE=200
REDIS_ARCHIVE_BACKLOG_MAX=100000
AUTOHEAL_IMAGE=willfarrell/autoheal:1.2.0
AUTOHEAL_INTERVAL=10
AUTOHEAL_START_PERIOD=30
AUTH_DEFAULT_ADMIN_ENABLED=1
AUTH_DEFAULT_ADMIN_USERNAME=admin
AUTH_DEFAULT_ADMIN_PASSWORD=123456
AUTH_BOOTSTRAP_USERS=

QDRANT_COLLECTION=tech_docs
DOCS_SOURCE=s3
DOCS_DIR=/app/data/docs
DOCS_S3_BUCKET=kba-docs
DOCS_S3_PREFIX=docs
DOCS_S3_ENDPOINT_URL=http://minio:9000
DOCS_S3_REGION=us-east-1
DOCS_S3_ACCESS_KEY_ID=minioadmin
DOCS_S3_SECRET_ACCESS_KEY=minioadmin
DOCS_S3_FORCE_PATH_STYLE=1
DOCS_S3_REQUIRE_VERSIONING=1
DOCS_S3_RETAIN_VERSIONS=5
DOCS_S3_PROCESSING_RETAIN_VERSIONS=6
DOCS_S3_MANIFEST_PREFIX=_kba/manifests/docs
DOCS_INIT_DELETE_REMOVED=1
QDRANT_RETAIN_VERSIONS=2
QDRANT_PROCESSING_RETAIN_VERSIONS=3
EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small
EMBEDDING_TRUST_REMOTE_CODE=1
EMBEDDING_QUERY_TASK=retrieval
EMBEDDING_PASSAGE_TASK=retrieval
EMBEDDING_CLASSIFICATION_TASK=classification
EMBEDDING_QUERY_PROMPT_NAME=query
EMBEDDING_PASSAGE_PROMPT_NAME=document
EMBEDDING_CLASSIFICATION_PROMPT_NAME=
EMBEDDING_DYNAMIC_BATCH_ENABLED=1
EMBEDDING_DYNAMIC_BATCH_MAX_SIZE=16
EMBEDDING_DYNAMIC_BATCH_WAIT_MS=10
EMBEDDING_OFFLINE_BATCH_SIZE=64
MODEL_WARMUP_CAPACITY_ENABLED=1
MODEL_WARMUP_TIMEOUT_SECONDS=900
EMBEDDING_WARMUP_CAPACITY_TOKENS=3000
EMBEDDING_WARMUP_REPRESENTATIVE_TOKENS=256
EMBEDDING_WARMUP_ROUNDS=2
RERANKER_WARMUP_CAPACITY_QUERY_TOKENS=512
RERANKER_WARMUP_REPRESENTATIVE_QUERY_TOKENS=256
RERANKER_WARMUP_REPRESENTATIVE_DOCUMENT_TOKENS=1800
RERANKER_WARMUP_ROUNDS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BM25_TOP_K=100
RECALL_TOP_K=100
RRF_TOP_K=64
RETRIEVE_TOP_K=5
CHUNK_TOKENIZER_MODEL=jinaai/jina-embeddings-v5-text-small
CHUNK_TOKENIZER_TRUST_REMOTE_CODE=1
CHUNK_BODY_TARGET_TOKENS=1600
CHUNK_BODY_MAX_TOKENS=1600
CHUNK_OVERLAP_TARGET_TOKENS=100
CHUNK_OVERLAP_MAX_TOKENS=100
RERANKER_ENABLED=1
RERANKER_MODEL=jinaai/jina-reranker-v3
RERANKER_PRELOAD=1
RERANKER_TRUST_REMOTE_CODE=1
RERANKER_DTYPE=auto
RERANKER_MAX_DOCUMENTS_PER_CALL=64
RAY_BOOTSTRAP_EMBEDDING_ACTOR_NAME=kba_embedding_bootstrap
RAY_RERANKER_ACTOR_REPLICAS=2

HISTORY_RECENT_TURNS=16
HISTORY_MAX_MESSAGES=0
CONVERSATION_SUMMARY_MAX_CHARS=256000
SUMMARY_HISTORY_MAX_CHARS=200000
SUMMARY_MAX_TOKENS=4096

INTENT_LLM_HISTORY_MAX_CHARS=12000
INTENT_LLM_SUMMARY_MAX_CHARS=32000
INTENT_LLM_MAX_TOKENS=512
INTENT_EMBEDDING_HISTORY_MAX_CHARS=8000
INTENT_EMBEDDING_SUMMARY_MAX_CHARS=8000
INTENT_EMBEDDING_TEXT_MAX_CHARS=12000

INGEST_ON_STARTUP=0
INGEST_USE_RAY=1
RECREATE_COLLECTION=0
WAIT_FOR_LLM=0

HEALTH_PROBE_INTERVAL_SECONDS=10
HEALTH_PROBE_DEGRADED_INTERVAL_SECONDS=3
HEALTH_PROBE_TIMEOUT_SECONDS=2
HEALTH_PROBE_FAILURE_THRESHOLD=2
HEALTH_PROBE_RECOVERY_THRESHOLD=2
```

当意图路由判断需要 RAG 时，系统会先用 BM25S 从当前生效的 Markdown chunk 召回
`BM25_TOP_K` 个关键词候选，再从 Qdrant 召回 `RECALL_TOP_K` 个余弦相似度候选，
然后用 RRF（reciprocal rank fusion）融合两路排序并保留 `RRF_TOP_K` 个候选，
再使用多语言 `jinaai/jina-reranker-v3` cross-encoder 重排，最后只保留
`RETRIEVE_TOP_K` 个 chunk 进入 LLM prompt 和引用列表。前端对应控件为 `BM25 K`、
`Cosine K`、`RRF K` 和 `Final K`，默认分别为 `100`、`100`、`64` 和 `5`。
`/health/details` 会返回当前生效的默认值，前端登录后用这些值初始化四个控件。
默认 embedding 模型是 `jinaai/jina-embeddings-v5-text-small`。文档 chunk 和
检索 query 都使用 `retrieval` task，并分别使用 `document` 和 `query` prompt；
意图路由第二层使用 `classification`。Jina encoder 有有限 token 窗口，因此第二层只使用有界的最近
history 和 summary 视图，不会直接把 256K 级别的主 summary 整段喂给 encoder。
这些 intent 预算目前是字符级保护；如果 encoder 仍然报错，router 会跳过第二层并
进入 LLM classifier，而不是让请求失败。

在线单query embedding调用会在Ray Embedding Actor内动态合批。默认最多等待10 ms，
把最多16个兼容query组成一个GPU batch；batch填满后立即执行，不继续等待deadline。
不同task、prompt和normalization mode使用独立队列，因此classification和retrieval
输入不会混批。一次性`docs-indexer`绕过在线队列，将不同文档的chunk累计成
`EMBEDDING_OFFLINE_BATCH_SIZE=64`的请求并交给bootstrap actor。alias提交后bootstrap
worker退出，在线actor使用全新的CUDA allocator，不继承离线峰值缓存。模型的32K
context上限约束的是batch内每条sequence，
不是整个GPU batch的token总和。Actor的轻量`health()`响应和检索压测报告会暴露
配置batch size、等待时间、队列深度、batch数、请求数和实际平均batch size。
`SEARCH_QUERY_MAX_CHARS=3000`仍是检索query的字符级保护，与模型token上限相互独立。
它只约束由最近用户消息组装、供BM25/Qdrant召回使用的辅助query，不会截断Session
存储内容，也不会截断最终LLM Prompt中的当前问题或历史消息。

意图路由是状态感知的，但不会把所有历史塞给每一层。第一层会把当前语料外的通用
技术/数据库问题判 direct；也只在上一轮 assistant 确实使用过检索 contexts，且
本轮是“继续讲”“后续呢？”这类短强回指时，才通过 `state_rag` 直接走 RAG。
带新技术名词的追问不会因为上一轮 RAG 被强制检索。
第二层继续使用本地缓存的 anchor 向量和 NumPy 点积，所有 `RAG_ANCHORS` /
`DIRECT_ANCHORS` 都应保持单意图 query。第三层 LLM classifier 会收到结构化的
上一轮 route state，并用
`<think>THINK_AND_JUDGEMENT</think><answer>JSON_ANS</answer>` 输出格式判断
模糊追问。
所有用户都可以在浏览器设置中打开 `RAG-only` 开关。开启后，请求会携带
`rag_only=true`，后端跳过意图路由并强制检索，route 记录为 `rag_only`。

对话 summary 的默认预算按 API 长上下文模型设置：
`LLM_CONTEXT_MAX_TOKENS=256000`、安全余量 `8192`、prompt overhead `2048`。
问题和已存储消息不再有应用层 8K/16K 字符上限，
`HISTORY_MAX_MESSAGES=0` 表示压缩前也不按消息条数截断；`RAGPipeline` 会估算
summary + 未压缩 history，并在扣除输出、当前 query、安全余量、prompt overhead 和
预期引用 token 后接近上下文上限时才触发压缩。触发后，最近
`HISTORY_RECENT_TURNS=16`轮之前的消息会合并进滚动summary；原始消息继续保存在
Redis/PostgreSQL中，只通过`compacted_message_count`从后续Prompt加载范围排除。
Intent Router只读取独立裁剪后的分类输入副本，其预算不会截断或改写Redis/PG
Session消息。唯一superuser可以在Admin UI修改运行时LLM context window，后续回答
会使用该值。

`CUDA=TRUE` 是默认值，Ray model worker 会暴露 GPU 资源，并把 1 个 Jina
embedding actor 和 2 个 Jina reranker actor replica 作为 detached named actor
常驻。`main` 容器只连接
`ray://ray-head:10001`；默认 `RAY_LOCAL_FALLBACK=0`，Ray Actor 不可用时不会在
API 进程本地加载这些模型。设置 `CUDA=FALSE` 可强制 Ray worker 和 actor 资源请求
都走 CPU。默认镜像使用 PyTorch CUDA runtime 镜像，不需要在宿主机安装 CUDA
toolkit。

文档版本闸门只覆盖query embedding、BM25/Qdrant召回和RRF。提交新版本时，新进入的
召回会排队，已经处于闸门内的召回会先排空；已经进入reranker或LLM生成的请求不在
等待计数中。RRF完成后会先释放闸门，再调用reranker，因此两个reranker replica和
direct/RAG的LLM生成都不会被索引指针交换阻塞。

`jina-reranker-v3` 是 listwise reranker；召回数量超过
`RERANKER_MAX_DOCUMENTS_PER_CALL` 时会分批重排，再按分数全局排序。默认
`RRF_TOP_K=64`与`RERANKER_MAX_DOCUMENTS_PER_CALL=64`对齐，因此默认检索只调用
一次reranker。每个组装后chunk硬上限为1800个Qwen3 token，因此64个候选最多贡献
115,200个文档token，低于reranker的131,072 token上下文，并为query、来源/标题和
模型模板保留空间；请求显式传入大于64的值时仍会安全分批。

Query 时的文档 RAG pipeline：

```mermaid
flowchart TD
    A["浏览器 / POST /rag"] --> RS{"Redis Session存储可访问?"}
    RS -- "否" --> R503["HTTP 503 + Retry-After: 3<br/>禁止返回过期PG历史"]
    RS -- "是" --> B["从Redis读取活跃Session<br/>miss时合并PG归档+Pending"]
    P["后台组件健康状态<br/>redis / embedding / qdrant / reranker<br/>正常10s探测，降级3s探测<br/>reranker replicas独立探测"]
    B --> C{"开启 RAG-only?"}
    C -- "是" --> F["强制 RAG route"]
    C -- "否" --> D["意图路由"]
    D --> E{"是否需要本地文档?"}
    E -- "否" --> Z["直接调用 LLM 对话"]
    E -- "是" --> F
    F --> GateIn["进入文档版本召回闸门"]
    GateIn --> G["Ray Embedding Actor<br/>动态合批：最大16 / 等待10 ms"]
    GateIn --> H["BM25S 关键词召回<br/>优先 S3 active manifest + Markdown<br/>Qdrant payload scroll 兜底"]
    G --> I["Qdrant 通过 tech_docs alias 做向量召回"]
    P -. "embedding 或 qdrant degraded 时跳过向量召回" .-> G
    P -. "qdrant degraded 时跳过向量召回" .-> I
    H --> J["RRF 融合"]
    I --> J
    J --> GateOut["释放文档版本召回闸门"]
    Commit["增量索引提交<br/>只排队新召回并排空进行中的召回"] -. "alias + manifest + BM25指针交换" .-> GateIn
    GateOut --> R{"reranker 组件健康?<br/>至少一个 replica ready"}
    P -. "最近一次健康快照" .-> R
    R -- "是" --> K["通过 Ray replicas 精排<br/>轮转 kba_reranker_1/_2"]
    K --> S{"选中的 replica 调用成功?"}
    S -- "成功" --> L["保留 Final K chunks"]
    S -- "失败；还有存活 replica" --> K
    S -- "全部失败" --> U["RRF-only Final K<br/>reranker_degraded=true"]
    R -- "否" --> U
    U --> L
    L --> M["拼接引用、会话记忆和 prompt"]
    M --> N["LLM 生成回答"]
    N --> O["Redis原子写入<br/>Hot messages + Pending payload + Stream引用"]
    O -. "Redis写入失败" .-> R503
    O --> V["不等待PG归档，直接返回"]
    O --> W["session-archiver批量消费"]
    W --> X["通过PgBouncer提交PG"]
    X --> Y["HDEL Pending + XACK/XDEL Stream"]
```

PostgreSQL不保存文档chunk的canonical副本。PG负责用户、运行设置和完整的Session/
消息归档，包括历史回答的retrieved contexts快照；Redis是活跃Session读写路径。
带滑动TTL的Session cache key可以被`volatile-lru`淘汰；Pending Hash和Archive
Stream没有TTL，不会被淘汰。Archiver必须先PG commit，之后才能确认并清理Redis
归档数据。完整exchange只在Hot Messages List和Pending Hash中各保存一份；Stream
只保存`event_id`和`session_id`引用，Archiver会批量读取对应Pending payload，不再在
Stream entry中复制完整内容。
只有Redis仍可访问时才允许cache miss冷加载，这样返回前可以合并PG历史与
尚未归档的Pending exchange。Redis不可用或Archive backlog达到上限时，Session读写
统一返回带`Retry-After: 3`的HTTP `503`；不会返回可能过期的PG-only历史，也不会同步
fallback写PG。账号鉴权仍可访问PG，因为账号数据本来就以PG为canonical source，
不属于活跃会话状态。第一次尚未被后台探测发现的故障最多消耗一次Redis socket
timeout；之后进程内熔断会立即返回503。恢复探测只由后台health supervisor执行，
不会让用户请求周期性等待Redis恢复检查。文档chunk text仍保存在Qdrant payload，
并且可以从S3原文重新构建。
`appendfsync everysec`明确接受约1秒AOF持久化窗口。Redis容器在32GB `maxmemory`
外预留到48GB；Linux宿主机应设置`vm.overcommit_memory=1`，避免AOF rewrite fork被
内核拒绝。

检索降级不会 silent failure。后台会独立探测 `embedding`、`qdrant` 和
`reranker` 三个组件：正常状态每 10 秒探测一次，连续 2 次失败后才进入 degraded；
degraded 后每 3 秒探测一次，连续 2 次恢复成功后才退出 degraded。探测是轻量的：
Qdrant 只查 collection/alias 元信息，embedding/reranker 只检查 Ray actor readiness，
不执行真实用户 query、真实代码 embedding 或真实 rerank。reranker 组件只要至少一个
replica ready 就仍视为可用。reranker replica 之间的健康探测相互隔离，并接受最先
成功的 readiness 结果，所以单个被 `docker compose pause` 冻结或 hang 住的 reranker
worker 不会拖住仍健康的另一个 replica。请求路径只读取最近一次健康状态，不在每次
query 上额外探测。

进入 degraded 后，各组件独立 fallback：BM25S 关键词召回失败时继续使用
Qdrant/vector-only；embedding 或 Qdrant degraded 时跳过向量召回并使用 BM25-only；
单个 reranker replica 请求失败时会先尝试另一个 replica；全部 reranker replica
degraded 或调用失败时才使用 RRF 粗排结果并按 `Final K` 截断。
故障注入期望也按这个语义：只 pause/stop `ray-worker-reranker-1` 或只 pause/stop
`ray-worker-reranker-2` 时，`reranker_degraded=false`，结果仍应包含 `rerank_score`；
两个 reranker worker 都 pause/stop 时，才应进入 `reranker_degraded=true`、
`reranker_ms=0`，并返回 `rerank_score=null` 的 RRF-only 结果。`/rag` 响应和
存储的 assistant 消息会包含 `retrieval_degraded`、`embedding_degraded`、
`qdrant_degraded`、`reranker_degraded` 和 `degradation_reason`；服务端日志也会写出
这些布尔值，前端会在回答和引用区显示降级提示。`/health/details` 的
`component_health` 会展示每个组件的状态、连续失败/成功次数、最近错误和当前探测
间隔。

容器级自愈由 Docker 层处理，但 Ray cluster 不做单容器 autoheal：`qdrant` 和
`main` 带 healthcheck 与 `autoheal=true` label，`autoheal` 容器通过 Docker socket
重启 unhealthy 容器；`ray-head` 和 Ray worker 保留 `restart: unless-stopped`，仅在
Ray 进程退出时由 Docker 重启。Ray actor 级异常由 Ray 的 actor restart 和应用健康
探测/fallback 处理，避免单独重启 head 后留下旧 worker/actor 连接。

这里有意只做进程级恢复，不实现索引存储灾备状态机。Qdrant普通重启会继续挂载
`qdrant_storage`，无需重新chunk或embedding；collection或volume损坏时保持
BM25-only，由运维显式从S3执行CPU offline rebuild。该低概率路径保持手动，避免为
长期只读的数据卷引入过高复杂度。BM25S是`main`内存索引：单次异常会降级到
Qdrant-only，但目前没有独立BM25S健康探测和后台自动重建；重启`main`或成功提交
文档更新会重建或替换它。

默认 Compose 配置会把 GPU 暴露给 `ray-worker-embedding-1`、
`ray-worker-reranker-1` 和 `ray-worker-reranker-2`；`main` 容器只是 Ray client，
不再申请 GPU：

```bash
docker compose up --build
```

如果希望一条命令先尝试 GPU、Docker 在容器创建前拒绝 GPU 时再自动重试 CPU，
使用这个 wrapper：

```bash
scripts/compose_up.sh up --build
```

如果你想从一开始就强制 CPU，使用 CPU override：

```bash
CUDA=FALSE docker compose -f compose.yaml -f compose.cpu.yaml up --build
```

如果 Docker 已经暴露 GPU，但 PyTorch 仍无法使用，或显卡架构太老不兼容当前
CUDA/PyTorch 构建，应用启动后会回退到 CPU。

## Code Search

Code Search 和 Markdown RAG 是两条独立索引链路。`data/docs/` 继续走原有
Markdown chunk、BM25S、Qdrant 文档向量和 RRF 融合；`data/code/` 下的源码仓库
走代码索引链路。自然语言 README、安装文档和教程仍建议放进 `data/docs/` 或通过
`DOCS_DIR` 指向对应文档目录。

默认代码配置：

```bash
DOCS_DIR=/app/data/docs
CODE_ROOT_DIR=/app/data/code
# 可选：CODE_SOURCE_DIR=/app/data/code/specific-repo
CODE_FILES_COLLECTION=code_files
CODE_FUNCTIONS_COLLECTION=code_functions
CODE_EMBEDDING_MODEL=microsoft/codebert-base
CODE_EMBEDDING_PRELOAD=0
CODE_EMBEDDING_PRELOAD_RETRIES=3
CODE_EMBEDDING_PRELOAD_RETRY_SECONDS=20
CODE_SEARCH_FILE_TOP_K=20
CODE_SEARCH_FUNCTION_TOP_K=50
CODE_SEARCH_FINAL_TOP_K=10
CODE_CALL_GRAPH_DEPTH=3
RAY_ENABLED=1
RAY_ADDRESS=ray://ray-head:10001
RAY_LOCAL_FALLBACK=0
GRPC_ENABLE_FORK_SUPPORT=0
RAY_CODE_EMBEDDING_ACTOR_NUM_GPUS=0
RAY_CODE_EMBEDDING_ACTOR_NAME=kba_code_embedding
```

推荐目录结构：

```text
data/docs/          # 原有 Markdown RAG 语料
data/code/xgboost/  # 前端 Code 模式可选择的源码仓库
data/code/lightgbm/ # 另一个源码仓库
```

普通重启不会扫描本地 Markdown。local 模式下，文档索引用
`python scripts/ingest_docs.py` 手动触发。S3 模式下，`docs-indexer`按Docker image
build id和索引配置指纹初始化文档：`docker compose up --build`生成镜像marker，
marker或配置指纹变化时才把`DOCS_DIR`同步到S3并构建版本化索引；之后
`docker compose restart`会跳过本地同步和diff。如果源码没有变化但需要强制
初始化，可在
`docker compose up --build` 前把 `DOCS_INIT_BUILD_ID` 改成新值。代码索引按需触发：
打开浏览器UI，切换到`Code`，选择`data/code`下的仓库并点击`Index`。也可以登录后
调用API：

```bash
curl -sS http://localhost:8080/code/index \
  -H "authorization: Bearer ${TOKEN}" \
  -H 'content-type: application/json' \
  -d '{"repository_ids":["xgboost"],"rebuild":true}'
```

代码索引会写入：

- `code_files` 表和 `code_files` Qdrant collection：路径、语言、完整源码和
  CodeBERT 文件向量。
- `code_functions` 表和 `code_functions` Qdrant collection：函数/类名、签名、
  body、docstring、行号和 CodeBERT 函数向量。
- `code_call_edges` 表：基于 AST 提取的 caller 到 callee 调用边。

当前 Code Search 是 CodeBERT-first，不是完整的 BM25S+dense hybrid search。
正常请求会用 Ray 常驻的 CodeBERT Actor embed query，先搜 `code_files`，再搜
top files 内的 `code_functions`，同时额外做一次 repo-wide `code_functions`
向量召回。lexical 匹配只用于补充函数候选；这些候选会重新用 Qdrant 中的
CodeBERT stored vector 和 query vector 计算相似度，再进入最终排序。最终分数是
CodeBERT vector score 加一个较小的 code-aware lexical boost。

如果 CodeBERT query embedding 失败，`/code/search` 会降级到 PostgreSQL 里的
代码 lexical 扫描，并在响应中返回 `retrieval_mode="text_only"`、
`code_embedding_degraded=true` 和 `degradation_reason`。这个 code lexical fallback
目前不是 BM25S；BM25S 只用于 Markdown RAG。

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

local 模式下，`data/docs/` 下的 Markdown 文件是 RAG 语料来源。S3 模式下，
配置的 S3 bucket/prefix 是语料真源，`QDRANT_COLLECTION` 会作为稳定 alias 指向
当前生效的版本化 Qdrant collection。当前语料递归存放在`data/docs/RAGBench/`，
包含RAGBench `cuad`测试集的102份Markdown合同文档。`RAGBench_Eval/corpus.jsonl`
保存原始记录，`RAGBench_Eval/queries.jsonl`保存500条标注query。文档召回评测应使用
`RAG-only`。当前`DOMAIN_RAG_PHRASES`、`DOMAIN_RAG_PATTERNS`和intent-router评测
样例仍用于演示上一版语料域；正式评估CUAD的自动意图路由前需要单独更新。检索链路
使用BM25关键词召回、Qdrant向量召回、RRF融合和可选reranker。

Markdown chunker会复用`jinaai/jina-embeddings-v5-text-small`附带的Qwen3 tokenizer，
但`main`只加载tokenizer，不加载embedding权重。同一标题section内按完整段落装箱，
body目标1600 token且硬上限同为1600 token；遇到标题变化立即结束当前chunk，因此
很短的section保持短chunk。超长正文优先按句子切，单句仍过长时继续按子句、词和
最终token边界切，不按任意字符位置截断。每个text chunk可从前一个和后一个body
分别加入目标/最大100 token的双向overlap，且禁止跨标题；分隔符计入各自overlap
预算，因此组装后的派生硬上限是`1600 + 100 + 100 = 1800` token。text、code、table
继续分开处理；只有显式围栏代码块才按code处理，视觉缩进的长正文仍作为text装箱。
代码围栏保持完整；标题存入metadata并加入embedding input，不重复写入chunk正文。
YAML frontmatter会先被解析并从正文移除，其中`title`作为文档级
标题metadata，正文原始行号保持不变。Qdrant payload会保存body、前后overlap和
组装后的token计数。S3 manifest会记录chunk算法版本、tokenizer和全部token预算，
因此之后手动
更新文档时会构建新索引版本，不会复用旧chunker生成的向量。

替换知识库步骤：

1. 替换或编辑 `data/docs/` 下的 Markdown 文件。
2. local 模式下重建 Qdrant collection：

   ```bash
   python scripts/ingest_docs.py --recreate
   ```

   Compose S3模式使用统一离线生命周期脚本，不要直接让在线actor执行ingest：

   ```bash
   ./scripts/update_docs.sh
   ```

   对外暴露`main`前必须把`DOCS_COMMIT_TOKEN`改成非默认值；内部prepare/commit端点
   会校验该共享token。CPU更新容量和超时由`DOCS_CPU_INGEST_THREADS`、
   `DOCS_CPU_INGEST_ACTOR_NUM_CPUS`、`EMBEDDING_CPU_INGEST_BATCH_SIZE`和
   `DOCS_CPU_INGEST_TIMEOUT_SECONDS`控制。

   脚本只启动`docs-update` profile中的CPU ingest worker和一次性updater，不会停止
   或重建在线embedding与两个reranker。临时Ray node声明0 GPU，并使用独立的
   `ray_worker_embedding_ingest`资源和`kba_embedding_ingest_cpu` actor名，因此
   离线actor不会被调度到在线GPU worker。更新结束前会用`no_restart`永久删除该
   detached actor，再停止临时Ray node，避免退出后被Ray重新调度。

   S3同步、manifest diff、chunk、CPU embedding、Qdrant候选写入和完整BM25S候选
   重建期间，上一版索引持续服务。`main`在内存中保留`active`和`candidate`两个
   BM25S引用；后台从完整候选manifest构建`candidate`，线上查询始终读取`active`。
   代价是准备阶段BM25S内存短暂翻倍，但不会暴露半成品。若diff为空，不加载
   embedding模型，也不切换任何索引指针。

3. 如需让意图路由识别新语料主题，更新 `app/intent_router.py` 中的
   `DOMAIN_RAG_PHRASES`、`DOMAIN_RAG_PATTERNS`、LLM fallback 的语料描述，以及
   `data/eval/intent_router_cases.jsonl`。
4. 如需改 `RAG_ANCHORS` / `DIRECT_ANCHORS`，保持每条 anchor 都是单意图 query。
5. 运行 `python scripts/intent_router_ab.py --fake-embedder`；有模型权重时再跑真实
   encoder 对比。
6. 通过浏览器或 `/rag` API 提问。

S3 增量更新 pipeline：

```mermaid
flowchart TD
    A["编辑 data/docs 下的 Markdown"] --> B["scripts/update_docs.sh"]
    B --> C["启动docs-updater + 临时CPU Ray worker<br/>在线GPU actor继续服务"]
    C --> D["把新增/修改/删除同步到版本化MinIO"]
    D --> E["写processing候选manifest<br/>固定精确S3 VersionId"]
    E --> F["读取上一版manifest + 完整候选manifest"]
    F --> G["按source_doc_id、VersionId、ETag、size、content hash和索引配置diff"]
    G --> H["新增 / 修改文档"]
    G --> I["未变化文档"]
    G --> J["已删除文档"]
    E --> BM["main后台构建完整BM25S candidate<br/>active BM25S继续服务"]
    H --> K["Qwen3切新增chunk<br/>CPU actor只向量化新chunk"]
    I --> L["复制原Qdrant points + vectors"]
    J --> M["不复制已删除文档的任何chunk"]
    K --> N["候选Qdrant collection"]
    L --> N
    M --> N
    N --> Q{"Qdrant数量校验通过且<br/>BM25 candidate ready?"}
    BM --> Q
    Q -- "否" --> Fail["丢弃Qdrant/BM25候选<br/>active alias和BM25不变"]
    Q -- "是" --> Gate["关闭召回闸门<br/>新召回排队；进行中召回排空"]
    Gate --> Alias["Qdrant alias切到候选collection"]
    Alias --> Manifest["提交并校验S3 active manifest"]
    Manifest --> Swap["交换BM25 active/candidate引用"]
    Swap --> Release["打开闸门<br/>排队召回统一读取新版本"]
    Release --> Cleanup["no_restart删除CPU actor<br/>停止临时Ray node"]
    Cleanup --> Keep["稳定保留：S3最新5版<br/>Qdrant active + rollback"]
    Gate -. "不等待" .-> Outside["Reranker和LLM在闸门外继续执行"]
    D -. "pipeline错误" .-> Fail
```

只有最后的短提交窗口会关闭召回闸门。排空和排队范围仅包括query embedding、
BM25S、Qdrant和RRF；已经离开召回阶段的reranker与LLM不读取可变索引，因此继续
执行且不计入drain。提交窗口依次切换Qdrant alias、提交active S3 manifest并交换
BM25S引用，然后统一释放排队请求。

local ingest 会递归扫描子目录，并用当前 Markdown 文件替换同一 source 的旧 chunk，
避免修改或缩短文件后留下陈旧 chunk。删除文档、整体替换语料或修改 embedding
模型维度时，local 模式请使用 `--recreate`。

本机 demo 默认用 MinIO 作为 S3 兼容对象存储；生产或远端部署只需要把
`DOCS_S3_ENDPOINT_URL`、bucket 和凭据换成对应服务。S3 ingest 每次都会扫描当前
S3 文件列表生成 manifest，而不是只扫描“变动/新增”的文件，所以能检测已经消失的
对象。上一版 manifest 存在
`DOCS_S3_MANIFEST_PREFIX` 下；每个 chunk payload 都带 `source_doc_id` 和
`version_id`。构建新版本时，会创建新的物理 Qdrant collection：未变更文档从旧
collection 复制已有向量，新增/修改文档重新 chunk + embed，删除文档不会被复制进
新 collection。Qdrant和BM25S两个候选都通过校验后，协调提交才会在同一个召回
闸门内切换`QDRANT_COLLECTION` alias和BM25S active引用；旧collection和S3对象
版本继续保留。提交前失败只丢弃候选；提交过程失败会在重新打开闸门前尝试恢复
上一版alias和manifest。

版本保留策略默认开启。稳定状态下，S3 每个 Markdown key 保留最新 5 个对象版本
（包含最新可用版本）；Qdrant 保留最新 2 个文档索引 collection（包含当前 alias
指向的可用版本）。ingest 处理中会临时放宽到 S3 6 个版本、Qdrant 3 个 collection，
给候选新版本留空间。若构建或切换后失败，系统会把 alias 回滚到上一版可用
collection，并删除本次失败的新 collection。

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
    \"bm25_top_k\": 100,
    \"recall_top_k\": 100,
    \"rrf_top_k\": 64,
    \"top_k\": 5
  }"
```

`/rag` 的历史由服务端根据 `session_id` 从 PostgreSQL 管理，客户端不需要传完整
history。

返回的 `contexts` 会先写入Redis热消息，再由Archiver写入PostgreSQL assistant消息，
是验证检索链路最可靠
的位置。`retrieval_source=hybrid` 表示同一个 chunk 同时被 BM25 和向量召回命中；
纯向量或纯 BM25 命中只会有对应的 `vector_score` 或 `bm25_score`。`rrf_score`
表示已经经过 RRF 融合，`rerank_score` 表示 Jina cross-encoder reranker 已运行。
Docker logs 默认未必显示这些 INFO 级业务日志，除非运行时日志级别打开应用 INFO。

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
docker compose exec -T main python scripts/test_redis_session.py
python scripts/test_prompt_budget.py
python scripts/test_chunking.py
# 在main镜像/共享HF cache内按真实Qwen3 token验证语料。
python scripts/test_chunking.py --real-tokenizer
python scripts/test_intent_router.py
python scripts/intent_router_ab.py --fake-embedder
python scripts/test_reranker.py
python scripts/test_embedding_batcher.py
python scripts/test_model_warmup.py
python scripts/test_vector_store.py
python scripts/test_document_commit.py
```

### Session链路压测

Compose服务已经启动时，用固定10 QPS压测Session存储链路：

```bash
docker compose run --rm --no-deps \
  -v /tmp:/bench-output \
  --entrypoint python main \
  scripts/bench_session_pipeline.py \
  --rate 10 --duration 60 --warmup-seconds 10 \
  --session-count 32 --workers 32 \
  --output /bench-output/kba_session_benchmark_10qps.json
```

脚本直接调用`RedisSessionStore`，只覆盖
`Redis -> Stream -> session-archiver -> PostgreSQL`，明确不包含HTTP、LLM生成、
意图路由和检索。报告包含实际完成QPS、服务处理/排队/定时端到端的
P50/P95/P99、Stream峰值/平均/最终backlog、归档吞吐和排空时间，以及精确的
消息数与event id一致性校验。脚本结束后会删除本次压测Session。

2026-07-10本机固定10 QPS验证结果：正式阶段600/600写入成功、错误数0，实际完成
QPS为10.016；服务处理延迟P50 2.875 ms / P95 3.440 ms / P99 3.941 ms；
定时端到端延迟P50 3.038 ms / P95 3.629 ms / P99 4.140 ms；Stream backlog
峰值1、最终0；归档吞吐10.000 events/s。该结果是固定负载下的正确性基线，
不是系统最大容量。

闭环模式用于测量短时接收峰值：

```bash
docker compose run --rm --no-deps \
  -v /tmp:/bench-output \
  --entrypoint python main \
  scripts/bench_session_pipeline.py \
  --mode closed-loop --concurrency 1 --duration 5 \
  --warmup-seconds 0 --session-count 128 --workers 1 \
  --max-requests 5000 --max-backlog 5000 \
  --output /bench-output/kba_session_peak.json
```

同一套本机部署保持Redis AOF开启，得到两个必须分开解释的容量指标。5秒受控突发
完成5000/5000 event，接收峰值为1016.4 QPS、服务P99为2.44 ms，但backlog
以325.6 events/s增长，停止写入后需要2.49秒排空，因此它只是突发峰值，不是
可持续吞吐。固定450 QPS运行60秒时，27000/27000 event成功，实际完成449.18
QPS，归档吞吐448.99 events/s；Stream结束backlog为0、峰值79、拟合增长斜率
0.034 events/s，PG中的54000条消息全部存在且event id唯一。服务延迟P50为
2.40 ms、P95为143.28 ms、P99为149.72 ms，定时端到端P99为391.19 ms；
尾延迟包含自动AOF rewrite成本。600 QPS offered-load测试实际饱和在约451 QPS并
产生客户端排队，因此不能写成可持续容量。

### 检索链路压测

检索压测直接调用生产检索协调器：

```text
BM25S ───────────────┐
                    ├─ RRF ─ 两个Ray Reranker副本
Ray Embedding ─ Qdrant ┘
```

测试明确排除HTTP、Session持久化、意图路由和LLM生成。闭环饱和测试与固定速率
持续测试命令如下：

```bash
docker compose run --rm --no-deps \
  -v /tmp:/bench-output \
  --entrypoint python main \
  scripts/bench_retrieval_pipeline.py \
  --mode closed-loop --concurrency 16 --workers 16 \
  --duration 30 --warmup-requests 4 --max-requests 600 \
  --output /bench-output/kba_retrieval_dynamic16_peak_c16.json

docker compose run --rm --no-deps \
  -v /tmp:/bench-output \
  --entrypoint python main \
  scripts/bench_retrieval_pipeline.py \
  --mode fixed-rate --rate 9 --workers 16 \
  --duration 60 --warmup-requests 4 --max-requests 700 \
  --output /bench-output/kba_retrieval_dynamic16_sustained_9qps_60s.json
```

2026-07-10本机RTX 5090的合批前基线使用1个Jina v5 text Embedding Actor、2个
Jina Reranker v3 Actor，BM25/向量Top K均为100、RRF K为64、Final K为5。
并发8闭环完成213/213请求，无degradation，峰值10.27 QPS；检索延迟P50为
769.0 ms、P95为806.6 ms、P99为928.7 ms。固定9 QPS运行60秒完成540/540，
实际8.962 QPS；检索P50/P99为250.2/418.2 ms，客户端排队增长仅0.024 ms/s。
12 QPS过载对照饱和在9.66 QPS，排队以211.5 ms/s增长，检索P99为1.82秒。

启用此前“最大8、等待5 ms”的动态合批后，并发8闭环完成212/212，吞吐10.22
QPS。Embedding P50从592.5 ms降至175.4 ms，但成批释放请求把排队转移到两个
reranker：reranker P50从161.1 ms升至546.4 ms，检索P99升至1.229秒。Actor
对214个正式query embedding组成161个batch，平均batch size为1.33，观测到的
最大batch为7。固定9 QPS下，请求间隔较大，平均batch size仅1.002；实际吞吐
8.982 QPS，检索P50/P99为265.4/426.8 ms，排队斜率保持在0.013 ms/s。

12 QPS过载对比体现了保护效果：完成吞吐从9.66升至10.31 QPS，客户端排队斜率
从211.5降至126.0 ms/s，端到端P99从7.32秒降至5.70秒，Embedding P50从
1.47秒降至181.5 ms。但该负载仍不可持续，reranker P99升至2.59秒。因此动态
合批能在竞争时提高embedding效率，却不能单独提高平衡后的端到端容量；下一步约束
是reranker准入控制或增加下游容量。BM25S和Qdrant仍只占总延迟的小部分。由于
每次Rerank最多传输64个候选chunk，累计任务payload超过10 MB后Ray Client会提示
细粒度任务传输开销；压缩Actor参数或使用object reference仍属于后续优化。

随后对当前“最大16、等待10 ms”配置使用相同检索参数重新完整压测。四组测试均为
0 error、0 degradation：

| 负载 | 实际完成QPS | 端到端P50 / P95 / P99 | 排队斜率 | 平均Embedding batch | 定性 |
| --- | ---: | ---: | ---: | ---: | --- |
| 闭环并发16 | 10.301 | 1.540 / 2.562 / 2.637 s | 不适用 | 1.498 | 短时峰值 |
| 固定9 QPS，60秒 | 8.968 | 325.9 / 424.8 / 535.3 ms | 0.048 ms/s | 1.017 | 保守可持续负载 |
| 固定10 QPS，60秒 | 9.963 | 398.8 / 566.1 / 639.3 ms | 0.021 ms/s | 1.210 | 实测容量边界 |
| 固定12 QPS，30秒 | 10.390 | 2.231 / 5.481 / 5.780 s | 83.419 ms/s | 1.504 | 过载，不可持续 |

闭环测试观测到的最大Embedding batch为15。固定9 QPS时均匀请求间隔约111 ms，
所以10 ms窗口几乎无法合批，这是预期行为。固定10 QPS在60秒观察期内没有客户端
backlog增长，但服务延迟仍有2.044 ms/s的正斜率，因此它是容量边界，不是长时间
稳定性保证。固定12 QPS时系统饱和在约10.39 QPS，reranker P99达到3.67秒，说明
当前过载约束是两个reranker replica，而不是embedding显存。可复现JSON报告保存在
压测宿主机的`/tmp/kba_retrieval_dynamic16_*.json`。

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
DOCKER_NO_PROXY=postgres,pgbouncer,redis,session-archiver,qdrant,minio,vllm,main,docs-indexer,docs-updater,ray-head,ray-worker-embedding-bootstrap,ray-worker-embedding-ingest-cpu,ray-worker-embedding-1,ray-worker-reranker-1,ray-worker-reranker-2,localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
GRPC_ENABLE_FORK_SUPPORT=0
```

不要在容器内把代理地址写成 `127.0.0.1`，它指向容器自身。Compose 已经给 `main`、
`ray-head` 和三个 Ray worker 配置 `host.docker.internal`，所以所有模型服务容器都能
通过同一个宿主机代理下载模型。

离线模型可挂载到 `./models:/models:ro`：

```text
models/
  jina-embeddings-v5-text-small/
  qwen2.5-7b-instruct/
```

然后配置：

```bash
EMBEDDING_MODEL=/models/jina-embeddings-v5-text-small
EMBEDDING_TRUST_REMOTE_CODE=1
VLLM_MODEL=/models/qwen2.5-7b-instruct
LLM_MODEL=qwen2.5-7b-instruct
```

## 仓库内容和安全提示

应提交源码、脚本、Docker 配置、`.env.example`、依赖文件和 Markdown 语料。不要
提交 `.env`、API key、Hugging Face token、模型权重、PostgreSQL 数据、Qdrant
存储、cache、日志或虚拟环境。
