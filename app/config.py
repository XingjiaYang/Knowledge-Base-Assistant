from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_PROVIDER_OPENAI = "openai_compatible"
LLM_PROVIDER_ANTHROPIC = "anthropic"
LLM_PROVIDER_CHOICES = {LLM_PROVIDER_OPENAI, LLM_PROVIDER_ANTHROPIC}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name)
    if value is None or not value.strip():
        return Path(default)
    return Path(value)


def _env_has_value(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(value.strip())


def normalize_llm_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"openai", "openai_compatible", "openai_compat"}:
        return LLM_PROVIDER_OPENAI
    if normalized in {"anthropic", "claude"}:
        return LLM_PROVIDER_ANTHROPIC
    raise ValueError("LLM provider must be openai_compatible or anthropic.")


@dataclass(frozen=True)
class LLMRuntimeSettings:
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_context_max_tokens: int
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    llm_timeout_seconds: float
    llm_retry_attempts: int
    llm_retry_backoff_seconds: float
    llm_retry_backoff_max_seconds: float
    llm_health_check_enabled: bool
    llm_health_path: str
    llm_anthropic_version: str

    @classmethod
    def from_settings(
        cls,
        config: "Settings",
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        context_max_tokens: int | str | None = None,
    ) -> "LLMRuntimeSettings":
        context_tokens = (
            config.llm_context_max_tokens
            if context_max_tokens is None
            else int(context_max_tokens)
        )
        return cls(
            llm_provider=normalize_llm_provider(provider or config.llm_provider),
            llm_base_url=(base_url or config.llm_base_url).strip(),
            llm_api_key=config.llm_api_key if api_key is None else api_key,
            llm_model=(model or config.llm_model).strip(),
            llm_context_max_tokens=context_tokens,
            llm_temperature=config.llm_temperature,
            llm_top_p=config.llm_top_p,
            llm_max_tokens=config.llm_max_tokens,
            llm_timeout_seconds=config.llm_timeout_seconds,
            llm_retry_attempts=config.llm_retry_attempts,
            llm_retry_backoff_seconds=config.llm_retry_backoff_seconds,
            llm_retry_backoff_max_seconds=config.llm_retry_backoff_max_seconds,
            llm_health_check_enabled=config.llm_health_check_enabled,
            llm_health_path=config.llm_health_path,
            llm_anthropic_version=config.llm_anthropic_version,
        )

    def validate(self) -> "LLMRuntimeSettings":
        normalize_llm_provider(self.llm_provider)
        if not self.llm_base_url.strip():
            raise ValueError("LLM base URL must not be empty.")
        if not self.llm_model.strip():
            raise ValueError("LLM model must not be empty.")
        if self.llm_context_max_tokens <= 0:
            raise ValueError("LLM context max tokens must be greater than 0.")
        return self


@dataclass(frozen=True)
class Settings:
    debug: bool = _env_bool("DEBUG", False)
    cuda_enabled: bool = _env_bool("CUDA", True)

    docs_dir: Path = Path(os.getenv("DOCS_DIR", str(PROJECT_ROOT / "data" / "docs")))
    docs_source: str = os.getenv("DOCS_SOURCE", "local")
    docs_s3_bucket: str = os.getenv("DOCS_S3_BUCKET", "")
    docs_s3_prefix: str = os.getenv("DOCS_S3_PREFIX", "docs")
    docs_s3_endpoint_url: str = os.getenv("DOCS_S3_ENDPOINT_URL", "")
    docs_s3_region: str = os.getenv("DOCS_S3_REGION", "us-east-1")
    docs_s3_access_key_id: str = os.getenv("DOCS_S3_ACCESS_KEY_ID", "")
    docs_s3_secret_access_key: str = os.getenv("DOCS_S3_SECRET_ACCESS_KEY", "")
    docs_s3_session_token: str = os.getenv("DOCS_S3_SESSION_TOKEN", "")
    docs_s3_force_path_style: bool = _env_bool("DOCS_S3_FORCE_PATH_STYLE", False)
    docs_s3_require_versioning: bool = _env_bool("DOCS_S3_REQUIRE_VERSIONING", True)
    docs_s3_retain_versions: int = _env_int("DOCS_S3_RETAIN_VERSIONS", 5)
    docs_s3_processing_retain_versions: int = _env_int(
        "DOCS_S3_PROCESSING_RETAIN_VERSIONS",
        6,
    )
    docs_s3_manifest_prefix: str = os.getenv(
        "DOCS_S3_MANIFEST_PREFIX",
        "_kba/manifests/docs",
    )
    qdrant_retain_versions: int = _env_int("QDRANT_RETAIN_VERSIONS", 2)
    qdrant_processing_retain_versions: int = _env_int(
        "QDRANT_PROCESSING_RETAIN_VERSIONS",
        3,
    )
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "tech_docs")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "jinaai/jina-embeddings-v5-text-small",
    )
    embedding_trust_remote_code: bool = _env_bool("EMBEDDING_TRUST_REMOTE_CODE", True)
    embedding_query_task: str = os.getenv("EMBEDDING_QUERY_TASK", "retrieval")
    embedding_passage_task: str = os.getenv(
        "EMBEDDING_PASSAGE_TASK",
        "retrieval",
    )
    embedding_classification_task: str = os.getenv(
        "EMBEDDING_CLASSIFICATION_TASK",
        "classification",
    )
    embedding_query_prompt_name: str = os.getenv("EMBEDDING_QUERY_PROMPT_NAME", "query")
    embedding_passage_prompt_name: str = os.getenv(
        "EMBEDDING_PASSAGE_PROMPT_NAME",
        "document",
    )
    embedding_classification_prompt_name: str = os.getenv(
        "EMBEDDING_CLASSIFICATION_PROMPT_NAME",
        "",
    )
    chunk_size: int = _env_int("CHUNK_SIZE", 2000)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 300)
    bm25_top_k: int = _env_int("BM25_TOP_K", 100)
    recall_top_k: int = _env_int("RECALL_TOP_K", 100)
    rrf_top_k: int = _env_int("RRF_TOP_K", 100)
    retrieve_top_k: int = _env_int("RETRIEVE_TOP_K", 5)
    retrieve_score_threshold: float = _env_float("RETRIEVE_SCORE_THRESHOLD", 0.0)
    api_top_k_max: int = _env_int("API_TOP_K_MAX", 20)
    api_recall_top_k_max: int = _env_int("API_RECALL_TOP_K_MAX", 1000)
    code_root_dir: Path = _env_path(
        "CODE_ROOT_DIR",
        str(PROJECT_ROOT / "data" / "code"),
    )
    code_source_dir_explicit: bool = _env_has_value("CODE_SOURCE_DIR")
    code_source_dir: Path = _env_path(
        "CODE_SOURCE_DIR",
        os.getenv("CODE_ROOT_DIR", str(PROJECT_ROOT / "data" / "code")),
    )
    code_files_collection: str = os.getenv("CODE_FILES_COLLECTION", "code_files")
    code_functions_collection: str = os.getenv(
        "CODE_FUNCTIONS_COLLECTION",
        "code_functions",
    )
    code_embedding_model: str = os.getenv(
        "CODE_EMBEDDING_MODEL",
        "microsoft/codebert-base",
    )
    code_embedding_preload: bool = _env_bool("CODE_EMBEDDING_PRELOAD", True)
    code_embedding_preload_retries: int = _env_int(
        "CODE_EMBEDDING_PRELOAD_RETRIES",
        3,
    )
    code_embedding_preload_retry_seconds: float = _env_float(
        "CODE_EMBEDDING_PRELOAD_RETRY_SECONDS",
        20.0,
    )
    code_embedding_batch_size: int = _env_int("CODE_EMBEDDING_BATCH_SIZE", 8)
    code_embedding_max_tokens: int = _env_int("CODE_EMBEDDING_MAX_TOKENS", 512)
    code_file_embedding_max_chars: int = _env_int(
        "CODE_FILE_EMBEDDING_MAX_CHARS",
        20000,
    )
    code_function_embedding_max_chars: int = _env_int(
        "CODE_FUNCTION_EMBEDDING_MAX_CHARS",
        12000,
    )
    code_payload_snippet_chars: int = _env_int("CODE_PAYLOAD_SNIPPET_CHARS", 2400)
    code_search_file_top_k: int = _env_int("CODE_SEARCH_FILE_TOP_K", 20)
    code_search_function_top_k: int = _env_int("CODE_SEARCH_FUNCTION_TOP_K", 50)
    code_search_final_top_k: int = _env_int("CODE_SEARCH_FINAL_TOP_K", 10)
    code_call_graph_depth: int = _env_int("CODE_CALL_GRAPH_DEPTH", 3)
    reranker_enabled: bool = _env_bool("RERANKER_ENABLED", True)
    reranker_model: str = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v3")
    reranker_preload: bool = _env_bool("RERANKER_PRELOAD", True)
    reranker_trust_remote_code: bool = _env_bool(
        "RERANKER_TRUST_REMOTE_CODE",
        True,
    )
    reranker_dtype: str = os.getenv("RERANKER_DTYPE", "auto")
    reranker_max_documents_per_call: int = _env_int(
        "RERANKER_MAX_DOCUMENTS_PER_CALL",
        64,
    )
    ray_enabled: bool = _env_bool("RAY_ENABLED", True)
    ray_address: str = os.getenv("RAY_ADDRESS", "")
    ray_local_fallback: bool = _env_bool(
        "RAY_LOCAL_FALLBACK",
        not bool(os.getenv("RAY_ADDRESS", "").strip()),
    )
    ray_namespace: str = os.getenv("RAY_NAMESPACE", "kba")
    ray_actor_num_cpus: float = _env_float("RAY_ACTOR_NUM_CPUS", 1.0)
    ray_embedding_actor_num_gpus: float = _env_float(
        "RAY_EMBEDDING_ACTOR_NUM_GPUS",
        0.5 if _env_bool("CUDA", True) else 0.0,
    )
    ray_reranker_actor_num_gpus: float = _env_float(
        "RAY_RERANKER_ACTOR_NUM_GPUS",
        0.5 if _env_bool("CUDA", True) else 0.0,
    )
    ray_reranker_actor_replicas: int = _env_int(
        "RAY_RERANKER_ACTOR_REPLICAS",
        2,
    )
    ray_code_embedding_actor_num_gpus: float = _env_float(
        "RAY_CODE_EMBEDDING_ACTOR_NUM_GPUS",
        0.0,
    )
    ray_embedding_actor_name: str = os.getenv(
        "RAY_EMBEDDING_ACTOR_NAME",
        "kba_embedding",
    )
    ray_code_embedding_actor_name: str = os.getenv(
        "RAY_CODE_EMBEDDING_ACTOR_NAME",
        "kba_code_embedding",
    )
    ray_reranker_actor_name: str = os.getenv(
        "RAY_RERANKER_ACTOR_NAME",
        "kba_reranker",
    )
    ray_embedding_actor_resource: str = os.getenv(
        "RAY_EMBEDDING_ACTOR_RESOURCE",
        "",
    )
    ray_code_embedding_actor_resource: str = os.getenv(
        "RAY_CODE_EMBEDDING_ACTOR_RESOURCE",
        "",
    )
    ray_reranker_actor_resource: str = os.getenv(
        "RAY_RERANKER_ACTOR_RESOURCE",
        "",
    )
    ray_task_timeout_seconds: float = _env_float("RAY_TASK_TIMEOUT_SECONDS", 300.0)
    health_probe_interval_seconds: float = _env_float(
        "HEALTH_PROBE_INTERVAL_SECONDS",
        10.0,
    )
    health_probe_degraded_interval_seconds: float = _env_float(
        "HEALTH_PROBE_DEGRADED_INTERVAL_SECONDS",
        3.0,
    )
    health_probe_timeout_seconds: float = _env_float(
        "HEALTH_PROBE_TIMEOUT_SECONDS",
        2.0,
    )
    health_probe_failure_threshold: int = _env_int(
        "HEALTH_PROBE_FAILURE_THRESHOLD",
        2,
    )
    health_probe_recovery_threshold: int = _env_int(
        "HEALTH_PROBE_RECOVERY_THRESHOLD",
        2,
    )

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai_compatible")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_context_max_tokens: int = _env_int("LLM_CONTEXT_MAX_TOKENS", 256000)
    llm_context_safety_margin_tokens: int = _env_int(
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
        8192,
    )
    llm_context_prompt_overhead_tokens: int = _env_int(
        "LLM_CONTEXT_PROMPT_OVERHEAD_TOKENS",
        2048,
    )
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.2)
    llm_top_p: float = _env_float("LLM_TOP_P", 0.9)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 4096)
    llm_timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 300.0)
    llm_retry_attempts: int = _env_int("LLM_RETRY_ATTEMPTS", 3)
    llm_retry_backoff_seconds: float = _env_float("LLM_RETRY_BACKOFF_SECONDS", 1.0)
    llm_retry_backoff_max_seconds: float = _env_float(
        "LLM_RETRY_BACKOFF_MAX_SECONDS",
        10.0,
    )
    llm_health_check_enabled: bool = _env_bool("LLM_HEALTH_CHECK_ENABLED", False)
    llm_health_path: str = os.getenv("LLM_HEALTH_PATH", "")
    llm_anthropic_version: str = os.getenv(
        "LLM_ANTHROPIC_VERSION",
        "2023-06-01",
    )

    api_message_max_chars: int = _env_int("API_MESSAGE_MAX_CHARS", 16000)
    api_question_max_chars: int = _env_int("API_QUESTION_MAX_CHARS", 16000)
    api_summary_max_chars: int = _env_int("API_SUMMARY_MAX_CHARS", 12000)
    api_history_max_messages: int = _env_int("API_HISTORY_MAX_MESSAGES", 120)
    auth_enabled: bool = True
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://kba:kba_password@localhost:5432/kba",
    )
    database_connect_timeout_seconds: int = _env_int(
        "DATABASE_CONNECT_TIMEOUT_SECONDS",
        5,
    )
    auth_bootstrap_users: str = os.getenv("AUTH_BOOTSTRAP_USERS", "")
    auth_default_admin_enabled: bool = _env_bool("AUTH_DEFAULT_ADMIN_ENABLED", True)
    auth_default_admin_username: str = os.getenv(
        "AUTH_DEFAULT_ADMIN_USERNAME",
        "admin",
    )
    auth_default_admin_password: str = os.getenv(
        "AUTH_DEFAULT_ADMIN_PASSWORD",
        "123456",
    )
    auth_session_ttl_seconds: int = _env_int("AUTH_SESSION_TTL_SECONDS", 604800)
    session_list_limit: int = _env_int("SESSION_LIST_LIMIT", 50)
    session_title_max_chars: int = _env_int("SESSION_TITLE_MAX_CHARS", 80)

    history_recent_turns: int = _env_int("HISTORY_RECENT_TURNS", 16)
    history_compact_after_turns: int = _env_int("HISTORY_COMPACT_AFTER_TURNS", 40)
    history_max_messages: int = _env_int("HISTORY_MAX_MESSAGES", 0)
    message_max_chars: int = _env_int("MESSAGE_MAX_CHARS", 8000)
    conversation_summary_max_chars: int = _env_int(
        "CONVERSATION_SUMMARY_MAX_CHARS",
        256000,
    )
    summary_history_max_chars: int = _env_int("SUMMARY_HISTORY_MAX_CHARS", 200000)
    summary_max_tokens: int = _env_int("SUMMARY_MAX_TOKENS", 4096)
    search_query_max_chars: int = _env_int("SEARCH_QUERY_MAX_CHARS", 3000)

    intent_router_enabled: bool = _env_bool("INTENT_ROUTER_ENABLED", True)
    intent_llm_fallback: bool = _env_bool("INTENT_LLM_FALLBACK", True)
    intent_llm_history_max_chars: int = _env_int(
        "INTENT_LLM_HISTORY_MAX_CHARS",
        12000,
    )
    intent_llm_summary_max_chars: int = _env_int(
        "INTENT_LLM_SUMMARY_MAX_CHARS",
        32000,
    )
    intent_llm_max_tokens: int = _env_int("INTENT_LLM_MAX_TOKENS", 512)
    intent_embedding_history_max_chars: int = _env_int(
        "INTENT_EMBEDDING_HISTORY_MAX_CHARS",
        8000,
    )
    intent_embedding_summary_max_chars: int = _env_int(
        "INTENT_EMBEDDING_SUMMARY_MAX_CHARS",
        8000,
    )
    intent_embedding_text_max_chars: int = _env_int(
        "INTENT_EMBEDDING_TEXT_MAX_CHARS",
        12000,
    )
    intent_embedding_rag_threshold: float = _env_float(
        "INTENT_EMBEDDING_RAG_THRESHOLD",
        # Preserve manual-development compatibility with older .env files.
        _env_float("INTENT_EMBEDDING_DB_THRESHOLD", 0.38),
    )
    intent_embedding_direct_threshold: float = _env_float(
        "INTENT_EMBEDDING_DIRECT_THRESHOLD",
        0.40,
    )
    intent_embedding_margin: float = _env_float("INTENT_EMBEDDING_MARGIN", 0.06)

    def __post_init__(self) -> None:
        if not self.auth_enabled:
            object.__setattr__(self, "auth_enabled", True)
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be greater than or equal to 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.chunk_overlap and self.chunk_overlap >= self.chunk_size - 1:
            raise ValueError("CHUNK_OVERLAP must leave room for new chunk content.")
        if self.docs_source not in {"local", "s3"}:
            raise ValueError("DOCS_SOURCE must be local or s3.")
        if self.docs_source == "s3" and not self.docs_s3_bucket.strip():
            raise ValueError("DOCS_S3_BUCKET must not be empty when DOCS_SOURCE=s3.")
        if self.docs_s3_prefix.startswith("/"):
            raise ValueError("DOCS_S3_PREFIX must not start with '/'.")
        if self.docs_s3_manifest_prefix.startswith("/"):
            raise ValueError("DOCS_S3_MANIFEST_PREFIX must not start with '/'.")
        if self.docs_s3_retain_versions <= 0:
            raise ValueError("DOCS_S3_RETAIN_VERSIONS must be greater than 0.")
        if self.docs_s3_processing_retain_versions < self.docs_s3_retain_versions:
            raise ValueError(
                "DOCS_S3_PROCESSING_RETAIN_VERSIONS must be greater than or "
                "equal to DOCS_S3_RETAIN_VERSIONS."
            )
        if self.qdrant_retain_versions <= 0:
            raise ValueError("QDRANT_RETAIN_VERSIONS must be greater than 0.")
        if self.qdrant_processing_retain_versions < self.qdrant_retain_versions:
            raise ValueError(
                "QDRANT_PROCESSING_RETAIN_VERSIONS must be greater than or "
                "equal to QDRANT_RETAIN_VERSIONS."
            )
        if not self.embedding_model.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty.")
        if self.retrieve_top_k <= 0:
            raise ValueError("RETRIEVE_TOP_K must be greater than 0.")
        if self.bm25_top_k <= 0:
            raise ValueError("BM25_TOP_K must be greater than 0.")
        if self.recall_top_k <= 0:
            raise ValueError("RECALL_TOP_K must be greater than 0.")
        if self.rrf_top_k <= 0:
            raise ValueError("RRF_TOP_K must be greater than 0.")
        if self.retrieve_score_threshold < 0:
            raise ValueError(
                "RETRIEVE_SCORE_THRESHOLD must be greater than or equal to 0."
            )
        if self.api_top_k_max <= 0:
            raise ValueError("API_TOP_K_MAX must be greater than 0.")
        if self.api_recall_top_k_max <= 0:
            raise ValueError("API_RECALL_TOP_K_MAX must be greater than 0.")
        if not self.code_files_collection.strip():
            raise ValueError("CODE_FILES_COLLECTION must not be empty.")
        if not self.code_functions_collection.strip():
            raise ValueError("CODE_FUNCTIONS_COLLECTION must not be empty.")
        if not self.code_embedding_model.strip():
            raise ValueError("CODE_EMBEDDING_MODEL must not be empty.")
        if self.code_embedding_preload_retries <= 0:
            raise ValueError("CODE_EMBEDDING_PRELOAD_RETRIES must be greater than 0.")
        if self.code_embedding_preload_retry_seconds < 0:
            raise ValueError(
                "CODE_EMBEDDING_PRELOAD_RETRY_SECONDS must be greater than or equal to 0."
            )
        if self.code_embedding_batch_size <= 0:
            raise ValueError("CODE_EMBEDDING_BATCH_SIZE must be greater than 0.")
        if self.code_embedding_max_tokens <= 0:
            raise ValueError("CODE_EMBEDDING_MAX_TOKENS must be greater than 0.")
        if self.code_file_embedding_max_chars <= 0:
            raise ValueError("CODE_FILE_EMBEDDING_MAX_CHARS must be greater than 0.")
        if self.code_function_embedding_max_chars <= 0:
            raise ValueError(
                "CODE_FUNCTION_EMBEDDING_MAX_CHARS must be greater than 0."
            )
        if self.code_payload_snippet_chars <= 0:
            raise ValueError("CODE_PAYLOAD_SNIPPET_CHARS must be greater than 0.")
        if self.code_search_file_top_k <= 0:
            raise ValueError("CODE_SEARCH_FILE_TOP_K must be greater than 0.")
        if self.code_search_function_top_k <= 0:
            raise ValueError("CODE_SEARCH_FUNCTION_TOP_K must be greater than 0.")
        if self.code_search_final_top_k <= 0:
            raise ValueError("CODE_SEARCH_FINAL_TOP_K must be greater than 0.")
        if self.code_call_graph_depth <= 0:
            raise ValueError("CODE_CALL_GRAPH_DEPTH must be greater than 0.")
        if not self.reranker_model.strip():
            raise ValueError("RERANKER_MODEL must not be empty.")
        if not self.reranker_dtype.strip():
            raise ValueError("RERANKER_DTYPE must not be empty.")
        if self.reranker_max_documents_per_call <= 0:
            raise ValueError(
                "RERANKER_MAX_DOCUMENTS_PER_CALL must be greater than 0."
            )
        if self.ray_actor_num_cpus <= 0:
            raise ValueError("RAY_ACTOR_NUM_CPUS must be greater than 0.")
        if self.ray_embedding_actor_num_gpus < 0:
            raise ValueError(
                "RAY_EMBEDDING_ACTOR_NUM_GPUS must be greater than or equal to 0."
            )
        if self.ray_reranker_actor_num_gpus < 0:
            raise ValueError(
                "RAY_RERANKER_ACTOR_NUM_GPUS must be greater than or equal to 0."
            )
        if self.ray_reranker_actor_replicas <= 0:
            raise ValueError("RAY_RERANKER_ACTOR_REPLICAS must be greater than 0.")
        if self.ray_code_embedding_actor_num_gpus < 0:
            raise ValueError(
                "RAY_CODE_EMBEDDING_ACTOR_NUM_GPUS must be greater than or equal to 0."
            )
        if not self.ray_namespace.strip():
            raise ValueError("RAY_NAMESPACE must not be empty.")
        if not self.ray_embedding_actor_name.strip():
            raise ValueError("RAY_EMBEDDING_ACTOR_NAME must not be empty.")
        if not self.ray_code_embedding_actor_name.strip():
            raise ValueError("RAY_CODE_EMBEDDING_ACTOR_NAME must not be empty.")
        if not self.ray_reranker_actor_name.strip():
            raise ValueError("RAY_RERANKER_ACTOR_NAME must not be empty.")
        for field_name, resource_name in (
            ("RAY_EMBEDDING_ACTOR_RESOURCE", self.ray_embedding_actor_resource),
            (
                "RAY_CODE_EMBEDDING_ACTOR_RESOURCE",
                self.ray_code_embedding_actor_resource,
            ),
            ("RAY_RERANKER_ACTOR_RESOURCE", self.ray_reranker_actor_resource),
        ):
            if resource_name and not resource_name.replace("_", "").isalnum():
                raise ValueError(
                    f"{field_name} must contain only letters, numbers, or underscores."
                )
        if self.ray_task_timeout_seconds < 0:
            raise ValueError(
                "RAY_TASK_TIMEOUT_SECONDS must be greater than or equal to 0."
            )
        if self.health_probe_interval_seconds <= 0:
            raise ValueError("HEALTH_PROBE_INTERVAL_SECONDS must be greater than 0.")
        if self.health_probe_degraded_interval_seconds <= 0:
            raise ValueError(
                "HEALTH_PROBE_DEGRADED_INTERVAL_SECONDS must be greater than 0."
            )
        if self.health_probe_timeout_seconds <= 0:
            raise ValueError("HEALTH_PROBE_TIMEOUT_SECONDS must be greater than 0.")
        if self.health_probe_failure_threshold <= 0:
            raise ValueError("HEALTH_PROBE_FAILURE_THRESHOLD must be greater than 0.")
        if self.health_probe_recovery_threshold <= 0:
            raise ValueError("HEALTH_PROBE_RECOVERY_THRESHOLD must be greater than 0.")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0.")
        if self.llm_retry_attempts <= 0:
            raise ValueError("LLM_RETRY_ATTEMPTS must be greater than 0.")
        if self.llm_retry_backoff_seconds < 0:
            raise ValueError(
                "LLM_RETRY_BACKOFF_SECONDS must be greater than or equal to 0."
            )
        if self.llm_retry_backoff_max_seconds < self.llm_retry_backoff_seconds:
            raise ValueError(
                "LLM_RETRY_BACKOFF_MAX_SECONDS must be greater than or equal to "
                "LLM_RETRY_BACKOFF_SECONDS."
            )
        if self.llm_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be greater than 0.")
        if self.llm_context_max_tokens <= 0:
            raise ValueError("LLM_CONTEXT_MAX_TOKENS must be greater than 0.")
        if self.llm_context_safety_margin_tokens < 0:
            raise ValueError(
                "LLM_CONTEXT_SAFETY_MARGIN_TOKENS must be greater than or equal to 0."
            )
        if self.llm_context_prompt_overhead_tokens < 0:
            raise ValueError(
                "LLM_CONTEXT_PROMPT_OVERHEAD_TOKENS must be greater than or equal to 0."
            )
        if self.api_message_max_chars <= 0:
            raise ValueError("API_MESSAGE_MAX_CHARS must be greater than 0.")
        if self.api_question_max_chars <= 0:
            raise ValueError("API_QUESTION_MAX_CHARS must be greater than 0.")
        if self.api_summary_max_chars < 0:
            raise ValueError(
                "API_SUMMARY_MAX_CHARS must be greater than or equal to 0."
            )
        if self.api_history_max_messages < 0:
            raise ValueError(
                "API_HISTORY_MAX_MESSAGES must be greater than or equal to 0."
            )
        if self.database_connect_timeout_seconds <= 0:
            raise ValueError(
                "DATABASE_CONNECT_TIMEOUT_SECONDS must be greater than 0."
            )
        if self.auth_default_admin_enabled:
            if not self.auth_default_admin_username.strip():
                raise ValueError("AUTH_DEFAULT_ADMIN_USERNAME must not be empty.")
            if not self.auth_default_admin_password:
                raise ValueError("AUTH_DEFAULT_ADMIN_PASSWORD must not be empty.")
        if self.auth_session_ttl_seconds <= 0:
            raise ValueError("AUTH_SESSION_TTL_SECONDS must be greater than 0.")
        if self.session_list_limit <= 0:
            raise ValueError("SESSION_LIST_LIMIT must be greater than 0.")
        if self.session_title_max_chars <= 0:
            raise ValueError("SESSION_TITLE_MAX_CHARS must be greater than 0.")
        if self.intent_llm_max_tokens <= 0:
            raise ValueError("INTENT_LLM_MAX_TOKENS must be greater than 0.")


settings = Settings()
