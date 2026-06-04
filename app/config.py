from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


@dataclass(frozen=True)
class Settings:
    debug: bool = _env_bool("DEBUG", False)

    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "tech_docs")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_size: int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 120)
    retrieve_top_k: int = _env_int("RETRIEVE_TOP_K", 4)
    retrieve_score_threshold: float = _env_float("RETRIEVE_SCORE_THRESHOLD", 0.0)
    api_top_k_max: int = _env_int("API_TOP_K_MAX", 20)

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai_compatible")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
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
    api_auth_token: str = os.getenv("API_AUTH_TOKEN", "")

    history_recent_turns: int = _env_int("HISTORY_RECENT_TURNS", 16)
    history_compact_after_turns: int = _env_int("HISTORY_COMPACT_AFTER_TURNS", 40)
    history_max_messages: int = _env_int("HISTORY_MAX_MESSAGES", 120)
    message_max_chars: int = _env_int("MESSAGE_MAX_CHARS", 8000)
    conversation_summary_max_chars: int = _env_int(
        "CONVERSATION_SUMMARY_MAX_CHARS",
        6000,
    )
    summary_history_max_chars: int = _env_int("SUMMARY_HISTORY_MAX_CHARS", 20000)
    summary_max_tokens: int = _env_int("SUMMARY_MAX_TOKENS", 1200)
    search_query_max_chars: int = _env_int("SEARCH_QUERY_MAX_CHARS", 3000)

    intent_router_enabled: bool = _env_bool("INTENT_ROUTER_ENABLED", True)
    intent_llm_fallback: bool = _env_bool("INTENT_LLM_FALLBACK", True)
    intent_llm_history_max_chars: int = _env_int(
        "INTENT_LLM_HISTORY_MAX_CHARS",
        4000,
    )
    intent_llm_summary_max_chars: int = _env_int(
        "INTENT_LLM_SUMMARY_MAX_CHARS",
        2500,
    )
    intent_llm_max_tokens: int = _env_int("INTENT_LLM_MAX_TOKENS", 120)
    intent_embedding_history_max_chars: int = _env_int(
        "INTENT_EMBEDDING_HISTORY_MAX_CHARS",
        5000,
    )
    intent_embedding_summary_max_chars: int = _env_int(
        "INTENT_EMBEDDING_SUMMARY_MAX_CHARS",
        2500,
    )
    intent_embedding_text_max_chars: int = _env_int(
        "INTENT_EMBEDDING_TEXT_MAX_CHARS",
        7000,
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
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be greater than or equal to 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.chunk_overlap and self.chunk_overlap >= self.chunk_size - 1:
            raise ValueError("CHUNK_OVERLAP must leave room for new chunk content.")
        if self.retrieve_top_k <= 0:
            raise ValueError("RETRIEVE_TOP_K must be greater than 0.")
        if self.retrieve_score_threshold < 0:
            raise ValueError(
                "RETRIEVE_SCORE_THRESHOLD must be greater than or equal to 0."
            )
        if self.api_top_k_max <= 0:
            raise ValueError("API_TOP_K_MAX must be greater than 0.")
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


settings = Settings()
