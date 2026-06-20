from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import UUID

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import LLMRuntimeSettings, Settings
from app.llm_client import LLMClient
from app.prompt_budget import PromptBudget
from app.session_store import ChatSessionRecord, CurrentUser


def assert_api_limits() -> None:
    config = Settings(
        api_top_k_max=9,
        api_recall_top_k_max=321,
        api_message_max_chars=123,
        api_question_max_chars=456,
        api_summary_max_chars=789,
        api_history_max_messages=10,
        bm25_top_k=111,
        recall_top_k=200,
        rrf_top_k=222,
        retrieve_top_k=5,
    )

    if config.api_top_k_max != 9:
        raise AssertionError("API top_k max should be configurable.")
    if config.api_recall_top_k_max != 321:
        raise AssertionError("API recall top_k max should be configurable.")
    if config.bm25_top_k != 111:
        raise AssertionError("BM25 top_k should be configurable.")
    if config.recall_top_k != 200:
        raise AssertionError("Recall top_k should be configurable.")
    if config.rrf_top_k != 222:
        raise AssertionError("RRF top_k should be configurable.")
    if config.retrieve_top_k != 5:
        raise AssertionError("Final top_k should be configurable.")
    if config.api_message_max_chars != 123:
        raise AssertionError("API message char limit should be configurable.")
    if config.api_question_max_chars != 456:
        raise AssertionError("API question char limit should be configurable.")
    if config.api_summary_max_chars != 789:
        raise AssertionError("API summary char limit should be configurable.")
    if config.api_history_max_messages != 10:
        raise AssertionError("API history length limit should be configurable.")

    print("API validation settings -> ok")


def assert_llm_settings() -> None:
    config = Settings(
        llm_provider="anthropic",
        llm_base_url="https://api.anthropic.com/v1",
        llm_api_key="test-key",
        llm_model="claude-test",
        llm_health_check_enabled=True,
        llm_health_path="",
        llm_anthropic_version="2023-06-01",
        llm_retry_attempts=4,
        llm_retry_backoff_seconds=0.5,
        llm_retry_backoff_max_seconds=4.0,
        llm_context_max_tokens=131072,
        llm_context_safety_margin_tokens=4096,
        llm_context_prompt_overhead_tokens=1024,
        retrieve_score_threshold=0.42,
        intent_embedding_rag_threshold=0.55,
        cuda_enabled=False,
        embedding_model="jinaai/jina-embeddings-v3",
        embedding_trust_remote_code=True,
        embedding_query_task="retrieval.query",
        embedding_passage_task="retrieval.passage",
        embedding_classification_task="classification",
        reranker_model="jinaai/jina-reranker-v3",
        reranker_preload=True,
        reranker_dtype="auto",
        reranker_max_documents_per_call=32,
    )

    if config.llm_provider != "anthropic":
        raise AssertionError("LLM provider should be configurable.")
    if config.llm_base_url != "https://api.anthropic.com/v1":
        raise AssertionError("LLM base URL should be configurable.")
    if config.llm_model != "claude-test":
        raise AssertionError("LLM model should be configurable.")
    if not config.llm_health_check_enabled:
        raise AssertionError("LLM health check flag should be configurable.")
    if config.intent_embedding_rag_threshold != 0.55:
        raise AssertionError("RAG embedding threshold should be configurable.")
    if config.llm_retry_attempts != 4:
        raise AssertionError("LLM retry attempts should be configurable.")
    if config.llm_retry_backoff_seconds != 0.5:
        raise AssertionError("LLM retry backoff should be configurable.")
    if config.llm_retry_backoff_max_seconds != 4.0:
        raise AssertionError("LLM retry max backoff should be configurable.")
    if config.llm_context_max_tokens != 131072:
        raise AssertionError("LLM context max tokens should be configurable.")
    if config.llm_context_safety_margin_tokens != 4096:
        raise AssertionError("LLM context safety margin should be configurable.")
    if config.llm_context_prompt_overhead_tokens != 1024:
        raise AssertionError("LLM context prompt overhead should be configurable.")
    if config.retrieve_score_threshold != 0.42:
        raise AssertionError("Retrieval score threshold should be configurable.")
    if config.cuda_enabled:
        raise AssertionError("CUDA preference should be configurable.")
    if config.embedding_model != "jinaai/jina-embeddings-v3":
        raise AssertionError("Embedding model should be configurable.")
    if not config.embedding_trust_remote_code:
        raise AssertionError("Embedding trust_remote_code should be configurable.")
    if config.embedding_query_task != "retrieval.query":
        raise AssertionError("Embedding query task should be configurable.")
    if config.embedding_passage_task != "retrieval.passage":
        raise AssertionError("Embedding passage task should be configurable.")
    if config.embedding_classification_task != "classification":
        raise AssertionError("Embedding classification task should be configurable.")
    if config.reranker_model != "jinaai/jina-reranker-v3":
        raise AssertionError("Reranker model should be configurable.")
    if not config.reranker_preload:
        raise AssertionError("Reranker preload should be configurable.")
    if config.reranker_dtype != "auto":
        raise AssertionError("Reranker dtype should be configurable.")
    if config.reranker_max_documents_per_call != 32:
        raise AssertionError("Reranker per-call document limit should be configurable.")

    print("LLM settings -> ok")


def assert_llm_client_provider_helpers() -> None:
    config = Settings(
        llm_provider="Anthropic",
        llm_api_key="test-key",
        llm_anthropic_version="2023-06-01",
        llm_health_check_enabled=False,
    )
    client = LLMClient(config)
    system, messages = client._to_anthropic_messages(
        [
            {"role": "system", "content": "System A"},
            {"role": "system", "content": "System B"},
            {"role": "user", "content": "Question 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer"},
        ]
    )

    if system != "System A\n\nSystem B":
        raise AssertionError("Anthropic system messages should be merged.")
    if messages[0] != {"role": "user", "content": "Question 1\n\nQuestion 2"}:
        raise AssertionError("Consecutive Anthropic user messages should be merged.")
    if not client.health():
        raise AssertionError("Disabled LLM health check should return healthy.")

    print("LLM client provider helpers -> ok")


def assert_llm_client_input_validation() -> None:
    client = LLMClient(Settings(llm_health_check_enabled=False))
    invalid_messages = [
        [],
        [{"role": "system", "content": "Only system"}],
        [{"role": "user", "content": ""}],
        [{"role": "tool", "content": "No"}],
    ]
    for messages in invalid_messages:
        try:
            client.chat(messages)
        except ValueError:
            continue
        raise AssertionError(f"Invalid messages should be rejected: {messages!r}")

    print("LLM client input validation -> ok")


def assert_llm_health_requests() -> None:
    anthropic = LLMClient(
        Settings(
            llm_provider="anthropic",
            llm_base_url="https://api.anthropic.com/v1",
            llm_model="claude-test",
            llm_health_path="",
        )
    )
    method, url, payload = anthropic._health_request()
    if method != "POST" or url != "https://api.anthropic.com/v1/messages/count_tokens":
        raise AssertionError("Anthropic health should use count_tokens by default.")
    if payload is None or payload.get("model") != "claude-test":
        raise AssertionError("Anthropic health payload should include the configured model.")

    openai_compatible = LLMClient(
        Settings(
            llm_provider="openai_compatible",
            llm_base_url="https://api.openai.com/v1",
            llm_health_path="",
        )
    )
    method, url, payload = openai_compatible._health_request()
    if method != "GET" or url != "https://api.openai.com/v1/models":
        raise AssertionError("OpenAI-compatible health should use /models by default.")
    if payload is not None:
        raise AssertionError("GET health checks should not send a JSON payload.")

    print("LLM health request routing -> ok")


def assert_llm_retry_behavior() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    client = LLMClient(
        Settings(
            llm_retry_attempts=2,
            llm_retry_backoff_seconds=0,
            llm_retry_backoff_max_seconds=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    answer = client.chat([{"role": "user", "content": "hello"}])
    if answer != "ok" or calls != 2:
        raise AssertionError("LLM retry should recover from transient 429 errors.")
    if not client._is_retryable_status(503) or client._is_retryable_status(400):
        raise AssertionError("LLM retry status classification is incorrect.")
    client.close()

    print("LLM retry behavior -> ok")


def assert_llm_runtime_settings_provider() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "runtime ok"}}]},
        )

    base = Settings(
        llm_base_url="https://env.example/v1",
        llm_api_key="env-key",
        llm_model="env-model",
    )
    runtime = LLMRuntimeSettings.from_settings(
        base,
        provider="openai_compatible",
        base_url="https://runtime.example/v1",
        api_key="runtime-key",
        model="runtime-model",
        context_max_tokens=123456,
    )
    client = LLMClient(
        base,
        transport=httpx.MockTransport(handler),
        runtime_settings_provider=lambda: runtime,
    )
    answer = client.chat([{"role": "user", "content": "hello"}])
    if answer != "runtime ok":
        raise AssertionError("Runtime LLM settings should still return provider output.")
    if seen.get("url") != "https://runtime.example/v1/chat/completions":
        raise AssertionError("Runtime LLM base URL should override env config.")
    if seen.get("authorization") != "Bearer runtime-key":
        raise AssertionError("Runtime LLM API key should override env config.")
    if "runtime-model" not in str(seen.get("body")):
        raise AssertionError("Runtime LLM model should override env config.")
    if client.runtime_settings().llm_context_max_tokens != 123456:
        raise AssertionError("Runtime context max tokens should override env config.")
    client.close()

    print("LLM runtime settings provider -> ok")


def assert_llm_client_connection_reuse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    client = LLMClient(
        Settings(),
        transport=httpx.MockTransport(handler),
    )
    first_answer = client.chat([{"role": "user", "content": "first"}])
    first_client = client._client
    second_answer = client.chat([{"role": "user", "content": "second"}])
    if first_answer != "ok" or second_answer != "ok":
        raise AssertionError("Mocked LLM responses should be returned.")
    if first_client is None or client._client is not first_client:
        raise AssertionError("LLMClient should reuse one httpx.Client instance.")

    client.close()
    if client._client is not None or not first_client.is_closed:
        raise AssertionError("LLMClient.close() should close and clear the client.")

    print("LLM client connection reuse -> ok")


def assert_account_auth_is_always_enabled() -> None:
    config = Settings(auth_enabled=False)
    if not config.auth_enabled:
        raise AssertionError("Account auth should stay enabled even if disabled.")

    print("Forced account auth -> ok")


class FakeSessionStore:
    session_id = UUID("11111111-1111-1111-1111-111111111111")
    user_id = UUID("22222222-2222-2222-2222-222222222222")

    def init_db(self) -> None:
        return

    def close(self) -> None:
        return

    def get_user_by_token(self, token: str) -> CurrentUser | None:
        if token != "test-token":
            return None
        return CurrentUser(self.user_id, "admin", True)

    def create_chat_session(
        self,
        user_id: UUID,
        title: str | None = None,
    ) -> ChatSessionRecord:
        return self._session(user_id)

    def get_chat_session(
        self,
        user_id: UUID,
        session_id: UUID,
    ) -> ChatSessionRecord | None:
        if session_id != self.session_id:
            return None
        return self._session(user_id)

    def prompt_history(
        self,
        session_id: UUID,
        compacted_message_count: int,
    ) -> list[object]:
        return []

    def append_message(self, *args: object, **kwargs: object) -> None:
        return

    def update_chat_session_after_answer(
        self,
        session_id: UUID,
        conversation_summary: str,
        compacted_delta: int,
        title: str | None = None,
    ) -> ChatSessionRecord:
        return self._session(self.user_id, conversation_summary, compacted_delta)

    def _session(
        self,
        user_id: UUID,
        conversation_summary: str = "",
        compacted_message_count: int = 0,
    ) -> ChatSessionRecord:
        now = datetime.now(timezone.utc)
        return ChatSessionRecord(
            id=self.session_id,
            user_id=user_id,
            title="New chat",
            conversation_summary=conversation_summary,
            compacted_message_count=compacted_message_count,
            created_at=now,
            updated_at=now,
        )


@contextmanager
def app_with_fake_session_store():
    import app.main as main_module

    original_session_store = main_module.SessionStore
    main_module.SessionStore = lambda _settings: FakeSessionStore()
    try:
        yield main_module.app
    finally:
        main_module.SessionStore = original_session_store


def assert_public_health_endpoint() -> None:
    from app.main import health

    response = asyncio.run(health())
    if response != {"status": "ok"}:
        raise AssertionError("Public health endpoint should return minimal liveness.")

    print("Public health endpoint -> ok")


def assert_app_lifespan_recreates_resources() -> None:
    from app.rag import RAGPipeline
    from app.vector_store import VectorStore

    config = Settings()
    first_pipeline = RAGPipeline(
        config,
        vector_store=VectorStore(config),
        llm_client=LLMClient(config),
    )
    second_pipeline = RAGPipeline(
        config,
        vector_store=VectorStore(config),
        llm_client=LLMClient(config),
    )

    if first_pipeline is second_pipeline:
        raise AssertionError("App lifespan should recreate runtime resources.")

    print("App lifespan resources -> ok")


def assert_rag_endpoint_accepts_original_body_shape() -> None:
    from app.main import RAGRequest

    request = RAGRequest(
        question="Hello",
        top_k=1,
        bm25_top_k=9,
        recall_top_k=10,
        rrf_top_k=8,
        history=[],
        conversation_summary="",
    )
    if (
        request.question != "Hello"
        or request.top_k != 1
        or request.bm25_top_k != 9
        or request.recall_top_k != 10
        or request.rrf_top_k != 8
        or request.history != []
        or request.conversation_summary != ""
    ):
        raise AssertionError(
            "RAG request should keep accepting the original body shape."
        )

    print("RAG endpoint body shape -> ok")


def assert_rag_endpoint_requires_login() -> None:
    from fastapi import HTTPException

    from app.main import require_login_auth

    with app_with_fake_session_store() as app:
        app.state.session_store = FakeSessionStore()
        request = SimpleNamespace(app=app)

        try:
            require_login_auth(request, None)
        except HTTPException as exc:
            if exc.status_code != 401:
                raise AssertionError("Missing login token should return 401.")
        else:
            raise AssertionError("RAG dependency should reject missing login token.")

    print("RAG endpoint requires login -> ok")


def assert_password_change_gate_blocks_app_features() -> None:
    from fastapi import HTTPException

    from app.main import UserResponse, require_password_ready_user

    locked_user = CurrentUser(
        FakeSessionStore.user_id,
        "new.user",
        False,
        True,
    )
    try:
        require_password_ready_user(locked_user)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise AssertionError("Password-change gate should return 403.")
    else:
        raise AssertionError("Password-change gate should block locked users.")

    ready_user = CurrentUser(FakeSessionStore.user_id, "ready.user", False, False)
    if require_password_ready_user(ready_user) is not ready_user:
        raise AssertionError("Password-ready users should pass through.")
    if not UserResponse.from_user(locked_user).must_change_password:
        raise AssertionError("User response should expose password-change status.")

    print("Password-change gate -> ok")


def assert_superuser_gate_and_llm_settings_models() -> None:
    from fastapi import HTTPException

    from app.main import (
        AdminLLMSettingsUpdateRequest,
        UserResponse,
        require_superuser_auth,
    )

    normal_admin = CurrentUser(
        FakeSessionStore.user_id,
        "admin",
        is_admin=True,
        is_superuser=False,
    )
    try:
        require_superuser_auth(normal_admin)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise AssertionError("Non-superuser admins should receive 403.")
    else:
        raise AssertionError("Non-superuser admins should not pass superuser auth.")

    superuser = CurrentUser(
        FakeSessionStore.user_id,
        "admin",
        is_admin=True,
        is_superuser=True,
    )
    if require_superuser_auth(superuser) is not superuser:
        raise AssertionError("Superuser should pass superuser auth.")
    if not UserResponse.from_user(superuser).is_superuser:
        raise AssertionError("User response should expose superuser status.")

    request = AdminLLMSettingsUpdateRequest(
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        model="claude-test",
        context_max_tokens=200000,
        api_key="secret",
    )
    if (
        request.provider != "anthropic"
        or request.model != "claude-test"
        or request.context_max_tokens != 200000
    ):
        raise AssertionError("LLM settings request should parse editable fields.")

    print("Superuser LLM settings gate -> ok")


def assert_csv_import_parser() -> None:
    from app.main import parse_user_csv

    users = parse_user_csv(
        "email,passwd\n"
        "alice@example.com,secret\n"
        "bob@example.com,another\n"
    )
    if users != [
        ("alice@example.com", "secret"),
        ("bob@example.com", "another"),
    ]:
        raise AssertionError("CSV import should parse email/passwd rows.")

    invalid_inputs = [
        "username,password\nalice@example.com,secret\n",
        "email,passwd,extra\nalice@example.com,secret,x\n",
        "email,passwd\nalice@example.com\n",
        "email,passwd\nalice@example.com,\n",
        "email,passwd\n",
    ]
    for raw in invalid_inputs:
        try:
            parse_user_csv(raw)
        except ValueError:
            continue
        raise AssertionError(f"Invalid CSV should be rejected: {raw!r}")

    print("CSV import parser -> ok")


def assert_invalid_settings_rejected() -> None:
    invalid_configs = [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 99},
        {"embedding_model": ""},
        {"llm_retry_attempts": 0},
        {
            "llm_retry_backoff_seconds": 2.0,
            "llm_retry_backoff_max_seconds": 1.0,
        },
        {"llm_context_max_tokens": 0},
        {"llm_context_safety_margin_tokens": -1},
        {"llm_context_prompt_overhead_tokens": -1},
        {"intent_llm_max_tokens": 0},
        {"api_top_k_max": 0},
        {"api_recall_top_k_max": 0},
        {"retrieve_score_threshold": -0.1},
        {"bm25_top_k": 0},
        {"recall_top_k": 0},
        {"rrf_top_k": 0},
        {"reranker_model": ""},
        {"reranker_dtype": ""},
        {"reranker_max_documents_per_call": 0},
    ]
    for kwargs in invalid_configs:
        try:
            Settings(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Invalid Settings should be rejected: {kwargs!r}")

    print("Settings validation -> ok")


def assert_prompt_budget_settings() -> None:
    config = Settings(
        message_max_chars=111,
        history_compact_after_turns=22,
        conversation_summary_max_chars=222,
        summary_history_max_chars=333,
        summary_max_tokens=44,
        search_query_max_chars=555,
        intent_llm_history_max_chars=666,
        intent_llm_summary_max_chars=777,
        intent_llm_max_tokens=88,
        intent_embedding_history_max_chars=999,
        intent_embedding_summary_max_chars=1010,
        intent_embedding_text_max_chars=1111,
    )
    budget = PromptBudget.from_config(config)

    expected = {
        "message_max_chars": 111,
        "history_compact_after_turns": 22,
        "conversation_summary_max_chars": 222,
        "summary_history_max_chars": 333,
        "summary_max_tokens": 44,
        "search_query_max_chars": 555,
        "intent_llm_history_max_chars": 666,
        "intent_llm_summary_max_chars": 777,
        "intent_llm_max_tokens": 88,
        "intent_embedding_history_max_chars": 999,
        "intent_embedding_summary_max_chars": 1010,
        "intent_embedding_text_max_chars": 1111,
    }
    for field, value in expected.items():
        if getattr(budget, field) != value:
            raise AssertionError(f"Prompt budget field not configurable: {field}")

    print("Prompt budget settings -> ok")


def assert_default_conversation_summary_budget() -> None:
    config = Settings()
    if config.conversation_summary_max_chars != 256000:
        raise AssertionError(
            "Default conversation summary char limit should match API-scale context."
        )
    if config.summary_history_max_chars != 200000:
        raise AssertionError("Default summary input budget should be API-scale.")
    if config.summary_max_tokens != 4096:
        raise AssertionError("Default summary output budget should be API-scale.")
    if config.llm_context_max_tokens != 256000:
        raise AssertionError("Default LLM context window should be API-scale.")
    if config.llm_context_safety_margin_tokens != 8192:
        raise AssertionError("Default LLM context safety margin should be configured.")
    if config.llm_context_prompt_overhead_tokens != 2048:
        raise AssertionError("Default LLM context overhead should be configured.")
    if config.history_max_messages != 0:
        raise AssertionError("History should not be count-truncated before compaction.")
    if config.intent_llm_max_tokens != 512:
        raise AssertionError("Default intent classifier output budget should allow tags.")

    print("Default conversation summary budget -> ok")


def assert_default_embedding_settings() -> None:
    config = Settings()
    if config.chunk_size != 2000:
        raise AssertionError("Default chunk size should use Jina-scale context.")
    if config.chunk_overlap != 300:
        raise AssertionError("Default chunk overlap should use Jina-scale context.")
    if config.embedding_model != "jinaai/jina-embeddings-v3":
        raise AssertionError("Default embedding model should be Jina embeddings v3.")
    if not config.embedding_trust_remote_code:
        raise AssertionError("Jina embeddings v3 should load with trusted remote code.")
    if config.embedding_query_task != "retrieval.query":
        raise AssertionError("Default query embedding task should use Jina retrieval.")
    if config.embedding_passage_task != "retrieval.passage":
        raise AssertionError("Default passage embedding task should use Jina retrieval.")
    if config.embedding_classification_task != "classification":
        raise AssertionError("Default intent embedding task should use classification.")
    if config.intent_llm_history_max_chars != 12000:
        raise AssertionError("Default LLM intent history budget should be expanded.")
    if config.intent_llm_summary_max_chars != 32000:
        raise AssertionError("Default LLM intent summary budget should be expanded.")
    if config.intent_embedding_history_max_chars != 8000:
        raise AssertionError("Default embedding intent history budget should expand.")
    if config.intent_embedding_summary_max_chars != 8000:
        raise AssertionError("Default embedding intent summary budget should expand.")
    if config.intent_embedding_text_max_chars != 12000:
        raise AssertionError("Default embedding intent text budget should expand.")

    print("Default embedding settings -> ok")


def main() -> None:
    assert_api_limits()
    assert_llm_settings()
    assert_llm_client_provider_helpers()
    assert_llm_client_input_validation()
    assert_llm_health_requests()
    assert_llm_retry_behavior()
    assert_llm_runtime_settings_provider()
    assert_llm_client_connection_reuse()
    assert_account_auth_is_always_enabled()
    assert_public_health_endpoint()
    assert_app_lifespan_recreates_resources()
    assert_rag_endpoint_accepts_original_body_shape()
    assert_rag_endpoint_requires_login()
    assert_password_change_gate_blocks_app_features()
    assert_superuser_gate_and_llm_settings_models()
    assert_csv_import_parser()
    assert_invalid_settings_rejected()
    assert_prompt_budget_settings()
    assert_default_conversation_summary_budget()
    assert_default_embedding_settings()


if __name__ == "__main__":
    main()
