from __future__ import annotations

from pathlib import Path
import sys

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.llm_client import LLMClient
from app.prompt_budget import PromptBudget
from app.security import is_authorized_api_request


def assert_api_limits() -> None:
    config = Settings(
        api_top_k_max=9,
        api_message_max_chars=123,
        api_question_max_chars=456,
        api_summary_max_chars=789,
        api_history_max_messages=10,
    )

    if config.api_top_k_max != 9:
        raise AssertionError("API top_k max should be configurable.")
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
        retrieve_score_threshold=0.42,
        intent_embedding_rag_threshold=0.55,
        api_auth_token="secret-token",
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
    if config.api_auth_token != "secret-token":
        raise AssertionError("API auth token should be configurable.")
    if config.llm_retry_attempts != 4:
        raise AssertionError("LLM retry attempts should be configurable.")
    if config.llm_retry_backoff_seconds != 0.5:
        raise AssertionError("LLM retry backoff should be configurable.")
    if config.llm_retry_backoff_max_seconds != 4.0:
        raise AssertionError("LLM retry max backoff should be configurable.")
    if config.retrieve_score_threshold != 0.42:
        raise AssertionError("Retrieval score threshold should be configurable.")

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


def assert_api_auth() -> None:
    open_config = Settings(api_auth_token="")
    if not is_authorized_api_request(open_config, None):
        raise AssertionError("Empty API auth token should leave local API open.")

    protected_config = Settings(api_auth_token="secret-token")
    if not is_authorized_api_request(protected_config, "Bearer secret-token"):
        raise AssertionError("Matching bearer token should authorize request.")

    rejected = [
        None,
        "",
        "secret-token",
        "Basic secret-token",
        "Bearer wrong-token",
    ]
    for authorization in rejected:
        if is_authorized_api_request(protected_config, authorization):
            raise AssertionError(f"Invalid authorization accepted: {authorization!r}")

    print("API bearer auth -> ok")


def assert_public_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    if response.status_code != 200 or response.json() != {"status": "ok"}:
        raise AssertionError("Public health endpoint should return minimal liveness.")

    print("Public health endpoint -> ok")


def assert_app_lifespan_recreates_resources() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
        if response.status_code != 200:
            raise AssertionError("Health endpoint should work during lifespan.")
        first_pipeline = app.state.rag_pipeline

    with TestClient(app) as client:
        response = client.get("/health")
        if response.status_code != 200:
            raise AssertionError("Health endpoint should work after restart.")
        second_pipeline = app.state.rag_pipeline

    if first_pipeline is second_pipeline:
        raise AssertionError("App lifespan should recreate runtime resources.")

    print("App lifespan resources -> ok")


def assert_rag_endpoint_accepts_original_body_shape() -> None:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.rag import RAGAnswer

    class FakePipeline:
        def answer(
            self,
            question: str,
            top_k: int | None = None,
            history: object | None = None,
            conversation_summary: str | None = None,
        ) -> RAGAnswer:
            return RAGAnswer(
                answer=f"echo: {question}",
                contexts=[],
                conversation_summary=conversation_summary or "",
                compacted_history_messages=0,
                used_rag=False,
                route="fallback_direct",
                route_reason="test",
            )

    with TestClient(app) as client:
        app.state.rag_pipeline = FakePipeline()
        response = client.post(
            "/rag",
            json={
                "question": "Hello",
                "top_k": 1,
                "history": [],
                "conversation_summary": "",
            },
        )

    if response.status_code != 200:
        raise AssertionError(f"RAG endpoint rejected original body: {response.text}")
    if response.json()["answer"] != "echo: Hello":
        raise AssertionError("RAG endpoint should use the parsed request body.")

    print("RAG endpoint body shape -> ok")


def assert_invalid_settings_rejected() -> None:
    invalid_configs = [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 99},
        {"llm_retry_attempts": 0},
        {
            "llm_retry_backoff_seconds": 2.0,
            "llm_retry_backoff_max_seconds": 1.0,
        },
        {"api_top_k_max": 0},
        {"retrieve_score_threshold": -0.1},
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


def main() -> None:
    assert_api_limits()
    assert_llm_settings()
    assert_llm_client_provider_helpers()
    assert_llm_client_input_validation()
    assert_llm_health_requests()
    assert_llm_retry_behavior()
    assert_llm_client_connection_reuse()
    assert_api_auth()
    assert_public_health_endpoint()
    assert_app_lifespan_recreates_resources()
    assert_rag_endpoint_accepts_original_body_shape()
    assert_invalid_settings_rejected()
    assert_prompt_budget_settings()


if __name__ == "__main__":
    main()
