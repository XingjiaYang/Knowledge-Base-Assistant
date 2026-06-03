from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.llm_client import LLMClient
from app.prompt_budget import PromptBudget


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
        llm_health_path="/models",
        llm_anthropic_version="2023-06-01",
        intent_embedding_rag_threshold=0.55,
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
    assert_prompt_budget_settings()


if __name__ == "__main__":
    main()
