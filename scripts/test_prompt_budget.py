from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.intent_router import IntentDecision
from app.prompt_budget import TRUNCATION_MARKER
from app.rag import ChatMessage, RAGPipeline
from app.vector_store import SearchResult


class FakeVectorStore:
    model = None

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return []


class FakeLLMClient:
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "summary"


class FakeRouter:
    def route(
        self,
        question: str,
        recent_history: list[ChatMessage],
        conversation_summary: str,
    ) -> IntentDecision:
        return IntentDecision(False, "fallback_direct", "test")


def make_pipeline(config: Settings) -> RAGPipeline:
    return RAGPipeline(
        config,
        vector_store=FakeVectorStore(),
        llm_client=FakeLLMClient(),
        intent_router=FakeRouter(),
    )


def assert_summary_middle_truncation() -> None:
    config = Settings(conversation_summary_max_chars=120)
    pipeline = make_pipeline(config)
    summary = (
        "CORE_GOAL: choose the right database. "
        + ("x" * 160)
        + " LATEST_DECISION: keep Qdrant for retrieval."
    )

    trimmed = pipeline._trim_summary(summary)
    if len(trimmed) > config.conversation_summary_max_chars:
        raise AssertionError("Summary trim exceeded configured max chars.")
    if "CORE_GOAL" not in trimmed or "Qdrant" not in trimmed:
        raise AssertionError("Summary trim should preserve both head and tail.")
    if TRUNCATION_MARKER.strip() not in trimmed:
        raise AssertionError("Summary trim should mark removed middle content.")

    print("Summary middle truncation -> ok")


def assert_history_normalization_budget() -> None:
    config = Settings(history_max_messages=2, message_max_chars=60)
    pipeline = make_pipeline(config)
    long_message = "MESSAGE_START " + ("x" * 120) + " MESSAGE_END"

    clean = pipeline._normalize_history(
        [
            ChatMessage("user", "old message"),
            ChatMessage("user", long_message),
            ChatMessage("assistant", "recent message"),
        ]
    )

    if len(clean) != 2:
        raise AssertionError("History max message count was not applied.")
    if clean[0].role != "user" or clean[1].content != "recent message":
        raise AssertionError("History should keep the most recent messages.")
    if (
        "MESSAGE_START" not in clean[0].content
        or "MESSAGE_END" not in clean[0].content
    ):
        raise AssertionError("Message trim should preserve both head and tail.")

    print("History normalization budget -> ok")


def assert_search_query_budget() -> None:
    config = Settings(search_query_max_chars=80)
    pipeline = make_pipeline(config)
    history = [
        ChatMessage("user", "older context " + ("x" * 100)),
        ChatMessage("user", "recent context " + ("y" * 100)),
    ]

    query = pipeline._build_search_query(
        "CURRENT_QUESTION: compare choices",
        history,
    )
    if len(query) > config.search_query_max_chars:
        raise AssertionError("Search query exceeded configured max chars.")
    if "CURRENT_QUESTION" not in query:
        raise AssertionError("Search query trimming should preserve current question.")

    print("Search query budget -> ok")


def assert_compaction_threshold() -> None:
    config = Settings(
        history_recent_turns=2,
        history_max_messages=0,
        llm_context_max_tokens=100000,
    )
    pipeline = make_pipeline(config)
    history = [
        ChatMessage("user" if idx % 2 == 0 else "assistant", f"message {idx}")
        for idx in range(82)
    ]

    summary, recent, compacted_count = pipeline._compact_history(history, "")
    if summary or len(recent) != 82 or compacted_count != 0:
        raise AssertionError("History should not compact only because turns are old.")

    config = Settings(
        history_recent_turns=2,
        llm_context_max_tokens=5000,
        llm_context_safety_margin_tokens=0,
        llm_context_prompt_overhead_tokens=0,
        llm_max_tokens=1,
        retrieve_top_k=1,
        chunk_body_target_tokens=10,
        chunk_body_max_tokens=20,
        chunk_overlap_target_tokens=0,
        chunk_overlap_max_tokens=0,
    )
    pipeline = make_pipeline(config)
    history = [
        ChatMessage("user" if idx % 2 == 0 else "assistant", "历史" * 1000)
        for idx in range(10)
    ]
    summary, recent, compacted_count = pipeline._compact_history(history, "")
    if summary != "summary":
        raise AssertionError("History should summarize when context budget is tight.")
    if len(recent) != 4:
        raise AssertionError("Compaction should preserve configured recent turns.")
    if compacted_count != 6:
        raise AssertionError("Compacted count should include older messages.")

    print("Context-budget compaction threshold -> ok")


def main() -> None:
    assert_summary_middle_truncation()
    assert_history_normalization_budget()
    assert_search_query_budget()
    assert_compaction_threshold()


if __name__ == "__main__":
    main()
