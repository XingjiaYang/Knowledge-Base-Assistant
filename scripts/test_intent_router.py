from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from threading import Lock, Thread
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.intent_router import IntentRouter
from app.rag import RAGPipeline
from app.vector_store import SearchResult


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class FakeVectorStore:
    model = None

    def __init__(self) -> None:
        self.search_calls = 0
        self.search_top_ks: list[int | None] = []

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.search_calls += 1
        self.search_top_ks.append(top_k)
        return [
            SearchResult(
                text="New employees request equipment through the IT portal.",
                source="data/docs/onboarding.md",
                chunk_id=0,
                score=0.9,
            ),
            SearchResult(
                text="Managers approve equipment requests within two business days.",
                source="data/docs/onboarding.md",
                chunk_id=1,
                score=0.8,
            ),
            SearchResult(
                text="Finance handles laptop budget exceptions.",
                source="data/docs/onboarding.md",
                chunk_id=2,
                score=0.7,
            )
        ]


class FakeLLMClient:
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "ok"


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    def rerank(
        self,
        query: str,
        contexts: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, len(contexts), top_k))
        return list(reversed(contexts))[:top_k or len(contexts)]


class FakeClassifierLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.response


class FakeIntentEmbedder:
    def encode(
        self,
        texts: str | list[str] | tuple[str, ...],
        normalize_embeddings: bool = True,
    ) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        lowered = text.lower()
        direct_markers = (
            "casual conversation",
            "creative writing",
            "explicitly do not need",
            "闲聊",
            "你好",
            "谢谢",
        )
        if any(marker in lowered for marker in direct_markers):
            return [0.0, 1.0]

        rag_markers = (
            "knowledge base documentation",
            "questions asking for comparisons",
            "troubleshooting questions",
            "知识库",
            "employee handbook",
            "onboarding",
            "vacation policy",
        )
        if any(marker in lowered for marker in rag_markers):
            return [1.0, 0.0]

        return [0.5, 0.5]


class CountingIntentEmbedder(FakeIntentEmbedder):
    def __init__(self) -> None:
        self.anchor_encode_calls = 0
        self._lock = Lock()

    def encode(
        self,
        texts: str | list[str] | tuple[str, ...],
        normalize_embeddings: bool = True,
    ) -> list[float] | list[list[float]]:
        if not isinstance(texts, str) and len(texts) > 1:
            time.sleep(0.01)
            with self._lock:
                self.anchor_encode_calls += 1
        return super().encode(texts, normalize_embeddings=normalize_embeddings)


def assert_route(
    router: IntentRouter,
    question: str,
    expected_use_rag: bool,
    history: list[Message] | None = None,
    expected_route: str | None = None,
    conversation_summary: str = "",
) -> None:
    decision = router.route(question, history or [], conversation_summary)
    if decision.use_rag != expected_use_rag:
        raise AssertionError(
            f"{question!r} routed to {decision.use_rag}, "
            f"expected {expected_use_rag}. Decision: {decision}"
        )
    if expected_route is not None and decision.route != expected_route:
        raise AssertionError(
            f"{question!r} used route {decision.route}, "
            f"expected {expected_route}. Decision: {decision}"
        )
    print(f"{question!r} -> {decision.route}: {decision.reason}")


def assert_pipeline_search_behavior() -> None:
    config = Settings()
    vector_store = FakeVectorStore()
    llm_client = FakeLLMClient()
    reranker = FakeReranker()
    router = IntentRouter(config, embedder=None, llm_client=None)
    pipeline = RAGPipeline(
        config,
        vector_store=vector_store,
        llm_client=llm_client,
        intent_router=router,
        reranker=reranker,
    )

    direct_answer = pipeline.answer("写一首短诗")
    if direct_answer.used_rag or vector_store.search_calls != 0 or reranker.calls:
        raise AssertionError("Direct route should not call vector search.")

    rag_answer = pipeline.answer("新员工如何申请开发设备？", top_k=2, recall_top_k=7)
    if not rag_answer.used_rag or vector_store.search_calls != 1:
        raise AssertionError("Knowledge-base route should call vector search once.")
    if vector_store.search_top_ks != [7]:
        raise AssertionError("Vector recall should use recall_top_k.")
    if reranker.calls != [("新员工如何申请开发设备？", 3, 2)]:
        raise AssertionError(f"Reranker should receive recalled contexts: {reranker.calls}")
    if [context.chunk_id for context in rag_answer.contexts] != [2, 1]:
        raise AssertionError("RAG answer should keep reranked final top_k contexts.")

    print("Pipeline search behavior -> ok")


def assert_embedding_context_behavior() -> None:
    router = IntentRouter(Settings(), embedder=FakeIntentEmbedder(), llm_client=None)
    history = [
        Message("user", "员工手册里的入职流程是什么？"),
        Message("assistant", "入职流程包括账号开通和设备申请。"),
    ]

    classification_text = router._classification_text(
        "设备申请具体怎么做？",
        history,
        "The user is reviewing the employee onboarding process.",
    )
    if "USER: 员工手册里的入职流程是什么？" not in classification_text:
        raise AssertionError("Embedding classification text should include user turns.")
    if "ASSISTANT: 入职流程包括账号开通" not in classification_text:
        raise AssertionError(
            "Embedding classification text should include assistant turns."
        )

    assert_route(
        router,
        "设备申请具体怎么做？",
        True,
        history=history,
        expected_route="embedding_rag",
        conversation_summary="The user is reviewing the employee onboarding process.",
    )
    assert_route(
        router,
        "谢谢你刚才的帮助",
        False,
        expected_route="embedding_direct",
    )


def assert_direct_task_matching_is_case_insensitive() -> None:
    router = IntentRouter(Settings(), embedder=None, llm_client=None)
    assert_route(
        router,
        "DRAFT an email for the onboarding team",
        False,
        expected_route="keyword_direct",
    )

    print("Case-insensitive direct task matching -> ok")


def assert_anchor_vectors_initialize_once_under_concurrency() -> None:
    embedder = CountingIntentEmbedder()
    router = IntentRouter(Settings(), embedder=embedder, llm_client=None)
    threads = [
        Thread(
            target=lambda: router.route(
                "设备申请具体怎么做？",
                [],
                "The user is reviewing employee onboarding.",
            )
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if embedder.anchor_encode_calls != 2:
        raise AssertionError(
            "Anchor vectors should initialize once for RAG and direct anchors."
        )

    print("Concurrent anchor initialization -> ok")


def assert_llm_fallback_behavior() -> None:
    rag_router = IntentRouter(
        Settings(),
        embedder=None,
        llm_client=FakeClassifierLLM(
            '{"use_rag": "true", "reason": "knowledge-base follow-up"}'
        ),
    )
    assert_route(
        rag_router,
        "那审批时限呢？",
        True,
        history=[Message("assistant", "We reviewed the vacation policy.")],
        expected_route="llm_rag",
    )

    direct_router = IntentRouter(
        Settings(),
        embedder=None,
        llm_client=FakeClassifierLLM(
            '{"use_rag": "false", "reason": "general chat"}'
        ),
    )
    assert_route(
        direct_router,
        "推荐一份晚餐菜单",
        False,
        expected_route="llm_direct",
    )


def main() -> None:
    router = IntentRouter(Settings(), embedder=None, llm_client=None)

    assert_route(
        router,
        "How should new employees request equipment?",
        True,
        expected_route="fallback_rag",
    )
    assert_route(
        router,
        "Hi, what is PostgreSQL?",
        True,
        expected_route="fallback_rag",
    )
    assert_route(
        router,
        "你好",
        False,
        expected_route="keyword_direct",
    )
    assert_route(
        router,
        "写一首关于 PostgreSQL 的诗",
        False,
        expected_route="keyword_direct",
    )
    assert_route(
        router,
        "PostgreSQL email notification pattern 怎么设计？",
        True,
        expected_route="fallback_rag",
    )
    assert_route(
        router,
        "不用知识库，直接回答：PostgreSQL 是什么？",
        False,
        expected_route="keyword_direct",
    )
    assert_route(
        router,
        "基于文档回答：SQLite 适合什么场景？",
        True,
        expected_route="keyword_rag",
    )
    assert_direct_task_matching_is_case_insensitive()
    assert_embedding_context_behavior()
    assert_anchor_vectors_initialize_once_under_concurrency()
    assert_llm_fallback_behavior()
    assert_pipeline_search_behavior()


if __name__ == "__main__":
    main()
