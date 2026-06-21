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
    used_rag: bool | None = None
    route: str = ""
    context_count: int = 0


class FakeVectorStore:
    model = None

    def __init__(self) -> None:
        self.search_calls = 0
        self.search_top_ks: list[int | None] = []
        self.bm25_top_ks: list[int | None] = []
        self.rrf_top_ks: list[int | None] = []

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.search_calls += 1
        self.search_top_ks.append(top_k)
        return self._results()

    def hybrid_search(
        self,
        query: str,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        rrf_top_k: int | None = None,
    ) -> list[SearchResult]:
        self.search_calls += 1
        self.bm25_top_ks.append(bm25_top_k)
        self.search_top_ks.append(vector_top_k)
        self.rrf_top_ks.append(rrf_top_k)
        return self._results()

    def _results(self) -> list[SearchResult]:
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


class QdrantFailingVectorStore(FakeVectorStore):
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        raise RuntimeError("qdrant unavailable")

    def search_bm25(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        self.bm25_top_ks.append(top_k)
        return [
            SearchResult(
                text="BM25-only fallback context A.",
                source="data/docs/fallback.md",
                chunk_id=10,
                score=3.0,
                bm25_score=3.0,
                retrieval_source="bm25",
            ),
            SearchResult(
                text="BM25-only fallback context B.",
                source="data/docs/fallback.md",
                chunk_id=11,
                score=2.0,
                bm25_score=2.0,
                retrieval_source="bm25",
            ),
        ][: top_k or 2]


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


class FailingReranker(FakeReranker):
    def rerank(
        self,
        query: str,
        contexts: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, len(contexts), top_k))
        raise RuntimeError("reranker unavailable")


class FakeClassifierLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []
        self.max_tokens: list[int | None] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
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
            "general chat",
            "write or rewrite",
            "without documents",
            "translate this text",
            "programming question",
            "greet the user",
            "闲聊",
            "直接聊天",
            "翻译这句话",
            "文案或邮件",
            "不要检索",
            "你好",
            "谢谢",
        )
        if any(marker in lowered for marker in direct_markers):
            return [0.0, 1.0]

        rag_markers = (
            "local knowledge base",
            "indexed documents",
            "document evidence",
            "local corpus",
            "local docs",
            "知识库",
            "本地文档",
            "引用资料",
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
    return decision


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

    rag_answer = pipeline.answer(
        "新员工如何申请开发设备？",
        top_k=2,
        bm25_top_k=6,
        recall_top_k=7,
        rrf_top_k=5,
    )
    if not rag_answer.used_rag or vector_store.search_calls != 1:
        raise AssertionError("Knowledge-base route should call vector search once.")
    if vector_store.search_top_ks != [7]:
        raise AssertionError("Vector recall should use recall_top_k.")
    if vector_store.bm25_top_ks != [6]:
        raise AssertionError("BM25 recall should use bm25_top_k.")
    if vector_store.rrf_top_ks != [5]:
        raise AssertionError("RRF fusion should use rrf_top_k.")
    if reranker.calls != [("新员工如何申请开发设备？", 3, 2)]:
        raise AssertionError(f"Reranker should receive recalled contexts: {reranker.calls}")
    if [context.chunk_id for context in rag_answer.contexts] != [2, 1]:
        raise AssertionError("RAG answer should keep reranked final top_k contexts.")

    print("Pipeline search behavior -> ok")


def assert_pipeline_retrieval_degradation_behavior() -> None:
    config = Settings()
    llm_client = FakeLLMClient()
    qdrant_config = Settings(reranker_enabled=False)
    qdrant_router = IntentRouter(qdrant_config, embedder=None, llm_client=None)

    qdrant_answer = RAGPipeline(
        qdrant_config,
        vector_store=QdrantFailingVectorStore(),
        llm_client=llm_client,
        intent_router=qdrant_router,
        reranker=FakeReranker(),
    ).answer("新员工如何申请开发设备？", top_k=1, bm25_top_k=4)
    if not qdrant_answer.retrieval_degraded or not qdrant_answer.qdrant_degraded:
        raise AssertionError("Qdrant failure should mark retrieval as degraded.")
    if qdrant_answer.reranker_degraded:
        raise AssertionError("Working reranker should not be marked degraded.")
    if [context.chunk_id for context in qdrant_answer.contexts] != [10]:
        raise AssertionError("Qdrant fallback should use BM25-only contexts.")

    vector_store = FakeVectorStore()
    failing_reranker = FailingReranker()
    router = IntentRouter(config, embedder=None, llm_client=None)
    reranker_answer = RAGPipeline(
        config,
        vector_store=vector_store,
        llm_client=llm_client,
        intent_router=router,
        reranker=failing_reranker,
    ).answer("新员工如何申请开发设备？", top_k=2, rrf_top_k=3)
    if not reranker_answer.retrieval_degraded or not reranker_answer.reranker_degraded:
        raise AssertionError("Reranker failure should mark retrieval as degraded.")
    if reranker_answer.qdrant_degraded:
        raise AssertionError("Healthy Qdrant recall should not be marked degraded.")
    if [context.chunk_id for context in reranker_answer.contexts] != [0, 1]:
        raise AssertionError("Reranker fallback should keep coarse recall top K.")
    if failing_reranker.calls != [("新员工如何申请开发设备？", 3, 2)]:
        raise AssertionError("Reranker should receive coarse recall before fallback.")

    print("Retrieval degradation behavior -> ok")


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


def assert_classification_text_prioritizes_current_question() -> None:
    router = IntentRouter(
        Settings(
            intent_embedding_history_max_chars=120,
            intent_embedding_summary_max_chars=120,
            intent_embedding_text_max_chars=180,
        )
    )
    text = router._classification_text(
        "What is the current routing question?",
        [
            Message("user", "Earlier user context " + ("x" * 200)),
            Message("assistant", "Earlier assistant context " + ("y" * 200)),
        ],
        "Long compact summary " + ("z" * 400),
    )

    if not text.startswith("Current question:\nWhat is the current routing question?"):
        raise AssertionError("Intent embedding text should prioritize current question.")
    if len(text) > 180:
        raise AssertionError("Intent embedding text should respect total budget.")

    print("Classification text current-question priority -> ok")


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


def assert_stateful_referential_routing() -> None:
    router = IntentRouter(Settings(), embedder=None, llm_client=None)
    rag_history = [
        Message("user", "先讲讲巨大历史机遇这个梗。"),
        Message(
            "assistant",
            "它来自家是本门店标语和后续社媒传播。",
            used_rag=True,
            route="keyword_rag",
            context_count=2,
        ),
    ]
    decision = assert_route(
        router,
        "后续呢？",
        True,
        history=rag_history,
        expected_route="state_rag",
    )
    if "previous answer used retrieval" not in decision.reason:
        raise AssertionError("State route should explain previous retrieval.")

    long_new_topic = router.route(
        "这个方案和 PostgreSQL connection pooling 有什么关系？",
        rag_history,
        "",
    )
    if long_new_topic.route == "state_rag":
        raise AssertionError("New technical topic should not be forced by state.")

    direct_history = [
        Message("user", "帮我翻译一句话。"),
        Message(
            "assistant",
            "I will arrive tomorrow.",
            used_rag=False,
            route="keyword_direct",
            context_count=0,
        ),
    ]
    direct_followup = router.route("那这个呢？", direct_history, "")
    if direct_followup.route == "state_rag":
        raise AssertionError("Previous direct answer should not force RAG.")

    no_context_history = [
        Message("user", "查一下这个主题。"),
        Message(
            "assistant",
            "没有找到相关资料。",
            used_rag=True,
            route="fallback_rag",
            context_count=0,
        ),
    ]
    no_context_followup = router.route("继续讲", no_context_history, "")
    if no_context_followup.route == "state_rag":
        raise AssertionError("Empty previous retrieval should not force RAG.")

    print("Stateful referential routing -> ok")


def assert_llm_fallback_behavior() -> None:
    classifier = FakeClassifierLLM(
        (
            "<think>The question is an ambiguous follow-up to documented policy."
            "</think><answer>{\"use_rag\": \"true\", "
            "\"reason\": \"knowledge-base follow-up\"}</answer>"
        )
    )
    rag_router = IntentRouter(
        Settings(),
        embedder=None,
        llm_client=classifier,
    )
    decision = assert_route(
        rag_router,
        "那审批时限呢？",
        True,
        history=[
            Message("user", "We reviewed the vacation policy."),
            Message(
                "assistant",
                "The policy requires manager approval.",
                used_rag=True,
                route="embedding_rag",
                context_count=2,
            ),
        ],
        expected_route="llm_rag",
    )
    if decision.reason != "knowledge-base follow-up":
        raise AssertionError("LLM route reason should come from <answer> JSON.")
    system_prompt = classifier.calls[-1][0]["content"]
    fallback_prompt = classifier.calls[-1][-1]["content"]
    if "<think>" not in system_prompt or "<answer>" not in system_prompt:
        raise AssertionError("LLM classifier should allow tagged judgement output.")
    if "THINK_AND_JUDGEMENT" not in fallback_prompt or "JSON_ANS" not in fallback_prompt:
        raise AssertionError("LLM fallback prompt should specify tagged output format.")
    if "Previous route state:" not in fallback_prompt:
        raise AssertionError("LLM fallback should receive previous route state.")
    if "used_rag=true" not in fallback_prompt or "route=embedding_rag" not in fallback_prompt:
        raise AssertionError("LLM fallback should include structured route metadata.")
    if classifier.max_tokens[-1] != 512:
        raise AssertionError("LLM classifier output budget should allow tagged output.")
    if fallback_prompt.find("Current question:") > fallback_prompt.find("Compact memory:"):
        raise AssertionError("LLM fallback prompt should prioritize current question.")
    if "家是本" not in fallback_prompt or "朱剑秋" not in fallback_prompt:
        raise AssertionError("LLM fallback prompt should describe the local corpus.")
    if "巨大历史机遇" not in fallback_prompt or "巨大历史鲫鱼" not in fallback_prompt:
        raise AssertionError("LLM fallback prompt should include the new meme topic.")
    if "SQL/database questions" not in fallback_prompt:
        raise AssertionError(
            "LLM fallback prompt should route unrelated SQL/database questions direct."
        )

    direct_router = IntentRouter(
        Settings(),
        embedder=None,
        llm_client=FakeClassifierLLM(
            '<think>Unrelated request.</think><answer>{"use_rag": "false", '
            '"reason": "general chat"}</answer>'
        ),
    )
    assert_route(
        direct_router,
        "推荐一份晚餐菜单",
        False,
        expected_route="llm_direct",
    )

    slash_tag_router = IntentRouter(
        Settings(),
        embedder=None,
        llm_client=FakeClassifierLLM(
            '</think>Unrelated request.</think></answer>{"use_rag": false, '
            '"reason": "slash-style answer tag"}</answer>'
        ),
    )
    assert_route(
        slash_tag_router,
        "今晚吃什么？",
        False,
        expected_route="llm_direct",
    )

    legacy_json_router = IntentRouter(
        Settings(),
        embedder=None,
        llm_client=FakeClassifierLLM(
            '{"use_rag": true, "reason": "legacy bare JSON"}'
        ),
    )
    decision = assert_route(
        legacy_json_router,
        "审批材料呢？",
        True,
        expected_route="llm_rag",
    )
    if decision.reason != "legacy bare JSON":
        raise AssertionError("LLM classifier should still accept bare JSON output.")


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
        False,
        expected_route="keyword_direct",
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
        False,
        expected_route="keyword_direct",
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
    assert_route(
        router,
        "朱剑秋和勇哥连线发生了什么？",
        True,
        expected_route="keyword_rag",
    )
    assert_route(
        router,
        "家是本菜单价格如何？",
        True,
        expected_route="keyword_rag",
    )
    assert_route(
        router,
        "巨大历史机遇和巨大历史鲫鱼是什么梗？",
        True,
        expected_route="keyword_rag",
    )
    assert_route(
        router,
        "春熙路门店照片里的标语为什么会火？",
        True,
        expected_route="keyword_rag",
    )
    assert_route(
        router,
        "写一首关于家是本的诗",
        False,
        expected_route="keyword_direct",
    )
    assert_route(
        router,
        "不用知识库，直接回答：朱剑秋是谁？",
        False,
        expected_route="keyword_direct",
    )
    assert_direct_task_matching_is_case_insensitive()
    assert_embedding_context_behavior()
    assert_classification_text_prioritizes_current_question()
    assert_anchor_vectors_initialize_once_under_concurrency()
    assert_stateful_referential_routing()
    assert_llm_fallback_behavior()
    assert_pipeline_search_behavior()
    assert_pipeline_retrieval_degradation_behavior()


if __name__ == "__main__":
    main()
