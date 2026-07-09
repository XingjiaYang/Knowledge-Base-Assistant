from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import re
import time
from typing import Literal

from app.config import Settings, settings
from app.intent_router import IntentDecision, IntentRouter
from app.llm_client import LLMClient
from app.prompt_budget import PromptBudget, TrimStrategy
from app.reranker import Reranker
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    used_rag: bool | None = None
    route: str = ""
    context_count: int = 0


@dataclass(frozen=True)
class RAGTimings:
    total_ms: float = 0.0
    history_ms: float = 0.0
    intent_ms: float = 0.0
    retrieval_ms: float = 0.0
    recall_ms: float = 0.0
    bm25_ms: float = 0.0
    vector_ms: float = 0.0
    embedding_ms: float = 0.0
    qdrant_ms: float = 0.0
    rrf_ms: float = 0.0
    reranker_ms: float = 0.0
    llm_ms: float = 0.0
    llm_ttft_ms: float | None = None
    llm_output_chars: int = 0
    llm_estimated_output_tokens: int = 0
    llm_estimated_tps: float = 0.0


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    contexts: list[SearchResult]
    conversation_summary: str
    compacted_history_messages: int
    used_rag: bool
    route: str
    route_reason: str
    retrieval_degraded: bool = False
    qdrant_degraded: bool = False
    reranker_degraded: bool = False
    degradation_reason: str = ""
    timings: RAGTimings = field(default_factory=RAGTimings)


@dataclass(frozen=True)
class RetrievalOutcome:
    contexts: list[SearchResult]
    retrieval_degraded: bool = False
    qdrant_degraded: bool = False
    reranker_degraded: bool = False
    degradation_reason: str = ""
    timings: RAGTimings = field(default_factory=RAGTimings)


@dataclass(frozen=True)
class RecallOutcome:
    contexts: list[SearchResult]
    recall_ms: float = 0.0
    bm25_ms: float = 0.0
    vector_ms: float = 0.0
    embedding_ms: float = 0.0
    qdrant_ms: float = 0.0
    rrf_ms: float = 0.0


class RAGPipeline:
    def __init__(
        self,
        config: Settings = settings,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        intent_router: IntentRouter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.budget = PromptBudget.from_config(config)
        self.vector_store = vector_store or VectorStore(config)
        self.llm_client = llm_client or LLMClient(config)
        self.intent_router = intent_router or IntentRouter(
            config,
            embedder=self._intent_embedder(self.vector_store),
            llm_client=self.llm_client,
        )
        self.reranker = reranker or Reranker(config)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        recall_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_top_k: int | None = None,
        history: list[ChatMessage] | None = None,
        conversation_summary: str | None = None,
        rag_only: bool = False,
    ) -> RAGAnswer:
        total_start = time.perf_counter()
        history_start = time.perf_counter()
        clean_history = self._normalize_history(history or [])
        summary, recent_history, compacted_count = self._compact_history(
            clean_history,
            conversation_summary or "",
            question,
            top_k=top_k,
        )
        history_ms = self._elapsed_ms(history_start)

        intent_start = time.perf_counter()
        if rag_only:
            intent = IntentDecision(
                True,
                "rag_only",
                "RAG-only mode enabled by the user.",
            )
        else:
            intent = self.intent_router.route(question, recent_history, summary)
        intent_ms = self._elapsed_ms(intent_start)
        logger.info(
            "RAG request routed: route=%s use_rag=%s bm25_top_k=%s "
            "vector_top_k=%s rrf_top_k=%s final_top_k=%s history=%s "
            "compacted=%s",
            intent.route,
            intent.use_rag,
            self._bm25_top_k(bm25_top_k),
            self._vector_top_k(recall_top_k),
            self._rrf_top_k(rrf_top_k, top_k),
            self._final_top_k(top_k),
            len(recent_history),
            compacted_count,
        )

        retrieval_outcome = RetrievalOutcome(contexts=[])
        if intent.use_rag:
            search_query = self._build_search_query(question, recent_history)
            retrieval_outcome = self._retrieve_contexts(
                search_query,
                top_k=top_k,
                recall_top_k=recall_top_k,
                bm25_top_k=bm25_top_k,
                rrf_top_k=rrf_top_k,
            )
            contexts = retrieval_outcome.contexts
            messages = self._build_rag_messages(
                question,
                contexts,
                recent_history,
                summary,
            )
        else:
            contexts = []
            messages = self._build_direct_messages(question, recent_history, summary)

        llm_start = time.perf_counter()
        answer = self.llm_client.chat(messages)
        llm_ms = self._elapsed_ms(llm_start)
        estimated_output_tokens = self._estimate_tokens(answer)
        llm_estimated_tps = (
            estimated_output_tokens / (llm_ms / 1000)
            if llm_ms > 0
            else 0.0
        )
        retrieval_timings = retrieval_outcome.timings
        timings = RAGTimings(
            total_ms=self._elapsed_ms(total_start),
            history_ms=history_ms,
            intent_ms=intent_ms,
            retrieval_ms=retrieval_timings.retrieval_ms,
            recall_ms=retrieval_timings.recall_ms,
            bm25_ms=retrieval_timings.bm25_ms,
            vector_ms=retrieval_timings.vector_ms,
            embedding_ms=retrieval_timings.embedding_ms,
            qdrant_ms=retrieval_timings.qdrant_ms,
            rrf_ms=retrieval_timings.rrf_ms,
            reranker_ms=retrieval_timings.reranker_ms,
            llm_ms=llm_ms,
            llm_ttft_ms=None,
            llm_output_chars=len(answer),
            llm_estimated_output_tokens=estimated_output_tokens,
            llm_estimated_tps=llm_estimated_tps,
        )
        return RAGAnswer(
            answer=answer,
            contexts=contexts,
            conversation_summary=summary,
            compacted_history_messages=compacted_count,
            used_rag=intent.use_rag,
            route=intent.route,
            route_reason=intent.reason,
            retrieval_degraded=retrieval_outcome.retrieval_degraded,
            qdrant_degraded=retrieval_outcome.qdrant_degraded,
            reranker_degraded=retrieval_outcome.reranker_degraded,
            degradation_reason=retrieval_outcome.degradation_reason,
            timings=timings,
        )

    def _retrieve_contexts(
        self,
        search_query: str,
        top_k: int | None = None,
        recall_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_top_k: int | None = None,
    ) -> RetrievalOutcome:
        retrieval_start = time.perf_counter()
        final_top_k = self._final_top_k(top_k)
        bm25_limit = self._bm25_top_k(bm25_top_k)
        vector_limit = self._vector_top_k(recall_top_k)
        rrf_limit = self._rrf_top_k(rrf_top_k, top_k)
        qdrant_degraded = False
        reranker_degraded = False
        degradation_reasons: list[str] = []

        recall_outcome = self._recall_contexts(
            search_query,
            bm25_limit=bm25_limit,
            vector_limit=vector_limit,
            rrf_limit=rrf_limit,
            degradation_reasons=degradation_reasons,
        )
        recalled_contexts = recall_outcome.contexts
        qdrant_degraded = bool(degradation_reasons)

        logger.info(
            "Hybrid recall returned %s contexts before reranking.",
            len(recalled_contexts),
        )

        if not self.config.reranker_enabled or not recalled_contexts:
            return RetrievalOutcome(
                contexts=recalled_contexts[:final_top_k],
                retrieval_degraded=qdrant_degraded,
                qdrant_degraded=qdrant_degraded,
                reranker_degraded=False,
                degradation_reason="; ".join(degradation_reasons),
                timings=self._retrieval_timings(
                    retrieval_start,
                    recall_outcome,
                    reranker_ms=0.0,
                ),
            )

        reranker_start = time.perf_counter()
        try:
            reranked_contexts = self.reranker.rerank(
                search_query,
                recalled_contexts,
                top_k=final_top_k,
            )
            reranker_ms = self._elapsed_ms(reranker_start)
        except Exception:
            reranker_ms = self._elapsed_ms(reranker_start)
            reranker_degraded = True
            degradation_reasons.append(
                "Reranker failed; using unre-ranked recall results."
            )
            logger.exception(
                "Retrieval degraded: retrieval_degraded=True "
                "qdrant_degraded=%s reranker_degraded=True "
                "fallback=rrf_or_bm25_top_k final_top_k=%s recalled_contexts=%s.",
                qdrant_degraded,
                final_top_k,
                len(recalled_contexts),
            )
            return RetrievalOutcome(
                contexts=recalled_contexts[:final_top_k],
                retrieval_degraded=True,
                qdrant_degraded=qdrant_degraded,
                reranker_degraded=reranker_degraded,
                degradation_reason="; ".join(degradation_reasons),
                timings=self._retrieval_timings(
                    retrieval_start,
                    recall_outcome,
                    reranker_ms=reranker_ms,
                ),
            )

        logger.info(
            "Reranker returned %s contexts from %s recalled contexts.",
            len(reranked_contexts),
            len(recalled_contexts),
        )
        return RetrievalOutcome(
            contexts=reranked_contexts,
            retrieval_degraded=qdrant_degraded,
            qdrant_degraded=qdrant_degraded,
            reranker_degraded=False,
            degradation_reason="; ".join(degradation_reasons),
            timings=self._retrieval_timings(
                retrieval_start,
                recall_outcome,
                reranker_ms=reranker_ms,
            ),
        )

    def _recall_contexts(
        self,
        search_query: str,
        bm25_limit: int,
        vector_limit: int,
        rrf_limit: int,
        degradation_reasons: list[str],
    ) -> RecallOutcome:
        recall_start = time.perf_counter()
        if hasattr(self.vector_store, "search_bm25") and hasattr(
            self.vector_store,
            "search",
        ):
            def run_bm25() -> tuple[list[SearchResult], float]:
                bm25_start = time.perf_counter()
                contexts = self.vector_store.search_bm25(
                    search_query,
                    top_k=bm25_limit,
                )
                return contexts, self._elapsed_ms(bm25_start)

            def run_vector() -> tuple[
                list[SearchResult],
                float,
                float,
                float,
            ]:
                vector_start = time.perf_counter()
                if hasattr(self.vector_store, "search_with_timing"):
                    vector_outcome = self.vector_store.search_with_timing(
                        search_query,
                        top_k=vector_limit,
                    )
                    return (
                        vector_outcome.results,
                        vector_outcome.total_ms,
                        vector_outcome.embedding_ms,
                        vector_outcome.qdrant_ms,
                    )

                return (
                    self.vector_store.search(
                        search_query,
                        top_k=vector_limit,
                    ),
                    self._elapsed_ms(vector_start),
                    0.0,
                    0.0,
                )

            vector_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                bm25_future = executor.submit(run_bm25)
                vector_future = executor.submit(run_vector)
                try:
                    (
                        vector_contexts,
                        vector_ms,
                        embedding_ms,
                        qdrant_ms,
                    ) = vector_future.result()
                except Exception:
                    vector_ms = self._elapsed_ms(vector_start)
                    bm25_contexts, bm25_ms = bm25_future.result()
                    degradation_reasons.append(
                        "Qdrant/vector recall failed; using BM25-only retrieval."
                    )
                    logger.exception(
                        "Retrieval degraded: retrieval_degraded=True "
                        "qdrant_degraded=True reranker_degraded=False "
                        "fallback=bm25_only bm25_contexts=%s.",
                        len(bm25_contexts),
                    )
                    return RecallOutcome(
                        contexts=bm25_contexts,
                        recall_ms=self._elapsed_ms(recall_start),
                        bm25_ms=bm25_ms,
                        vector_ms=vector_ms,
                        embedding_ms=0.0,
                        qdrant_ms=0.0,
                        rrf_ms=0.0,
                    )

                bm25_contexts, bm25_ms = bm25_future.result()

            try:
                rrf_start = time.perf_counter()
                fused_contexts = VectorStore._rrf_fuse(
                    vector_contexts,
                    bm25_contexts,
                    top_k=rrf_limit,
                )
                rrf_ms = self._elapsed_ms(rrf_start)
            except Exception:
                logger.exception("RRF fusion failed.")
                raise
            logger.info(
                "Hybrid recall fused bm25=%s vector=%s into rrf=%s contexts.",
                len(bm25_contexts),
                len(vector_contexts),
                len(fused_contexts),
            )
            return RecallOutcome(
                contexts=fused_contexts,
                recall_ms=self._elapsed_ms(recall_start),
                bm25_ms=bm25_ms,
                vector_ms=vector_ms,
                embedding_ms=embedding_ms,
                qdrant_ms=qdrant_ms,
                rrf_ms=rrf_ms,
            )

        if hasattr(self.vector_store, "hybrid_search"):
            try:
                hybrid_start = time.perf_counter()
                contexts = self.vector_store.hybrid_search(
                    search_query,
                    bm25_top_k=bm25_limit,
                    vector_top_k=vector_limit,
                    rrf_top_k=rrf_limit,
                )
                return RecallOutcome(
                    contexts=contexts,
                    recall_ms=self._elapsed_ms(hybrid_start),
                )
            except Exception:
                if not hasattr(self.vector_store, "search_bm25"):
                    raise
                bm25_start = time.perf_counter()
                bm25_contexts = self.vector_store.search_bm25(
                    search_query,
                    top_k=bm25_limit,
                )
                bm25_ms = self._elapsed_ms(bm25_start)
                degradation_reasons.append(
                    "Qdrant/vector recall failed; using BM25-only retrieval."
                )
                logger.exception(
                    "Retrieval degraded: retrieval_degraded=True "
                    "qdrant_degraded=True reranker_degraded=False "
                    "fallback=bm25_only bm25_contexts=%s.",
                    len(bm25_contexts),
                )
                return RecallOutcome(
                    contexts=bm25_contexts,
                    recall_ms=self._elapsed_ms(recall_start),
                    bm25_ms=bm25_ms,
                )

        vector_start = time.perf_counter()
        contexts = self.vector_store.search(search_query, top_k=vector_limit)[:rrf_limit]
        return RecallOutcome(
            contexts=contexts,
            recall_ms=self._elapsed_ms(recall_start),
            vector_ms=self._elapsed_ms(vector_start),
        )

    def _retrieval_timings(
        self,
        retrieval_start: float,
        recall_outcome: RecallOutcome,
        reranker_ms: float,
    ) -> RAGTimings:
        return RAGTimings(
            retrieval_ms=self._elapsed_ms(retrieval_start),
            recall_ms=recall_outcome.recall_ms,
            bm25_ms=recall_outcome.bm25_ms,
            vector_ms=recall_outcome.vector_ms,
            embedding_ms=recall_outcome.embedding_ms,
            qdrant_ms=recall_outcome.qdrant_ms,
            rrf_ms=recall_outcome.rrf_ms,
            reranker_ms=reranker_ms,
        )

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    def _final_top_k(self, top_k: int | None = None) -> int:
        return max(1, top_k or self.config.retrieve_top_k)

    def _bm25_top_k(self, bm25_top_k: int | None = None) -> int:
        return max(1, bm25_top_k or self.config.bm25_top_k)

    def _vector_top_k(self, recall_top_k: int | None = None) -> int:
        return max(1, recall_top_k or self.config.recall_top_k)

    def _rrf_top_k(
        self,
        rrf_top_k: int | None = None,
        top_k: int | None = None,
    ) -> int:
        final_top_k = self._final_top_k(top_k)
        fused_limit = max(1, rrf_top_k or self.config.rrf_top_k)
        return max(fused_limit, final_top_k)

    def _normalize_history(self, history: list[ChatMessage]) -> list[ChatMessage]:
        max_messages = max(0, self.config.history_max_messages)
        if max_messages:
            history = history[-max_messages:]

        clean: list[ChatMessage] = []
        for message in history:
            content = message.content.strip()
            if not content or message.role not in {"user", "assistant"}:
                continue
            clean.append(
                ChatMessage(
                    role=message.role,
                    content=self.budget.trim_message(content),
                    used_rag=getattr(message, "used_rag", None),
                    route=getattr(message, "route", ""),
                    context_count=max(
                        0,
                        int(getattr(message, "context_count", 0) or 0),
                    ),
                )
            )
        return clean

    def _compact_history(
        self,
        history: list[ChatMessage],
        conversation_summary: str,
        question: str = "",
        top_k: int | None = None,
    ) -> tuple[str, list[ChatMessage], int]:
        recent_limit = max(0, self.config.history_recent_turns * 2)
        summary = self._trim_summary(conversation_summary)
        if not recent_limit or not history:
            return summary, history, 0
        if not self._should_compact_history(history, summary, question, top_k):
            return summary, history, 0

        older_history = history[:-recent_limit]
        recent_history = history[-recent_limit:]
        if not older_history:
            return summary, history, 0

        summary = self._summarize_history(summary, older_history)
        return summary, recent_history, len(older_history)

    def _should_compact_history(
        self,
        history: list[ChatMessage],
        conversation_summary: str,
        question: str,
        top_k: int | None,
    ) -> bool:
        runtime_settings = self._runtime_settings()
        context_limit = max(1, runtime_settings.llm_context_max_tokens)
        output_reserve = max(0, runtime_settings.llm_max_tokens)
        reserved_tokens = (
            max(0, self.config.llm_context_safety_margin_tokens)
            + max(0, self.config.llm_context_prompt_overhead_tokens)
            + output_reserve
            + self._expected_reference_tokens(top_k)
            + self._estimate_tokens(question)
        )
        available_tokens = context_limit - reserved_tokens
        if available_tokens <= 0:
            return True

        history_tokens = self._estimate_tokens(conversation_summary)
        history_tokens += self._estimate_tokens(self._format_history(history))
        should_compact = history_tokens > available_tokens
        logger.info(
            "History compaction budget: history_tokens=%s available_tokens=%s "
            "context_limit=%s reserved_tokens=%s compact=%s",
            history_tokens,
            available_tokens,
            context_limit,
            reserved_tokens,
            should_compact,
        )
        return should_compact

    def _runtime_settings(self) -> object:
        if hasattr(self.llm_client, "runtime_settings"):
            return self.llm_client.runtime_settings()
        return self.config

    def _expected_reference_tokens(self, top_k: int | None) -> int:
        per_context_tokens = max(1, self.config.chunk_size + 256)
        return self._final_top_k(top_k) * per_context_tokens

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        cjk_chars = len(_CJK_CHAR_RE.findall(text))
        non_cjk_chars = max(0, len(text) - cjk_chars)
        return cjk_chars + math.ceil(non_cjk_chars / 4)

    def _summarize_history(
        self,
        conversation_summary: str,
        older_history: list[ChatMessage],
    ) -> str:
        if not older_history:
            return self._trim_summary(conversation_summary)

        existing_summary = self._trim_summary(conversation_summary)
        history_text = self._format_history(
            older_history,
            max_chars=self.budget.summary_history_max_chars,
            strategy="middle",
        )
        summary_prompt = (
            "Existing compact summary:\n"
            f"{existing_summary or 'None'}\n\n"
            "New conversation turns to merge:\n"
            f"{history_text}\n\n"
            "Write an updated compact memory for future answers. Keep durable "
            "user goals, constraints, decisions, technical facts, and unresolved "
            "questions. Omit greetings and repeated wording."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You compact chat history for Knowledge Base Assistant. "
                    "Return only the compact memory."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ]

        try:
            summary = self.llm_client.chat(
                messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=min(
                    self.budget.summary_max_tokens,
                    self.config.llm_max_tokens,
                ),
            )
        except Exception:
            logger.exception("Conversation summarization failed; using fallback.")
            fallback = "\n".join(
                part
                for part in [existing_summary, history_text]
                if part
            )
            return self._trim_summary(fallback)

        return self._trim_summary(summary)

    def _trim_summary(self, summary: str) -> str:
        return self.budget.trim_summary(summary)

    def _format_history(
        self,
        history: list[ChatMessage],
        max_chars: int | None = None,
        strategy: TrimStrategy = "middle",
    ) -> str:
        return self.budget.format_history(history, max_chars, strategy=strategy)

    def _build_search_query(
        self,
        question: str,
        recent_history: list[ChatMessage],
    ) -> str:
        recent_user_messages = [
            message.content
            for message in recent_history
            if message.role == "user"
        ][-2:]
        query = "\n".join([*recent_user_messages, question]).strip()
        if not query:
            return question
        return self.budget.trim_text(
            query,
            self.budget.search_query_max_chars,
            strategy="tail",
        )

    def _build_rag_messages(
        self,
        question: str,
        contexts: list[SearchResult],
        recent_history: list[ChatMessage],
        conversation_summary: str,
    ) -> list[dict[str, str]]:
        context_text = "\n\n".join(
            (
                f"[{idx}] source={item.source} chunk={item.chunk_id} "
                f"score={item.score:.4f} type={item.content_type} "
                f"headings={self._format_context_headings(item)}\n{item.text}"
            )
            for idx, item in enumerate(contexts, start=1)
        )
        if not context_text:
            context_text = "No retrieved context."

        system_prompt = (
            "You are Knowledge Base Assistant. Use the retrieved context when it "
            "is relevant, combine it with the conversation history, and avoid "
            "inventing details when evidence is insufficient. Give complete "
            "answers when the user asks for depth."
        )
        user_prompt = (
            "Retrieved context:\n"
            f"{context_text}\n\n"
            "Current question:\n"
            f"{question}\n\n"
            "Answer in the same language as the current question when possible. "
            "When retrieved context supports a claim, mention the relevant "
            "source names naturally."
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if conversation_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Compact conversation memory:\n"
                        f"{conversation_summary}"
                    ),
                }
            )

        messages.extend(
            {"role": message.role, "content": message.content}
            for message in recent_history
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _format_context_headings(self, item: SearchResult) -> str:
        if item.headings:
            return " > ".join(item.headings)
        headings = [item.h1, item.h2, item.h3]
        return " > ".join(heading for heading in headings if heading) or "None"

    def _build_direct_messages(
        self,
        question: str,
        recent_history: list[ChatMessage],
        conversation_summary: str,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are Knowledge Base Assistant. This request was routed to direct "
            "chat, so no knowledge-base retrieval was used. Answer from the "
            "conversation context and your general knowledge, and do not claim "
            "that local documents support the answer."
        )
        user_prompt = (
            "Current question:\n"
            f"{question}\n\n"
            "Answer in the same language as the current question when possible."
        )
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Compact conversation memory:\n"
                        f"{conversation_summary}"
                    ),
                }
            )

        messages.extend(
            {"role": message.role, "content": message.content}
            for message in recent_history
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _intent_embedder(vector_store: object) -> object | None:
        if hasattr(vector_store, "encode"):
            return vector_store
        return getattr(vector_store, "model", None)
