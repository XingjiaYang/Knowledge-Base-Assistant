from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Literal

from app.config import Settings, settings
from app.intent_router import IntentRouter
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
class RAGAnswer:
    answer: str
    contexts: list[SearchResult]
    conversation_summary: str
    compacted_history_messages: int
    used_rag: bool
    route: str
    route_reason: str


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
    ) -> RAGAnswer:
        clean_history = self._normalize_history(history or [])
        summary, recent_history, compacted_count = self._compact_history(
            clean_history,
            conversation_summary or "",
            question,
            top_k=top_k,
        )
        intent = self.intent_router.route(question, recent_history, summary)
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

        if intent.use_rag:
            search_query = self._build_search_query(question, recent_history)
            contexts = self._retrieve_contexts(
                search_query,
                top_k=top_k,
                recall_top_k=recall_top_k,
                bm25_top_k=bm25_top_k,
                rrf_top_k=rrf_top_k,
            )
            messages = self._build_rag_messages(
                question,
                contexts,
                recent_history,
                summary,
            )
        else:
            contexts = []
            messages = self._build_direct_messages(question, recent_history, summary)

        answer = self.llm_client.chat(messages)
        return RAGAnswer(
            answer=answer,
            contexts=contexts,
            conversation_summary=summary,
            compacted_history_messages=compacted_count,
            used_rag=intent.use_rag,
            route=intent.route,
            route_reason=intent.reason,
        )

    def _retrieve_contexts(
        self,
        search_query: str,
        top_k: int | None = None,
        recall_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_top_k: int | None = None,
    ) -> list[SearchResult]:
        final_top_k = self._final_top_k(top_k)
        bm25_limit = self._bm25_top_k(bm25_top_k)
        vector_limit = self._vector_top_k(recall_top_k)
        rrf_limit = self._rrf_top_k(rrf_top_k, top_k)
        if hasattr(self.vector_store, "hybrid_search"):
            recalled_contexts = self.vector_store.hybrid_search(
                search_query,
                bm25_top_k=bm25_limit,
                vector_top_k=vector_limit,
                rrf_top_k=rrf_limit,
            )
        else:
            recalled_contexts = self.vector_store.search(
                search_query,
                top_k=vector_limit,
            )[:rrf_limit]
        logger.info(
            "Hybrid recall returned %s contexts before reranking.",
            len(recalled_contexts),
        )

        if not self.config.reranker_enabled or not recalled_contexts:
            return recalled_contexts[:final_top_k]

        reranked_contexts = self.reranker.rerank(
            search_query,
            recalled_contexts,
            top_k=final_top_k,
        )
        logger.info(
            "Reranker returned %s contexts from %s recalled contexts.",
            len(reranked_contexts),
            len(recalled_contexts),
        )
        return reranked_contexts

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
