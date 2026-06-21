from __future__ import annotations

from dataclasses import dataclass
import json
import re
from threading import Lock
from typing import Literal, Protocol, Sequence

import numpy as np

from app.config import Settings, settings
from app.llm_client import LLMClient
from app.prompt_budget import PromptBudget, TrimStrategy


RouteName = Literal[
    "disabled",
    "keyword_rag",
    "keyword_direct",
    "state_rag",
    "embedding_rag",
    "embedding_direct",
    "llm_rag",
    "llm_direct",
    "fallback_rag",
    "fallback_direct",
]


class HistoryMessage(Protocol):
    role: str
    content: str


@dataclass(frozen=True)
class PreviousRouteState:
    used_rag: bool | None
    route: str
    had_contexts: bool
    previous_user_question: str
    previous_assistant_excerpt: str


@dataclass(frozen=True)
class IntentDecision:
    use_rag: bool
    route: RouteName
    reason: str


class IntentRouter:
    FORCE_RAG_PHRASES = (
        "use rag",
        "use retrieval",
        "search the docs",
        "search docs",
        "use the docs",
        "according to the docs",
        "according to documentation",
        "based on the knowledge base",
        "based on local docs",
        "from the knowledge base",
        "with references",
        "cite sources",
        "基于文档",
        "根据文档",
        "根据知识库",
        "查知识库",
        "搜索知识库",
        "检索知识库",
        "使用检索",
        "引用资料",
        "带引用",
    )

    FORCE_DIRECT_PHRASES = (
        "do not use rag",
        "don't use rag",
        "without rag",
        "no rag",
        "do not retrieve",
        "don't retrieve",
        "without retrieval",
        "no retrieval",
        "answer directly",
        "不要用rag",
        "不用rag",
        "不要检索",
        "不用检索",
        "不要查库",
        "不要查知识库",
        "不用知识库",
        "直接回答",
        "别查",
    )

    DOMAIN_RAG_PHRASES = (
        "家是本",
        "家是本餐饮",
        "家是本之歌",
        "家是本，本是家",
        "朱剑秋",
        "朱老板",
        "勇哥连线",
        "勇哥",
        "老妈蹄花",
        "葱油饼",
        "巨大历史机遇",
        "巨大历史鲫鱼",
        "历史鲫鱼",
        "历史机遇体验店",
        "春熙路神秘小吃店",
        "三年千店",
        "千店扩张",
        "1000家",
        "豆包ai",
        "豆包视频",
        "b站评论",
        "b站账号",
        "b站弹幕",
        "哔哩哔哩评论",
        "哔哩哔哩账号",
        "哔哩哔哩弹幕",
        "bilibili comments",
        "giant historical opportunity",
        "huge historical opportunity",
        "giant historical crucian carp",
        "historical crucian carp",
        "jiashiben",
        "jia shi ben",
        "zhu jianqiu",
        "yongge",
        "yong ge",
        "laoma tihua",
        "cong you bing",
    )

    DOMAIN_RAG_PATTERNS = (
        (
            r"(家是本|朱剑秋|朱老板|勇哥|巨大历史机遇|巨大历史鲫鱼|历史机遇|历史鲫鱼).{0,40}"
            r"(菜单|定价|价格|财务|营业额|亏损|盈利|客流|租金|选址|春熙路|"
            r"口号|家文化|商标|品牌|妻子|老婆|评论|差评|弹幕|黑粉|引流|"
            r"豆包|ai|视频|连线|千店|1000家|扩张|标语|门店照片|红色大字|"
            r"梗|谐音|二创|历史机遇|历史鲫鱼)"
        ),
        (
            r"(菜单|定价|价格|财务|营业额|亏损|盈利|客流|租金|选址|春熙路|"
            r"口号|家文化|商标|品牌|妻子|老婆|评论|差评|弹幕|黑粉|引流|"
            r"豆包|ai|视频|连线|千店|1000家|扩张|标语|门店照片|红色大字|"
            r"梗|谐音|二创|历史机遇|历史鲫鱼).{0,40}"
            r"(家是本|朱剑秋|朱老板|勇哥|巨大历史机遇|巨大历史鲫鱼|历史机遇|历史鲫鱼)"
        ),
        (
            r"(春熙路|门店照片|小吃店|红色大字|标语).{0,40}"
            r"(巨大|历史机遇|历史鲫鱼|谐音|玩梗|二创|标语|口号|家是本|朱剑秋)"
        ),
        (
            r"(巨大|历史机遇|历史鲫鱼|谐音|玩梗|二创|标语|口号|家是本|朱剑秋).{0,40}"
            r"(春熙路|门店照片|小吃店|红色大字|标语)"
        ),
        (
            r"(jiashiben|jia shi ben|zhu jianqiu|yongge|yong ge|"
            r"giant historical opportunity|huge historical opportunity|"
            r"giant historical crucian carp|historical crucian carp).{0,80}"
            r"(menu|price|pricing|finance|revenue|rent|comments|reviews|"
            r"bilibili|doubao|live|incident|expansion|1000 stores|meme|"
            r"slogan|wall slogan|viral phrase)"
        ),
    )

    DIRECT_TASK_PATTERNS = (
        (
            r"\b(write|draft|compose|create|generate)\b.{0,80}"
            r"\b(poem|joke|email|cover letter|resume|linkedin)\b"
        ),
        r"\b(tell|make)\b.{0,40}\bjoke\b",
        r"\btranslate\b",
        r"\bsummarize my resume\b",
        r"^\s*(hi|hello|hey|thanks|thank you)[!.?\s]*$",
        r"(写|创作|生成).{0,40}(诗|邮件|简历)",
        r"讲.{0,20}笑话",
        r"翻译",
        r"简历",
        r"^\s*(你好|您好|谢谢|感谢)[！!。.\s]*$",
    )

    RAG_ANCHORS = (
        "Answer using the local knowledge base.",
        "Find facts in the indexed documents.",
        "Use document evidence for this question.",
        "Compare details from the local corpus.",
        "Troubleshoot using the local docs.",
        "根据知识库回答这个问题。",
        "查本地文档里的事实。",
        "引用资料回答这个问题。",
    )

    DIRECT_ANCHORS = (
        "Answer this as general chat.",
        "Write or rewrite text without documents.",
        "Translate this text.",
        "Answer a programming question from general knowledge.",
        "Greet the user conversationally.",
        "直接聊天回答。",
        "翻译这句话。",
        "写一段文案或邮件。",
        "不要检索资料。",
    )

    STRONG_REFERENTIAL_PHRASES = (
        "这个呢",
        "那个呢",
        "它呢",
        "他们呢",
        "后续呢",
        "然后呢",
        "继续",
        "继续讲",
        "继续说",
        "展开",
        "展开讲",
        "详细说说",
        "再展开",
        "接着说",
        "刚才那个呢",
        "上面那个呢",
        "前面那个呢",
        "这个为什么",
        "为什么会这样",
        "这是什么意思",
        "这怎么来的",
        "这个怎么来的",
        "这有什么风险",
        "这个有什么风险",
        "这个问题呢",
        "这个计划呢",
        "这个事件呢",
        "what about that",
        "what about it",
        "how about that",
        "and then",
        "what happened next",
        "continue",
        "go on",
        "elaborate",
        "tell me more",
        "why is that",
        "what does that mean",
    )

    STRONG_REFERENTIAL_PATTERNS = (
        r"^(那|那么|所以)?(这个|那个|它|他们|上述|上面|前面|刚才).{0,12}"
        r"(呢|吗|为什么|怎么|如何|啥意思|什么意思|后续|风险|问题)[？?。!！\s]*$",
        r"^(继续|接着|展开|详细).{0,8}(讲|说|解释|分析)?[。！？?\s]*$",
        r"^(后续|之后|然后).{0,12}(呢|是什么|有哪些|怎么|如何|节点)?[？?。!！\s]*$",
    )

    NEW_TOPIC_TECH_TERMS = (
        "postgresql",
        "postgres",
        "mysql",
        "sqlite",
        "duckdb",
        "clickhouse",
        "python",
        "fastapi",
        "javascript",
        "typescript",
        "redis",
        "kafka",
        "flink",
        "spark",
        "sql",
        "api",
        "database",
        "db",
    )

    def __init__(
        self,
        config: Settings = settings,
        embedder: object | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.budget = PromptBudget.from_config(config)
        self.embedder = embedder
        self.llm_client = llm_client
        self._rag_anchor_vectors: np.ndarray | None = None
        self._direct_anchor_vectors: np.ndarray | None = None
        self._anchor_lock = Lock()

    def route(
        self,
        question: str,
        recent_history: Sequence[HistoryMessage],
        conversation_summary: str,
    ) -> IntentDecision:
        if not self.config.intent_router_enabled:
            return IntentDecision(True, "disabled", "Intent router disabled.")

        question = question.strip()
        keyword_decision = self._route_with_keywords(question, recent_history)
        if keyword_decision is not None:
            return keyword_decision

        embedding_decision = self._route_with_embeddings(
            question,
            recent_history,
            conversation_summary,
        )
        if embedding_decision is not None:
            return embedding_decision

        if self.config.intent_llm_fallback and self.llm_client is not None:
            llm_decision = self._route_with_llm(
                question,
                recent_history,
                conversation_summary,
            )
            if llm_decision is not None:
                return llm_decision

        return IntentDecision(
            True,
            "fallback_rag",
            "No clear direct-chat intent matched; defaulting to knowledge-base retrieval.",
        )

    def _route_with_keywords(
        self,
        question: str,
        recent_history: Sequence[HistoryMessage],
    ) -> IntentDecision | None:
        text = question.lower()

        if self._contains_any(text, self.FORCE_DIRECT_PHRASES):
            return IntentDecision(
                False,
                "keyword_direct",
                "User explicitly asked to avoid retrieval.",
            )

        if self._contains_any(text, self.FORCE_RAG_PHRASES):
            return IntentDecision(
                True,
                "keyword_rag",
                "User explicitly asked to use local documents.",
            )

        if self._matches_direct_task(text):
            return IntentDecision(
                False,
                "keyword_direct",
                "Question is a clear direct-chat task.",
            )

        if self._matches_domain_rag(text):
            return IntentDecision(
                True,
                "keyword_rag",
                "Question mentions entities or topics from the local corpus.",
            )

        if self._contains_new_topic_tech_term(text):
            return IntentDecision(
                False,
                "keyword_direct",
                "Question is a general technical or database topic outside the local corpus.",
            )

        previous_state = self._previous_route_state(recent_history)
        if (
            previous_state is not None
            and previous_state.used_rag is True
            and previous_state.had_contexts
            and self._is_strong_referential_followup(question)
        ):
            return IntentDecision(
                True,
                "state_rag",
                "Strong referential follow-up after the previous answer used retrieval.",
            )

        return None

    def _route_with_embeddings(
        self,
        question: str,
        recent_history: Sequence[HistoryMessage],
        conversation_summary: str,
    ) -> IntentDecision | None:
        if self.embedder is None:
            return None

        try:
            self._ensure_anchor_vectors()
            query_vector = self._embed_text(
                self._classification_text(
                    question,
                    recent_history,
                    conversation_summary,
                )
            )
        except Exception:
            return None

        if self._rag_anchor_vectors is None or self._direct_anchor_vectors is None:
            return None

        rag_score = float(np.max(self._rag_anchor_vectors @ query_vector))
        direct_score = float(np.max(self._direct_anchor_vectors @ query_vector))
        margin = rag_score - direct_score

        if (
            rag_score >= self.config.intent_embedding_rag_threshold
            and margin >= self.config.intent_embedding_margin
        ):
            return IntentDecision(
                True,
                "embedding_rag",
                (
                    "Embedding intent matched knowledge-base question patterns "
                    f"(rag={rag_score:.3f}, direct={direct_score:.3f})."
                ),
            )

        if (
            direct_score >= self.config.intent_embedding_direct_threshold
            and -margin >= self.config.intent_embedding_margin
        ):
            return IntentDecision(
                False,
                "embedding_direct",
                (
                    "Embedding intent matched direct-chat topics "
                    f"(rag={rag_score:.3f}, direct={direct_score:.3f})."
                ),
            )

        return None

    def _route_with_llm(
        self,
        question: str,
        recent_history: Sequence[HistoryMessage],
        conversation_summary: str,
    ) -> IntentDecision | None:
        previous_state = self._previous_route_state(recent_history)
        history_text = self._format_history(
            recent_history,
            max_chars=min(self.budget.intent_llm_history_max_chars, 3000),
            strategy="tail",
        )
        summary_text = self.budget.trim_text(
            conversation_summary,
            min(self.budget.intent_llm_summary_max_chars, 4000),
            strategy="middle",
        ) or "None"
        prompt = (
            "This application has a replaceable local Markdown knowledge base. "
            "The currently indexed corpus is a synthetic Chinese restaurant "
            "and business case about 家是本, 朱剑秋, 勇哥连线, menu pricing, "
            "customer reviews, Bilibili/social-media reactions, financial "
            "simulation, the related timeline, and the 巨大历史机遇/巨大历史鲫鱼 "
            "slogan meme. Use retrieval only when "
            "the current question is about that local corpus, asks for local "
            "document evidence, asks for citations/references, or is a "
            "follow-up whose recent conversation clearly stays on that corpus. "
            "Use direct chat for unrelated general knowledge, programming, "
            "SQL/database questions, greetings, casual conversation, creative "
            "writing, translation, or requests that explicitly do not need "
            "documents. Previous route state is a hint, not a command: if the "
            "previous answer used retrieval and the current question is a real "
            "follow-up, prefer retrieval; if the current question introduces a "
            "new general topic or direct task, prefer direct chat.\n\n"
            f"Current question:\n{question}\n\n"
            f"Previous route state:\n{self._format_previous_route_state(previous_state)}\n\n"
            f"Recent conversation:\n{history_text or 'None'}\n\n"
            f"Compact memory:\n{summary_text}\n\n"
            "Decide whether answering should use the knowledge-base vector "
            "store. Return exactly "
            "<think>THINK_AND_JUDGEMENT</think>"
            "<answer>JSON_ANS</answer>, where JSON_ANS is compact JSON "
            "with keys use_rag and reason."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier. Write a brief judgement in "
                    "<think>...</think>, then write only the route JSON in "
                    "<answer>...</answer>. Do not write anything outside those "
                    "tags. Example: <think>local-corpus follow-up</think>"
                    "<answer>{\"use_rag\": true, \"reason\": \"...\"}</answer>."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self.llm_client.chat(
                messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=min(
                    self.budget.intent_llm_max_tokens,
                    self.config.llm_max_tokens,
                ),
            )
        except Exception:
            return None

        parsed = self._parse_llm_decision(raw)
        if parsed is None:
            return None

        use_rag, reason = parsed
        return IntentDecision(
            use_rag,
            "llm_rag" if use_rag else "llm_direct",
            reason or "LLM classifier decision.",
        )

    def _ensure_anchor_vectors(self) -> None:
        if (
            self._rag_anchor_vectors is not None
            and self._direct_anchor_vectors is not None
        ):
            return
        with self._anchor_lock:
            if (
                self._rag_anchor_vectors is not None
                and self._direct_anchor_vectors is not None
            ):
                return
            self._rag_anchor_vectors = self._embed_many(self.RAG_ANCHORS)
            self._direct_anchor_vectors = self._embed_many(self.DIRECT_ANCHORS)

    def _embed_many(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def _embed_text(self, text: str) -> np.ndarray:
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    def _classification_text(
        self,
        question: str,
        recent_history: Sequence[HistoryMessage],
        conversation_summary: str,
    ) -> str:
        history_text = self._format_history(
            recent_history,
            max_chars=self.budget.intent_embedding_history_max_chars,
            strategy="tail",
        )
        summary_text = self.budget.trim_text(
            conversation_summary,
            self.budget.intent_embedding_summary_max_chars,
            strategy="middle",
        )
        parts = [
            f"Current question:\n{question.strip()}",
            f"Recent conversation:\n{history_text}" if history_text else "",
            f"Compact memory:\n{summary_text}" if summary_text else "",
        ]
        return self.budget.trim_text(
            "\n\n".join(part for part in parts if part),
            self.budget.intent_embedding_text_max_chars,
            strategy="middle",
        )

    def _format_history(
        self,
        history: Sequence[HistoryMessage],
        max_chars: int,
        strategy: TrimStrategy = "middle",
    ) -> str:
        return self.budget.format_history(history, max_chars, strategy=strategy)

    def _previous_route_state(
        self,
        history: Sequence[HistoryMessage],
    ) -> PreviousRouteState | None:
        previous_user_question = ""
        for index in range(len(history) - 1, -1, -1):
            message = history[index]
            if message.role != "assistant":
                continue

            used_rag = getattr(message, "used_rag", None)
            route = str(getattr(message, "route", "") or "")
            context_count = max(0, int(getattr(message, "context_count", 0) or 0))
            for previous in range(index - 1, -1, -1):
                if history[previous].role == "user":
                    previous_user_question = history[previous].content.strip()
                    break

            if used_rag is None and not route and not context_count:
                return None

            return PreviousRouteState(
                used_rag=used_rag,
                route=route,
                had_contexts=context_count > 0,
                previous_user_question=previous_user_question,
                previous_assistant_excerpt=self.budget.trim_text(
                    message.content,
                    600,
                    strategy="head",
                ),
            )
        return None

    def _format_previous_route_state(
        self,
        state: PreviousRouteState | None,
    ) -> str:
        if state is None:
            return "None"

        used_rag = "unknown" if state.used_rag is None else str(state.used_rag).lower()
        parts = [
            f"used_rag={used_rag}",
            f"route={state.route or 'unknown'}",
            f"had_contexts={str(state.had_contexts).lower()}",
        ]
        if state.previous_user_question:
            parts.append(
                "previous_user_question="
                + self.budget.trim_text(
                    state.previous_user_question,
                    600,
                    strategy="head",
                )
            )
        if state.previous_assistant_excerpt:
            parts.append(
                "previous_assistant_excerpt="
                + self.budget.trim_text(
                    state.previous_assistant_excerpt,
                    600,
                    strategy="head",
                )
            )
        return "\n".join(parts)

    def _is_strong_referential_followup(self, question: str) -> bool:
        stripped = question.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if len(stripped) > 80 or len(re.findall(r"\w+", lowered)) > 14:
            return False
        if self._contains_new_topic_tech_term(lowered):
            return False

        normalized = re.sub(r"[\s。！？!?，,、；;：:]+", "", lowered)
        if normalized in self.STRONG_REFERENTIAL_PHRASES:
            return True
        if lowered.strip(" \t\r\n.!?。！？") in self.STRONG_REFERENTIAL_PHRASES:
            return True
        return any(
            re.search(pattern, stripped, flags=re.IGNORECASE)
            for pattern in self.STRONG_REFERENTIAL_PATTERNS
        )

    def _contains_new_topic_tech_term(self, text: str) -> bool:
        return any(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            )
            for term in self.NEW_TOPIC_TECH_TERMS
        )

    def _parse_llm_decision(self, raw: str) -> tuple[bool, str] | None:
        answer_match = re.search(
            r"</?answer>\s*(.*?)\s*</answer>",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )
        source = answer_match.group(1) if answer_match else raw
        json_match = re.search(r"\{.*\}", source, flags=re.DOTALL)
        candidate = json_match.group(0) if json_match else source.strip()

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            lowered = source.lower()
            if "true" in lowered and "false" not in lowered:
                return True, "LLM classifier returned true."
            if "false" in lowered and "true" not in lowered:
                return False, "LLM classifier returned false."
            return None

        if "use_rag" not in data:
            return None

        use_rag = data["use_rag"]
        if isinstance(use_rag, bool):
            parsed_use_rag = use_rag
        elif isinstance(use_rag, str) and use_rag.lower() in {"true", "false"}:
            parsed_use_rag = use_rag.lower() == "true"
        else:
            return None

        return parsed_use_rag, str(data.get("reason", "")).strip()

    def _matches_direct_task(self, text: str) -> bool:
        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in self.DIRECT_TASK_PATTERNS
        )

    def _matches_domain_rag(self, text: str) -> bool:
        if self._contains_any(text, self.DOMAIN_RAG_PHRASES):
            return True
        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in self.DOMAIN_RAG_PATTERNS
        )

    def _first_match(self, text: str, patterns: Sequence[str]) -> str | None:
        for pattern in patterns:
            if pattern in text:
                return pattern
        return None

    def _contains_any(self, text: str, patterns: Sequence[str]) -> bool:
        return self._first_match(text, patterns) is not None
