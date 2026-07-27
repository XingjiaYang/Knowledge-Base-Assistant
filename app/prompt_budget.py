from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


TrimStrategy = Literal["head", "tail", "middle"]
TRUNCATION_MARKER = "\n...[truncated]...\n"


class PromptMessage(Protocol):
    role: str
    content: str


@dataclass(frozen=True)
class PromptBudget:
    conversation_summary_max_chars: int
    summary_history_max_chars: int
    summary_max_tokens: int
    search_query_max_chars: int
    intent_llm_history_max_chars: int
    intent_llm_summary_max_chars: int
    intent_llm_max_tokens: int
    intent_embedding_history_max_chars: int
    intent_embedding_summary_max_chars: int
    intent_embedding_text_max_chars: int

    @classmethod
    def from_config(cls, config: object) -> "PromptBudget":
        return cls(
            conversation_summary_max_chars=getattr(
                config,
                "conversation_summary_max_chars",
            ),
            summary_history_max_chars=getattr(config, "summary_history_max_chars"),
            summary_max_tokens=getattr(config, "summary_max_tokens"),
            search_query_max_chars=getattr(config, "search_query_max_chars"),
            intent_llm_history_max_chars=getattr(
                config,
                "intent_llm_history_max_chars",
            ),
            intent_llm_summary_max_chars=getattr(
                config,
                "intent_llm_summary_max_chars",
            ),
            intent_llm_max_tokens=getattr(config, "intent_llm_max_tokens"),
            intent_embedding_history_max_chars=getattr(
                config,
                "intent_embedding_history_max_chars",
            ),
            intent_embedding_summary_max_chars=getattr(
                config,
                "intent_embedding_summary_max_chars",
            ),
            intent_embedding_text_max_chars=getattr(
                config,
                "intent_embedding_text_max_chars",
            ),
        )

    def trim_summary(self, summary: str) -> str:
        return self.trim_text(
            summary,
            self.conversation_summary_max_chars,
            strategy="middle",
        )

    def trim_text(
        self,
        text: str,
        max_chars: int,
        strategy: TrimStrategy = "middle",
    ) -> str:
        text = text.strip()
        max_chars = max(0, max_chars)
        if not max_chars or len(text) <= max_chars:
            return text

        if max_chars <= len(TRUNCATION_MARKER) + 2:
            return text[:max_chars].strip()

        if strategy == "head":
            return text[:max_chars].strip()
        if strategy == "tail":
            return text[-max_chars:].strip()

        available = max_chars - len(TRUNCATION_MARKER)
        head_chars = available // 2
        tail_chars = available - head_chars
        return (
            text[:head_chars].rstrip()
            + TRUNCATION_MARKER
            + text[-tail_chars:].lstrip()
        ).strip()

    def format_history(
        self,
        history: Sequence[PromptMessage],
        max_chars: int | None = None,
        strategy: TrimStrategy = "middle",
    ) -> str:
        text = "\n\n".join(
            f"{message.role.upper()}: {message.content.strip()}"
            for message in history
            if message.content.strip()
        )
        if max_chars is None:
            return text
        return self.trim_text(text, max_chars, strategy=strategy)
