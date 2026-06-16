from __future__ import annotations

from dataclasses import replace
import logging
from threading import Lock
from typing import Sequence

from app.config import Settings, settings
from app.vector_store import SearchResult


logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        config: Settings = settings,
        model: object | None = None,
    ) -> None:
        self.config = config
        self._model = model
        self._model_lock = Lock()

    @property
    def model(self) -> object:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from transformers import AutoModel

                    logger.info(
                        "Loading reranker model %s.",
                        self.config.reranker_model,
                    )
                    model = AutoModel.from_pretrained(
                        self.config.reranker_model,
                        trust_remote_code=self.config.reranker_trust_remote_code,
                        dtype=self.config.reranker_dtype,
                    )
                    model.eval()
                    self._model = model
        return self._model

    def warmup(self) -> None:
        if not self.config.reranker_enabled:
            return

        self.rerank(
            "health check",
            [
                SearchResult(
                    text="This document verifies reranker startup.",
                    source="startup",
                    chunk_id=0,
                    score=0.0,
                )
            ],
            top_k=1,
        )

    def rerank(
        self,
        query: str,
        contexts: Sequence[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not contexts:
            return []

        limit = max(1, top_k or self.config.retrieve_top_k)
        documents = [self._document_text(context) for context in contexts]
        batch_size = self.config.reranker_max_documents_per_call
        if len(documents) <= batch_size:
            results = self.model.rerank(
                query,
                documents,
                top_n=min(limit, len(contexts)),
            )
            return self._results_to_contexts(contexts, results)

        results: list[dict[str, object]] = []
        for offset in range(0, len(documents), batch_size):
            batch_documents = documents[offset : offset + batch_size]
            batch_results = self.model.rerank(query, batch_documents)
            for result in batch_results:
                result = dict(result)
                result["index"] = int(result["index"]) + offset
                results.append(result)

        reranked = self._results_to_contexts(contexts, results)
        reranked.sort(
            key=lambda context: (
                context.rerank_score
                if context.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )
        return reranked[:limit]

    @staticmethod
    def _results_to_contexts(
        contexts: Sequence[SearchResult],
        results: Sequence[dict[str, object]],
    ) -> list[SearchResult]:
        reranked: list[SearchResult] = []
        for result in results:
            try:
                index = int(result["index"])
                score = float(result["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Reranker returned malformed results.") from exc

            if index < 0 or index >= len(contexts):
                raise RuntimeError("Reranker returned an out-of-range index.")
            reranked.append(replace(contexts[index], rerank_score=score))

        return reranked

    @staticmethod
    def _document_text(context: SearchResult) -> str:
        parts = [
            f"Source: {context.source}" if context.source else "",
            (
                f"Headings: {' > '.join(context.headings)}"
                if context.headings
                else ""
            ),
            f"Content type: {context.content_type}",
            context.text,
        ]
        return "\n\n".join(part for part in parts if part).strip()
