from __future__ import annotations

from dataclasses import replace
import logging
from threading import Lock
from typing import Sequence

from app.config import Settings, settings
from app.device import preferred_torch_device
from app.transformers_compat import patch_all_tied_weights_keys
from app.vector_store import SearchResult


logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        config: Settings = settings,
        model: object | None = None,
        use_ray: bool = True,
    ) -> None:
        self.config = config
        self._model = model
        self._model_device = "cpu" if model is not None else None
        self._use_ray = use_ray and model is None
        self._model_lock = Lock()

    @property
    def model(self) -> object:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    patch_all_tied_weights_keys()
                    from transformers import AutoModel

                    device = preferred_torch_device(
                        self.config.cuda_enabled,
                        "Reranker model",
                    )
                    logger.info(
                        "Loading reranker model %s on %s.",
                        self.config.reranker_model,
                        device,
                    )
                    model = AutoModel.from_pretrained(
                        self.config.reranker_model,
                        trust_remote_code=self.config.reranker_trust_remote_code,
                        dtype=self.config.reranker_dtype,
                    )
                    self._model_device = self._move_model_to_device(model, device)
                    model.eval()
                    self._model = model
        return self._model

    @staticmethod
    def _move_model_to_device(model: object, device: str) -> str:
        if not hasattr(model, "to"):
            return device

        try:
            model.to(device)
            return device
        except Exception as exc:
            if device != "cuda":
                raise
            logger.warning(
                "Reranker model failed to move to CUDA; falling back to CPU: %s",
                exc,
            )
            model.to("cpu")
            return "cpu"

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

        if self._use_ray:
            attempted_actor_names: set[str] = set()
            while True:
                selected_actor = self._reranker_actor(
                    exclude_names=attempted_actor_names,
                )
                if selected_actor is None:
                    break
                actor_name, actor = selected_actor
                attempted_actor_names.add(actor_name)
                try:
                    from app.model_actors import mark_ray_unavailable, ray_get

                    return ray_get(
                        actor.rerank.remote(query, list(contexts), top_k),
                        self.config,
                    )
                except Exception:
                    logger.exception(
                        "Reranker Ray actor %s failed; local_model_fallback=%s.",
                        actor_name,
                        self.config.ray_local_fallback,
                    )
                    mark_ray_unavailable(actor_name)
                    if (
                        not self.config.ray_local_fallback
                        and len(attempted_actor_names)
                        >= self.config.ray_reranker_actor_replicas
                    ):
                        raise RuntimeError(
                            "All Reranker Ray actors failed and RAY_LOCAL_FALLBACK=0."
                        )
            if not self.config.ray_local_fallback:
                raise RuntimeError(
                    "Reranker Ray actors are unavailable and RAY_LOCAL_FALLBACK=0."
                )

        limit = max(1, top_k or self.config.retrieve_top_k)
        documents = [self._document_text(context) for context in contexts]
        batch_size = self.config.reranker_max_documents_per_call
        if len(documents) <= batch_size:
            results = self._rerank_documents(
                query,
                documents,
                min(limit, len(contexts)),
            )
            return self._results_to_contexts(contexts, results)

        results: list[dict[str, object]] = []
        for offset in range(0, len(documents), batch_size):
            batch_documents = documents[offset : offset + batch_size]
            batch_results = self._rerank_documents(query, batch_documents)
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

    def _rerank_documents(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        try:
            return self.model.rerank(query, documents, top_n=top_n)
        except Exception as exc:
            if self._model_device != "cuda":
                raise
            logger.warning(
                "Reranker failed during CUDA inference; falling back to CPU: %s",
                exc,
            )
            self._model_device = self._move_model_to_device(self.model, "cpu")
            return self.model.rerank(query, documents, top_n=top_n)

    def _reranker_actor(
        self,
        *,
        exclude_names: set[str] | None = None,
    ) -> tuple[str, object] | None:
        if not self._use_ray:
            return None
        try:
            from app.model_actors import get_reranker_actor_for_request

            return get_reranker_actor_for_request(
                self.config,
                exclude_names=exclude_names,
            )
        except Exception:
            logger.exception("Reranker Ray actor setup failed.")
            return None

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
