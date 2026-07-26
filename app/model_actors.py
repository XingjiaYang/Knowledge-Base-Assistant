from __future__ import annotations

import asyncio
from dataclasses import dataclass
import gc
import logging
from threading import Lock
import time
from typing import Any

from app.config import Settings, settings


logger = logging.getLogger(__name__)
_MEBIBYTE = 1024 * 1024

_ray_lock = Lock()
_embedding_actor: Any | None = None
_code_embedding_actor: Any | None = None
_reranker_actors: dict[str, Any] = {}
_reranker_round_robin_index = 0
_unavailable_actor_names: set[str] = set()


@dataclass
class _EmbeddingBatchRequest:
    text: str
    future: asyncio.Future[list[list[float]]]


class ModelWarmupError(RuntimeError):
    """Raised when the online model fleet cannot satisfy its startup envelope."""


def _cuda_memory_snapshot(device: str | None) -> dict[str, object]:
    if not device or not device.startswith("cuda"):
        return {
            "cuda": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }

    import torch

    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }

    torch.cuda.synchronize()
    return {
        "cuda": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / _MEBIBYTE, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / _MEBIBYTE, 2),
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated() / _MEBIBYTE,
            2,
        ),
        "peak_reserved_mb": round(
            torch.cuda.max_memory_reserved() / _MEBIBYTE,
            2,
        ),
    }


def _reset_cuda_peak_memory(device: str | None) -> None:
    if not device or not device.startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _release_cuda_cache(device: str | None) -> dict[str, object]:
    gc.collect()
    if device and device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return _cuda_memory_snapshot(device)


def _token_ids(tokenizer: object, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _synthetic_text(tokenizer: object, target_tokens: int, label: str) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("Warmup token target must be greater than 0.")

    seed = f"{label} retrieval capacity validation input. "
    text = seed
    token_ids = _token_ids(tokenizer, text)
    while len(token_ids) < target_tokens:
        multiplier = max(2, (target_tokens + len(token_ids) - 1) // len(token_ids))
        text *= multiplier
        token_ids = _token_ids(tokenizer, text)

    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode(
            token_ids[:target_tokens],
            skip_special_tokens=True,
        )
        token_ids = _token_ids(tokenizer, text)

    while len(token_ids) < target_tokens:
        text += seed
        token_ids = _token_ids(tokenizer, text)
    return text, len(token_ids)


def _stage_report(
    *,
    role: str,
    stage: str,
    device: str | None,
    started_at: float,
    **details: object,
) -> dict[str, object]:
    return {
        "role": role,
        "stage": stage,
        "ready": stage == "performance",
        "device": device or "unknown",
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "memory": _cuda_memory_snapshot(device),
        **details,
    }


class EmbeddingActor:
    def __init__(self, config: Settings = settings) -> None:
        from app.vector_store import VectorStore

        self.store = VectorStore(config, use_ray=False)
        self.config = config
        self._batch_queues: dict[
            tuple[bool, str | None, str | None],
            asyncio.Queue[_EmbeddingBatchRequest],
        ] = {}
        self._batch_tasks: dict[
            tuple[bool, str | None, str | None],
            asyncio.Task[None],
        ] = {}
        self._batch_events: dict[
            tuple[bool, str | None, str | None],
            asyncio.Event,
        ] = {}
        self._inference_lock: asyncio.Lock | None = None
        self._batch_count = 0
        self._batched_request_count = 0
        self._batch_size_total = 0
        self._last_batch_size = 0
        self._max_observed_batch_size = 0
        self._warmup_state = "cold"
        self._warmup_error = ""
        self._last_warmup_report: dict[str, object] = {}

    async def encode_matrix(
        self,
        texts: str | list[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        if not self.config.embedding_dynamic_batch_enabled or not isinstance(
            texts,
            str,
        ):
            return await self._encode_direct(
                texts,
                normalize_embeddings,
                task,
                prompt_name,
            )

        key = (normalize_embeddings, task, prompt_name)
        queue = self._batch_queues.get(key)
        if queue is None:
            queue = asyncio.Queue()
            event = asyncio.Event()
            self._batch_queues[key] = queue
            self._batch_events[key] = event
            self._batch_tasks[key] = asyncio.create_task(
                self._run_batch_queue(key, queue, event)
            )
        else:
            event = self._batch_events[key]

        future = asyncio.get_running_loop().create_future()
        queue.put_nowait(_EmbeddingBatchRequest(text=texts, future=future))
        event.set()
        return await future

    async def _run_batch_queue(
        self,
        key: tuple[bool, str | None, str | None],
        queue: asyncio.Queue[_EmbeddingBatchRequest],
        event: asyncio.Event,
    ) -> None:
        normalize_embeddings, task, prompt_name = key
        max_size = self.config.embedding_dynamic_batch_max_size
        wait_seconds = self.config.embedding_dynamic_batch_wait_ms / 1000
        while True:
            first = await queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + wait_seconds
            while len(batch) < max_size:
                event.clear()
                while len(batch) < max_size:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if len(batch) >= max_size or wait_seconds <= 0:
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except TimeoutError:
                    break

            try:
                rows = await self._encode_direct(
                    [request.text for request in batch],
                    normalize_embeddings,
                    task,
                    prompt_name,
                )
                if len(rows) != len(batch):
                    raise RuntimeError(
                        "Embedding dynamic batch returned an unexpected row count."
                    )
                self._batch_count += 1
                self._batched_request_count += len(batch)
                self._batch_size_total += len(batch)
                self._last_batch_size = len(batch)
                self._max_observed_batch_size = max(
                    self._max_observed_batch_size,
                    len(batch),
                )
                for request, row in zip(batch, rows, strict=True):
                    if not request.future.done():
                        request.future.set_result([row])
            except Exception as exc:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
            finally:
                for _request in batch:
                    queue.task_done()

    async def _encode_direct(
        self,
        texts: str | list[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None,
    ) -> list[list[float]]:
        if self._inference_lock is None:
            self._inference_lock = asyncio.Lock()
        async with self._inference_lock:
            return await asyncio.to_thread(
                self.store._encode_matrix,  # noqa: SLF001 - actor owns model process
                texts,
                normalize_embeddings=normalize_embeddings,
                task=task,
                prompt_name=prompt_name,
            )

    def vector_size(self) -> int:
        return self.store.vector_size

    def load_model(self) -> dict[str, object]:
        self._warmup_state = "loading"
        self._warmup_error = ""
        started_at = time.perf_counter()
        try:
            vector_size = self.vector_size()
            self._require_expected_device()
            self._warmup_state = "loaded"
            report = _stage_report(
                role="embedding",
                stage="load",
                device=self._model_device(),
                started_at=started_at,
                vector_size=vector_size,
            )
            self._last_warmup_report = report
            return report
        except Exception as exc:
            self._record_warmup_failure(exc)
            raise

    async def capacity_probe(
        self,
        batch_size: int,
        input_tokens: int,
    ) -> dict[str, object]:
        self._warmup_state = "capacity"
        self._warmup_error = ""
        started_at = time.perf_counter()
        try:
            self._require_expected_device()
            tokenizer = self._tokenizer()
            text, actual_tokens = _synthetic_text(
                tokenizer,
                input_tokens,
                "embedding",
            )
            _reset_cuda_peak_memory(self._model_device())
            rows = await self._encode_strict(
                [text] * max(1, batch_size),
                normalize_embeddings=True,
                task=self.config.embedding_query_task,
                prompt_name=self.config.embedding_query_prompt_name,
            )
            if len(rows) != max(1, batch_size):
                raise RuntimeError(
                    "Embedding capacity probe returned an unexpected row count."
                )
            self._require_expected_device()
            self._warmup_state = "capacity_validated"
            report = _stage_report(
                role="embedding",
                stage="capacity",
                device=self._model_device(),
                started_at=started_at,
                batch_size=max(1, batch_size),
                input_tokens=actual_tokens,
                padded_tokens=actual_tokens * max(1, batch_size),
            )
            self._last_warmup_report = report
            return report
        except Exception as exc:
            self._record_warmup_failure(exc)
            raise

    async def performance_warmup(
        self,
        batch_size: int,
        input_tokens: int,
        rounds: int,
    ) -> dict[str, object]:
        self._warmup_state = "performance"
        self._warmup_error = ""
        started_at = time.perf_counter()
        try:
            self._require_expected_device()
            text, actual_tokens = _synthetic_text(
                self._tokenizer(),
                input_tokens,
                "embedding",
            )
            _reset_cuda_peak_memory(self._model_device())
            for _round in range(max(1, rounds)):
                if self.config.embedding_dynamic_batch_enabled:
                    rows = await asyncio.gather(
                        *(
                            self.encode_matrix(
                                text,
                                True,
                                self.config.embedding_query_task,
                                self.config.embedding_query_prompt_name,
                            )
                            for _index in range(max(1, batch_size))
                        )
                    )
                    if len(rows) != max(1, batch_size):
                        raise RuntimeError(
                            "Embedding performance warmup lost batched requests."
                        )
                else:
                    rows = await self._encode_strict(
                        [text] * max(1, batch_size),
                        normalize_embeddings=True,
                        task=self.config.embedding_query_task,
                        prompt_name=self.config.embedding_query_prompt_name,
                    )
                    if len(rows) != max(1, batch_size):
                        raise RuntimeError(
                            "Embedding performance warmup returned an unexpected "
                            "row count."
                        )
            self._require_expected_device()
            self._warmup_state = "ready"
            report = _stage_report(
                role="embedding",
                stage="performance",
                device=self._model_device(),
                started_at=started_at,
                batch_size=max(1, batch_size),
                input_tokens=actual_tokens,
                padded_tokens=actual_tokens * max(1, batch_size),
                rounds=max(1, rounds),
            )
            self._last_warmup_report = report
            return report
        except Exception as exc:
            self._record_warmup_failure(exc)
            raise

    async def warmup(self) -> dict[str, object]:
        await asyncio.to_thread(self.load_model)
        if self.config.model_warmup_capacity_enabled and self._expects_cuda():
            await self.capacity_probe(
                self.config.embedding_dynamic_batch_max_size,
                self.config.embedding_warmup_capacity_tokens,
            )
            await asyncio.to_thread(self.release_cuda_cache)
        return await self.performance_warmup(
            self.config.embedding_dynamic_batch_max_size,
            self.config.embedding_warmup_representative_tokens,
            self.config.embedding_warmup_rounds,
        )

    def release_cuda_cache(self) -> dict[str, object]:
        return _release_cuda_cache(self._model_device())

    async def _encode_strict(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None,
    ) -> list[list[float]]:
        if self._inference_lock is None:
            self._inference_lock = asyncio.Lock()
        async with self._inference_lock:
            return await asyncio.to_thread(
                self._encode_strict_sync,
                texts,
                normalize_embeddings,
                task,
                prompt_name,
            )

    def _encode_strict_sync(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None,
    ) -> list[list[float]]:
        encode_kwargs = self.store._encode_kwargs(  # noqa: SLF001
            task,
            prompt_name=prompt_name,
        )
        encode_kwargs["batch_size"] = len(texts)
        raw_embeddings = self.store.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            **encode_kwargs,
        )
        return self.store._embedding_matrix(raw_embeddings)  # noqa: SLF001

    def _tokenizer(self) -> object:
        model = self.store.model
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.store.chunk_tokenizer
        return tokenizer

    def _model_device(self) -> str:
        return str(getattr(self.store, "_model_device", "unknown"))

    def _expects_cuda(self) -> bool:
        return (
            self.config.cuda_enabled
            and self.config.ray_embedding_actor_num_gpus > 0
        )

    def _require_expected_device(self) -> None:
        if self._expects_cuda() and not self._model_device().startswith("cuda"):
            raise RuntimeError(
                "Embedding warmup required CUDA but the model is not on CUDA."
            )

    def _record_warmup_failure(self, error: Exception) -> None:
        self._warmup_state = "failed"
        self._warmup_error = str(error)

    def health(self) -> dict[str, object]:
        vector_size = getattr(self.store, "_vector_size", None)
        ready = getattr(self.store, "_model", None) is not None and bool(vector_size)
        ready = ready and self._warmup_state == "ready"
        if not ready:
            detail = f": {self._warmup_error}" if self._warmup_error else ""
            raise RuntimeError(
                "Document embedding actor is not warmed up "
                f"(state={self._warmup_state}){detail}."
            )
        return {
            "ready": True,
            "model": self.store.config.embedding_model,
            "vector_size": int(vector_size),
            "device": str(getattr(self.store, "_model_device", "unknown")),
            "warmup": {
                "state": self._warmup_state,
                "last_report": self._last_warmup_report,
            },
            "dynamic_batching": {
                "enabled": self.config.embedding_dynamic_batch_enabled,
                "max_size": self.config.embedding_dynamic_batch_max_size,
                "wait_ms": self.config.embedding_dynamic_batch_wait_ms,
                "batch_count": self._batch_count,
                "request_count": self._batched_request_count,
                "average_batch_size": (
                    self._batch_size_total / self._batch_count
                    if self._batch_count
                    else 0.0
                ),
                "last_batch_size": self._last_batch_size,
                "max_observed_batch_size": self._max_observed_batch_size,
                "queue_depth": sum(
                    queue.qsize() for queue in self._batch_queues.values()
                ),
            },
        }


class RerankerActor:
    def __init__(self, config: Settings = settings) -> None:
        from app.reranker import Reranker

        self.reranker = Reranker(config, use_ray=False)
        self.config = config
        self._warmup_state = "cold"
        self._warmup_error = ""
        self._last_warmup_report: dict[str, object] = {}

    def rerank(
        self,
        query: str,
        contexts: list[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        return self.reranker.rerank(query, contexts, top_k=top_k)

    def load_model(self) -> dict[str, object]:
        self._warmup_state = "loading"
        self._warmup_error = ""
        started_at = time.perf_counter()
        try:
            self.reranker.model
            self._require_expected_device()
            self._warmup_state = "loaded"
            report = _stage_report(
                role="reranker",
                stage="load",
                device=self._model_device(),
                started_at=started_at,
            )
            self._last_warmup_report = report
            return report
        except Exception as exc:
            self._record_warmup_failure(exc)
            raise

    def capacity_probe(
        self,
        document_count: int,
        document_tokens: int,
        query_tokens: int,
    ) -> dict[str, object]:
        return self._run_probe(
            stage="capacity",
            document_count=document_count,
            document_tokens=document_tokens,
            query_tokens=query_tokens,
            rounds=1,
        )

    def performance_warmup(
        self,
        document_count: int,
        document_tokens: int,
        query_tokens: int,
        rounds: int,
    ) -> dict[str, object]:
        return self._run_probe(
            stage="performance",
            document_count=document_count,
            document_tokens=document_tokens,
            query_tokens=query_tokens,
            rounds=rounds,
        )

    def warmup(self) -> dict[str, object]:
        self.load_model()
        if self.config.model_warmup_capacity_enabled and self._expects_cuda():
            self.capacity_probe(
                _reranker_warmup_document_count(self.config),
                _reranker_capacity_document_tokens(self.config),
                self.config.reranker_warmup_capacity_query_tokens,
            )
            self.release_cuda_cache()
        return self.performance_warmup(
            _reranker_warmup_document_count(self.config),
            self.config.reranker_warmup_representative_document_tokens,
            self.config.reranker_warmup_representative_query_tokens,
            self.config.reranker_warmup_rounds,
        )

    def release_cuda_cache(self) -> dict[str, object]:
        return _release_cuda_cache(self._model_device())

    def _run_probe(
        self,
        *,
        stage: str,
        document_count: int,
        document_tokens: int,
        query_tokens: int,
        rounds: int,
    ) -> dict[str, object]:
        self._warmup_state = stage
        self._warmup_error = ""
        started_at = time.perf_counter()
        try:
            self._require_expected_device()
            tokenizer = self._tokenizer()
            query, actual_query_tokens = _synthetic_text(
                tokenizer,
                query_tokens,
                f"reranker {stage} query",
            )
            document, actual_document_tokens = _synthetic_text(
                tokenizer,
                document_tokens,
                f"reranker {stage} document",
            )
            documents = [
                self.reranker._document_text(  # noqa: SLF001
                    self._warmup_context(document, index)
                )
                for index in range(max(1, document_count))
            ]
            _reset_cuda_peak_memory(self._model_device())
            results: object = None
            for _round in range(max(1, rounds)):
                results = self.reranker.model.rerank(
                    query,
                    documents,
                    top_n=1,
                )
            if not isinstance(results, list) or not results:
                raise RuntimeError("Reranker warmup returned no results.")
            self._require_expected_device()
            self._warmup_state = (
                "capacity_validated" if stage == "capacity" else "ready"
            )
            report = _stage_report(
                role="reranker",
                stage=stage,
                device=self._model_device(),
                started_at=started_at,
                document_count=max(1, document_count),
                document_tokens=actual_document_tokens,
                query_tokens=actual_query_tokens,
                total_document_tokens=(
                    actual_document_tokens * max(1, document_count)
                ),
                rounds=max(1, rounds),
            )
            self._last_warmup_report = report
            return report
        except Exception as exc:
            self._record_warmup_failure(exc)
            raise

    def _tokenizer(self) -> object:
        model = self.reranker.model
        ensure_tokenizer = getattr(model, "_ensure_tokenizer", None)
        if callable(ensure_tokenizer):
            ensure_tokenizer()
        tokenizer = getattr(model, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Reranker model did not expose its tokenizer.")
        return tokenizer

    @staticmethod
    def _warmup_context(text: str, index: int) -> Any:
        from app.vector_store import SearchResult

        return SearchResult(
            text=text,
            source=f"warmup/candidate-{index}.md",
            chunk_id=index,
            score=0.0,
            headings=("Warmup",),
        )

    def _model_device(self) -> str:
        return str(getattr(self.reranker, "_model_device", "unknown"))

    def _expects_cuda(self) -> bool:
        return (
            self.config.cuda_enabled
            and self.config.ray_reranker_actor_num_gpus > 0
        )

    def _require_expected_device(self) -> None:
        if self._expects_cuda() and not self._model_device().startswith("cuda"):
            raise RuntimeError(
                "Reranker warmup required CUDA but the model is not on CUDA."
            )

    def _record_warmup_failure(self, error: Exception) -> None:
        self._warmup_state = "failed"
        self._warmup_error = str(error)

    def health(self) -> dict[str, object]:
        ready = getattr(self.reranker, "_model", None) is not None
        ready = ready and self._warmup_state == "ready"
        if not ready:
            detail = f": {self._warmup_error}" if self._warmup_error else ""
            raise RuntimeError(
                "Reranker actor is not warmed up "
                f"(state={self._warmup_state}){detail}."
            )
        return {
            "ready": True,
            "model": self.reranker.config.reranker_model,
            "device": str(getattr(self.reranker, "_model_device", "unknown")),
            "warmup": {
                "state": self._warmup_state,
                "last_report": self._last_warmup_report,
            },
        }


class CodeEmbeddingActor:
    def __init__(self, config: Settings = settings) -> None:
        from app.code_indexer import CodeEmbedder

        self.embedder = CodeEmbedder(config)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def vector_size(self) -> int:
        return self.embedder.vector_size

    def warmup(self) -> int:
        return self.embedder.warmup()

    def health(self) -> dict[str, object]:
        vector_size = getattr(self.embedder, "_vector_size", None)
        ready = getattr(self.embedder, "_model", None) is not None and bool(vector_size)
        if not ready:
            raise RuntimeError("Code embedding actor is not warmed up.")
        return {
            "ready": True,
            "model": self.embedder.config.code_embedding_model,
            "vector_size": int(vector_size),
        }


def get_embedding_actor(config: Settings = settings) -> Any | None:
    global _embedding_actor
    if not config.ray_enabled:
        return None
    with _ray_lock:
        if _embedding_actor is None:
            _embedding_actor = _get_or_create_actor(
                config,
                EmbeddingActor,
                config.ray_embedding_actor_name,
                num_gpus=config.ray_embedding_actor_num_gpus,
                node_resource=config.ray_embedding_actor_resource,
                max_concurrency=max(
                    32,
                    config.embedding_dynamic_batch_max_size * 4,
                ),
            )
        return _embedding_actor


def get_code_embedding_actor(config: Settings = settings) -> Any | None:
    global _code_embedding_actor
    if not config.ray_enabled:
        return None
    with _ray_lock:
        if _code_embedding_actor is None:
            _code_embedding_actor = _get_or_create_actor(
                config,
                CodeEmbeddingActor,
                config.ray_code_embedding_actor_name,
                num_gpus=config.ray_code_embedding_actor_num_gpus,
                node_resource=config.ray_code_embedding_actor_resource,
            )
        return _code_embedding_actor


def reset_code_embedding_actor(config: Settings = settings) -> None:
    global _code_embedding_actor
    with _ray_lock:
        _code_embedding_actor = None
        _unavailable_actor_names.discard(config.ray_code_embedding_actor_name)


def reset_embedding_actor(config: Settings = settings) -> None:
    global _embedding_actor
    with _ray_lock:
        _embedding_actor = None
        _unavailable_actor_names.discard(config.ray_embedding_actor_name)


def destroy_embedding_actor(config: Settings = settings) -> bool:
    """Permanently remove the configured detached embedding actor from Ray."""
    global _embedding_actor
    if not config.ray_enabled:
        return False

    with _ray_lock:
        actor = _embedding_actor
        _embedding_actor = None
        _unavailable_actor_names.discard(config.ray_embedding_actor_name)

    ray = _ensure_ray(config)
    if actor is None:
        try:
            actor = ray.get_actor(
                config.ray_embedding_actor_name,
                namespace=config.ray_namespace,
            )
        except ValueError:
            return False

    ray.kill(actor, no_restart=True)
    logger.info(
        "Destroyed detached Ray embedding actor %s.",
        config.ray_embedding_actor_name,
    )
    return True


def reset_reranker_actor(
    config: Settings = settings,
    actor_name: str | None = None,
) -> None:
    with _ray_lock:
        actor_names = (
            (actor_name,)
            if actor_name is not None
            else _reranker_actor_names(config)
        )
        for name in actor_names:
            _reranker_actors.pop(name, None)
            _unavailable_actor_names.discard(name)


def get_reranker_actor(config: Settings = settings) -> Any | None:
    selected = get_reranker_actor_for_request(config)
    if selected is None:
        return None
    _name, actor = selected
    return actor


def get_reranker_actor_for_request(
    config: Settings = settings,
    *,
    exclude_names: set[str] | None = None,
) -> tuple[str, Any] | None:
    global _reranker_round_robin_index
    if not config.ray_enabled:
        return None
    excluded = exclude_names or set()
    with _ray_lock:
        actors = _get_reranker_actors_locked(config)
        actors = [(name, actor) for name, actor in actors if name not in excluded]
        if not actors:
            return None
        index = _reranker_round_robin_index % len(actors)
        _reranker_round_robin_index += 1
        return actors[index]


def get_reranker_actors(
    config: Settings = settings,
    *,
    retry_unavailable: bool = False,
) -> list[tuple[str, Any]]:
    if not config.ray_enabled:
        return []
    with _ray_lock:
        if retry_unavailable:
            for name in _reranker_actor_names(config):
                _unavailable_actor_names.discard(name)
        return _get_reranker_actors_locked(config)


def reranker_actor_names(config: Settings = settings) -> tuple[str, ...]:
    return _reranker_actor_names(config)


def _get_reranker_actors_locked(config: Settings) -> list[tuple[str, Any]]:
    actors: list[tuple[str, Any]] = []
    for index, actor_name in enumerate(_reranker_actor_names(config), start=1):
        if actor_name in _unavailable_actor_names:
            continue
        actor = _reranker_actors.get(actor_name)
        if actor is None:
            actor = _get_or_create_actor(
                config,
                RerankerActor,
                actor_name,
                num_gpus=config.ray_reranker_actor_num_gpus,
                node_resource=_reranker_actor_resource(config, index),
            )
            if actor is not None:
                _reranker_actors[actor_name] = actor
        if actor is not None:
            actors.append((actor_name, actor))
    return actors


def _reranker_actor_names(config: Settings) -> tuple[str, ...]:
    replicas = max(1, config.ray_reranker_actor_replicas)
    base_name = config.ray_reranker_actor_name.strip()
    if replicas == 1:
        return (base_name,)
    return tuple(f"{base_name}_{index}" for index in range(1, replicas + 1))


def _reranker_actor_resource(config: Settings, replica_index: int) -> str:
    base_resource = config.ray_reranker_actor_resource.strip()
    if not base_resource:
        return ""
    if config.ray_reranker_actor_replicas <= 1:
        return base_resource
    return f"{base_resource}_{replica_index}"


def ray_get(
    ref: Any,
    config: Settings = settings,
    timeout_seconds: float | None = None,
) -> Any:
    import ray

    timeout = (
        config.ray_task_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    if timeout <= 0:
        return ray.get(ref)
    return ray.get(ref, timeout=timeout)


def mark_ray_unavailable(actor_name: str | None = None, config: Settings = settings) -> None:
    if actor_name is None:
        actor_names = (
            config.ray_embedding_actor_name,
            config.ray_code_embedding_actor_name,
            *_reranker_actor_names(config),
        )
    else:
        actor_names = (actor_name,)
    with _ray_lock:
        _unavailable_actor_names.update(actor_names)


def warmup_model_actors(config: Settings = settings) -> dict[str, object]:
    """Load and validate the online GPU fleet before retrieval accepts traffic."""
    started_at = time.perf_counter()
    report: dict[str, object] = {
        "ready": False,
        "load_order": [],
        "load": {},
        "capacity": {},
        "capacity_cache_release": {},
        "performance": {},
    }

    embedding_actor = get_embedding_actor(config)
    if embedding_actor is None:
        raise ModelWarmupError("Ray document embedding actor is unavailable.")

    reranker_actors = (
        get_reranker_actors(config, retry_unavailable=True)
        if config.reranker_enabled
        else []
    )
    if config.reranker_enabled and (
        len(reranker_actors) != config.ray_reranker_actor_replicas
    ):
        raise ModelWarmupError(
            "Not all configured Ray reranker replicas are available: "
            f"expected={config.ray_reranker_actor_replicas} "
            f"actual={len(reranker_actors)}."
        )

    load_reports = report["load"]
    load_order = report["load_order"]
    assert isinstance(load_reports, dict)
    assert isinstance(load_order, list)

    for actor_name, actor in reranker_actors:
        load_order.append(actor_name)
        load_reports[actor_name] = _run_remote_stage(
            actor.load_model.remote(),
            config,
            stage=f"{actor_name}.load",
        )

    load_order.append(config.ray_embedding_actor_name)
    load_reports[config.ray_embedding_actor_name] = _run_remote_stage(
        embedding_actor.load_model.remote(),
        config,
        stage=f"{config.ray_embedding_actor_name}.load",
    )

    capacity_reports = report["capacity"]
    cache_reports = report["capacity_cache_release"]
    assert isinstance(capacity_reports, dict)
    assert isinstance(cache_reports, dict)
    actors = [(config.ray_embedding_actor_name, embedding_actor), *reranker_actors]

    if config.model_warmup_capacity_enabled and config.cuda_enabled:
        capacity_refs: list[tuple[str, Any]] = [
            (
                config.ray_embedding_actor_name,
                embedding_actor.capacity_probe.remote(
                    config.embedding_dynamic_batch_max_size,
                    config.embedding_warmup_capacity_tokens,
                ),
            )
        ]
        capacity_refs.extend(
            (
                actor_name,
                actor.capacity_probe.remote(
                    _reranker_warmup_document_count(config),
                    _reranker_capacity_document_tokens(config),
                    config.reranker_warmup_capacity_query_tokens,
                ),
            )
            for actor_name, actor in reranker_actors
        )
        errors: list[str] = []
        for actor_name, ref in capacity_refs:
            try:
                capacity_reports[actor_name] = _run_remote_stage(
                    ref,
                    config,
                    stage=f"{actor_name}.capacity",
                )
            except Exception as exc:
                errors.append(f"{actor_name}: {exc}")

        release_refs = [
            (actor_name, actor.release_cuda_cache.remote())
            for actor_name, actor in actors
        ]
        for actor_name, ref in release_refs:
            try:
                cache_reports[actor_name] = _run_remote_stage(
                    ref,
                    config,
                    stage=f"{actor_name}.capacity_cache_release",
                )
            except Exception as exc:
                errors.append(f"{actor_name} cache release: {exc}")

        if errors:
            raise ModelWarmupError(
                "Concurrent maximum-capacity validation failed; retrieval will "
                "not become ready. " + "; ".join(errors)
            )
    else:
        report["capacity"] = {
            "skipped": True,
            "reason": (
                "disabled"
                if not config.model_warmup_capacity_enabled
                else "CUDA disabled"
            ),
        }

    performance_reports = report["performance"]
    assert isinstance(performance_reports, dict)
    performance_reports[config.ray_embedding_actor_name] = _run_remote_stage(
        embedding_actor.performance_warmup.remote(
            config.embedding_dynamic_batch_max_size,
            config.embedding_warmup_representative_tokens,
            config.embedding_warmup_rounds,
        ),
        config,
        stage=f"{config.ray_embedding_actor_name}.performance",
    )
    for actor_name, actor in reranker_actors:
        performance_reports[actor_name] = _run_remote_stage(
            actor.performance_warmup.remote(
                _reranker_warmup_document_count(config),
                config.reranker_warmup_representative_document_tokens,
                config.reranker_warmup_representative_query_tokens,
                config.reranker_warmup_rounds,
            ),
            config,
            stage=f"{actor_name}.performance",
        )

    _warmup_code_embedding_actor(config)
    report["ready"] = True
    report["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info("Ray online model fleet warmup complete: %s", report)
    return report


def _run_remote_stage(
    ref: Any,
    config: Settings,
    *,
    stage: str,
) -> object:
    try:
        return ray_get(
            ref,
            config,
            timeout_seconds=config.model_warmup_timeout_seconds,
        )
    except Exception as exc:
        raise ModelWarmupError(f"Model warmup stage {stage} failed: {exc}") from exc


def _reranker_warmup_document_count(config: Settings) -> int:
    return max(
        1,
        min(config.rrf_top_k, config.reranker_max_documents_per_call),
    )


def _reranker_capacity_document_tokens(config: Settings) -> int:
    return (
        config.chunk_body_max_tokens
        + (2 * config.chunk_overlap_max_tokens)
    )


def _warmup_code_embedding_actor(config: Settings) -> None:
    if not config.code_embedding_preload:
        return
    try:
        actor = get_code_embedding_actor(config)
    except Exception:
        logger.exception("Ray CodeBERT embedding actor startup failed.")
        return
    if actor is None:
        return

    for attempt in range(1, config.code_embedding_preload_retries + 1):
        try:
            ray_get(actor.warmup.remote(), config)
            return
        except Exception as exc:
            if attempt >= config.code_embedding_preload_retries:
                logger.exception("Ray CodeBERT embedding actor warmup failed.")
                return
            logger.warning(
                "Ray CodeBERT embedding actor warmup failed; retrying "
                "attempt %s/%s in %.1fs: %s",
                attempt + 1,
                config.code_embedding_preload_retries,
                config.code_embedding_preload_retry_seconds,
                exc,
            )
            time.sleep(config.code_embedding_preload_retry_seconds)


def _get_or_create_actor(
    config: Settings,
    actor_cls: type[Any],
    name: str,
    num_gpus: float,
    node_resource: str = "",
    max_concurrency: int | None = None,
) -> Any | None:
    if name in _unavailable_actor_names:
        return None

    try:
        ray = _ensure_ray(config)
        requested_gpus = max(0.0, num_gpus)
        available_gpus = float(ray.available_resources().get("GPU", 0.0))
        if requested_gpus > 0 and available_gpus <= 0 and not node_resource.strip():
            logger.warning(
                "Ray has no visible GPU resources for actor %s; scheduling on CPU.",
                name,
            )
            requested_gpus = 0.0
        try:
            return ray.get_actor(name, namespace=config.ray_namespace)
        except ValueError:
            pass

        remote_options: dict[str, Any] = {
            "num_cpus": config.ray_actor_num_cpus,
            "num_gpus": requested_gpus,
            "max_restarts": -1,
            "max_task_retries": 0,
        }
        if node_resource.strip():
            remote_options["resources"] = {node_resource.strip(): 0.001}
        if max_concurrency is not None:
            remote_options["max_concurrency"] = max(1, max_concurrency)

        remote_cls = ray.remote(**remote_options)(actor_cls)
        logger.info(
            "Starting Ray actor %s with cpus=%s gpus=%s resource=%s.",
            name,
            config.ray_actor_num_cpus,
            requested_gpus,
            node_resource or "none",
        )
        return remote_cls.options(
            name=name,
            lifetime="detached",
            get_if_exists=True,
            namespace=config.ray_namespace,
        ).remote(config)
    except Exception:
        _unavailable_actor_names.add(name)
        logger.exception(
            "Ray actor %s is unavailable; local_model_fallback=%s.",
            name,
            config.ray_local_fallback,
        )
        return None


def _ensure_ray(config: Settings) -> Any:
    import ray

    if ray.is_initialized():
        return ray

    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "namespace": config.ray_namespace,
        "log_to_driver": False,
    }
    if config.ray_address.strip():
        init_kwargs["address"] = config.ray_address.strip()

    ray.init(**init_kwargs)
    return ray
