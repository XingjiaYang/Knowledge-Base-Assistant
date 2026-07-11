from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from threading import Lock
import time
from typing import Any

from app.config import Settings, settings


logger = logging.getLogger(__name__)

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

    def warmup(self) -> int:
        return self.vector_size()

    def health(self) -> dict[str, object]:
        vector_size = getattr(self.store, "_vector_size", None)
        ready = getattr(self.store, "_model", None) is not None and bool(vector_size)
        if not ready:
            raise RuntimeError("Document embedding actor is not warmed up.")
        return {
            "ready": True,
            "model": self.store.config.embedding_model,
            "vector_size": int(vector_size),
            "device": str(getattr(self.store, "_model_device", "unknown")),
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

    def rerank(
        self,
        query: str,
        contexts: list[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        return self.reranker.rerank(query, contexts, top_k=top_k)

    def warmup(self) -> None:
        self.reranker.warmup()

    def health(self) -> dict[str, object]:
        if getattr(self.reranker, "_model", None) is None:
            raise RuntimeError("Reranker actor is not warmed up.")
        return {
            "ready": True,
            "model": self.reranker.config.reranker_model,
            "device": str(getattr(self.reranker, "_model_device", "unknown")),
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


def warmup_model_actors(config: Settings = settings) -> None:
    embedding_ref: Any | None = None
    code_embedding_actor: Any | None = None
    reranker_refs: list[tuple[str, Any]] = []

    try:
        embedding_actor = get_embedding_actor(config)
        if embedding_actor is not None:
            embedding_ref = embedding_actor.warmup.remote()
    except Exception:
        logger.exception("Ray document embedding actor startup failed.")

    if config.code_embedding_preload:
        try:
            code_embedding_actor = get_code_embedding_actor(config)
        except Exception:
            logger.exception("Ray CodeBERT embedding actor startup failed.")

    if config.reranker_enabled:
        try:
            for actor_name, reranker_actor in get_reranker_actors(config):
                reranker_refs.append((actor_name, reranker_actor.warmup.remote()))
        except Exception:
            logger.exception("Ray reranker actor startup failed.")

    if code_embedding_actor is not None:
        for attempt in range(1, config.code_embedding_preload_retries + 1):
            try:
                ray_get(code_embedding_actor.warmup.remote(), config)
                break
            except Exception as exc:
                if attempt >= config.code_embedding_preload_retries:
                    logger.exception("Ray CodeBERT embedding actor warmup failed.")
                    break
                logger.warning(
                    "Ray CodeBERT embedding actor warmup failed; retrying "
                    "attempt %s/%s in %.1fs: %s",
                    attempt + 1,
                    config.code_embedding_preload_retries,
                    config.code_embedding_preload_retry_seconds,
                    exc,
                )
                time.sleep(config.code_embedding_preload_retry_seconds)

    if embedding_ref is not None:
        try:
            ray_get(embedding_ref, config)
        except Exception:
            logger.exception("Ray document embedding actor warmup failed.")

    for actor_name, reranker_ref in reranker_refs:
        try:
            ray_get(reranker_ref, config)
        except Exception:
            logger.exception("Ray reranker actor %s warmup failed.", actor_name)


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
