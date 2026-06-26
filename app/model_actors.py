from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from app.config import Settings, settings


logger = logging.getLogger(__name__)

_ray_lock = Lock()
_embedding_actor: Any | None = None
_reranker_actor: Any | None = None
_ray_unavailable = False


class EmbeddingActor:
    def __init__(self, config: Settings = settings) -> None:
        from app.vector_store import VectorStore

        self.store = VectorStore(config, use_ray=False)

    def encode_matrix(
        self,
        texts: str | list[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        return self.store._encode_matrix(  # noqa: SLF001 - actor owns model process
            texts,
            normalize_embeddings=normalize_embeddings,
            task=task,
            prompt_name=prompt_name,
        )

    def vector_size(self) -> int:
        return self.store.vector_size

    def warmup(self) -> int:
        return self.vector_size()


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
            )
        return _embedding_actor


def get_reranker_actor(config: Settings = settings) -> Any | None:
    global _reranker_actor
    if not config.ray_enabled:
        return None
    with _ray_lock:
        if _reranker_actor is None:
            _reranker_actor = _get_or_create_actor(
                config,
                RerankerActor,
                config.ray_reranker_actor_name,
                num_gpus=config.ray_reranker_actor_num_gpus,
            )
        return _reranker_actor


def ray_get(ref: Any, config: Settings = settings) -> Any:
    import ray

    timeout = config.ray_task_timeout_seconds
    if timeout <= 0:
        return ray.get(ref)
    return ray.get(ref, timeout=timeout)


def mark_ray_unavailable() -> None:
    global _ray_unavailable
    _ray_unavailable = True


def warmup_model_actors(config: Settings = settings) -> None:
    embedding_actor = get_embedding_actor(config)
    if embedding_actor is not None:
        ray_get(embedding_actor.warmup.remote(), config)

    if config.reranker_enabled:
        reranker_actor = get_reranker_actor(config)
        if reranker_actor is not None:
            ray_get(reranker_actor.warmup.remote(), config)


def _get_or_create_actor(
    config: Settings,
    actor_cls: type[Any],
    name: str,
    num_gpus: float,
) -> Any | None:
    global _ray_unavailable
    if _ray_unavailable:
        return None

    try:
        ray = _ensure_ray(config)
        requested_gpus = max(0.0, num_gpus)
        available_gpus = float(ray.available_resources().get("GPU", 0.0))
        if requested_gpus > 0 and available_gpus <= 0:
            logger.warning(
                "Ray has no visible GPU resources for actor %s; scheduling on CPU.",
                name,
            )
            requested_gpus = 0.0
        try:
            return ray.get_actor(name, namespace=config.ray_namespace)
        except ValueError:
            pass

        remote_cls = ray.remote(
            num_cpus=config.ray_actor_num_cpus,
            num_gpus=requested_gpus,
        )(actor_cls)
        logger.info(
            "Starting Ray actor %s with cpus=%s gpus=%s.",
            name,
            config.ray_actor_num_cpus,
            requested_gpus,
        )
        return remote_cls.options(name=name).remote(config)
    except Exception:
        _ray_unavailable = True
        logger.exception(
            "Ray actor %s is unavailable; falling back to local model.",
            name,
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
