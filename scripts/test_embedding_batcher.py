from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.model_actors import EmbeddingActor


class FakeEmbeddingStore:
    def __init__(self) -> None:
        self.config = SimpleNamespace(embedding_model="fake-embedding")
        self._model = object()
        self._vector_size = 2
        self.calls: list[tuple[list[str], str | None, str | None]] = []

    def _encode_matrix(
        self,
        texts: str | list[str],
        *,
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None,
    ) -> list[list[float]]:
        del normalize_embeddings
        rows = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append((rows, task, prompt_name))
        return [
            [float(len(text)), float(sum(text.encode("utf-8")))]
            for text in rows
        ]


async def stop_batch_tasks(actor: EmbeddingActor) -> None:
    for task in actor._batch_tasks.values():  # noqa: SLF001
        task.cancel()
    await asyncio.gather(
        *actor._batch_tasks.values(),  # noqa: SLF001
        return_exceptions=True,
    )


async def assert_query_batching() -> None:
    config = Settings(
        embedding_dynamic_batch_enabled=True,
        embedding_dynamic_batch_max_size=4,
        embedding_dynamic_batch_wait_ms=20.0,
        ray_enabled=False,
    )
    actor = EmbeddingActor(config)
    fake_store = FakeEmbeddingStore()
    actor.store = fake_store

    async def fake_encode_direct(
        _actor: EmbeddingActor,
        texts: str | list[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None,
    ) -> list[list[float]]:
        return fake_store._encode_matrix(
            texts,
            normalize_embeddings=normalize_embeddings,
            task=task,
            prompt_name=prompt_name,
        )

    actor._encode_direct = MethodType(  # type: ignore[method-assign]  # noqa: SLF001
        fake_encode_direct,
        actor,
    )

    queries = [f"query-{index}" for index in range(6)]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    actor.encode_matrix(query, True, "retrieval", "query")
                    for query in queries
                )
            ),
            timeout=2.0,
        )
    except TimeoutError as exc:
        task_states = {
            key: (
                task.done(),
                repr(task.exception()) if task.done() and not task.cancelled() else "",
            )
            for key, task in actor._batch_tasks.items()  # noqa: SLF001
        }
        raise AssertionError(f"Dynamic batch timed out: {task_states}") from exc

    if sorted(len(call[0]) for call in fake_store.calls) != [2, 4]:
        raise AssertionError(f"Expected dynamic batches of 4 and 2: {fake_store.calls}")
    expected = [
        [[float(len(query)), float(sum(query.encode("utf-8")))]]
        for query in queries
    ]
    if results != expected:
        raise AssertionError("Dynamic batch results were not returned to the right caller.")

    actor._warmup_state = "ready"  # noqa: SLF001 - fake store bypasses startup warmup
    health = actor.health()
    batching = health["dynamic_batching"]
    if not isinstance(batching, dict):
        raise AssertionError("Embedding health should include dynamic batch metrics.")
    if (
        batching["batch_count"] != 2
        or batching["request_count"] != 6
        or batching["average_batch_size"] != 3.0
        or batching["max_observed_batch_size"] != 4
        or batching["queue_depth"] != 0
    ):
        raise AssertionError(f"Unexpected dynamic batch metrics: {batching}")

    before = len(fake_store.calls)
    document_rows = await actor.encode_matrix(
        ["document-a", "document-b"],
        True,
        "retrieval",
        "document",
    )
    if len(fake_store.calls) != before + 1 or len(document_rows) != 2:
        raise AssertionError("List-based document embedding should bypass query batching.")

    await stop_batch_tasks(actor)


async def assert_full_batch_wakes_early() -> None:
    config = Settings(
        embedding_dynamic_batch_enabled=True,
        embedding_dynamic_batch_max_size=4,
        embedding_dynamic_batch_wait_ms=500.0,
        ray_enabled=False,
    )
    actor = EmbeddingActor(config)
    fake_store = FakeEmbeddingStore()
    actor.store = fake_store

    async def fake_encode_direct(
        _actor: EmbeddingActor,
        texts: str | list[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None,
    ) -> list[list[float]]:
        return fake_store._encode_matrix(
            texts,
            normalize_embeddings=normalize_embeddings,
            task=task,
            prompt_name=prompt_name,
        )

    actor._encode_direct = MethodType(  # type: ignore[method-assign]  # noqa: SLF001
        fake_encode_direct,
        actor,
    )

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    first = asyncio.create_task(
        actor.encode_matrix("query-0", True, "retrieval", "query")
    )
    await asyncio.sleep(0.02)
    rest = [
        asyncio.create_task(
            actor.encode_matrix(f"query-{index}", True, "retrieval", "query")
        )
        for index in range(1, 4)
    ]
    await asyncio.wait_for(asyncio.gather(first, *rest), timeout=0.3)
    elapsed = loop.time() - started_at

    if len(fake_store.calls) != 1 or len(fake_store.calls[0][0]) != 4:
        raise AssertionError(f"Expected one full batch: {fake_store.calls}")
    if elapsed >= 0.3:
        raise AssertionError(
            "A full dynamic batch should execute before the wait deadline."
        )

    await stop_batch_tasks(actor)


def main() -> None:
    asyncio.run(assert_query_batching())
    asyncio.run(assert_full_batch_wakes_early())
    print("Embedding dynamic batcher -> ok")


if __name__ == "__main__":
    main()
