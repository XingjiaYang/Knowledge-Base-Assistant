from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
import app.model_actors as model_actors


class FakeRef:
    def __init__(
        self,
        name: str,
        result: dict[str, object],
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.result = result
        self.error = error


class FakeRemoteMethod:
    def __init__(
        self,
        owner: str,
        method: str,
        events: list[tuple[object, ...]],
        *,
        fail: bool = False,
    ) -> None:
        self.owner = owner
        self.method = method
        self.events = events
        self.fail = fail

    def remote(self, *args: object) -> FakeRef:
        name = f"{self.owner}.{self.method}"
        self.events.append(("submit", name, args))
        error = RuntimeError(f"{name} failed") if self.fail else None
        return FakeRef(
            name,
            {
                "role": self.owner,
                "stage": self.method,
                "args": list(args),
            },
            error=error,
        )


class FakeActor:
    def __init__(
        self,
        name: str,
        events: list[tuple[object, ...]],
        *,
        fail_capacity: bool = False,
    ) -> None:
        self.load_model = FakeRemoteMethod(name, "load", events)
        self.capacity_probe = FakeRemoteMethod(
            name,
            "capacity",
            events,
            fail=fail_capacity,
        )
        self.release_cuda_cache = FakeRemoteMethod(name, "release", events)
        self.performance_warmup = FakeRemoteMethod(name, "performance", events)


def _config() -> Settings:
    return Settings(
        cuda_enabled=True,
        ray_enabled=True,
        ray_embedding_actor_name="embedding",
        ray_reranker_actor_name="reranker",
        ray_reranker_actor_replicas=2,
        code_embedding_preload=False,
        embedding_dynamic_batch_max_size=4,
        embedding_warmup_capacity_tokens=32,
        embedding_warmup_representative_tokens=8,
        embedding_warmup_rounds=2,
        rrf_top_k=6,
        reranker_max_documents_per_call=6,
        chunk_body_target_tokens=10,
        chunk_body_max_tokens=10,
        chunk_overlap_target_tokens=2,
        chunk_overlap_max_tokens=2,
        reranker_warmup_capacity_query_tokens=8,
        reranker_warmup_representative_query_tokens=4,
        reranker_warmup_representative_document_tokens=14,
        reranker_warmup_rounds=2,
        model_warmup_capacity_enabled=True,
        model_warmup_timeout_seconds=30.0,
    )


def _run_with_fakes(*, fail_capacity: bool = False) -> tuple[
    dict[str, object] | None,
    Exception | None,
    list[tuple[object, ...]],
]:
    events: list[tuple[object, ...]] = []
    embedding = FakeActor("embedding", events)
    reranker_1 = FakeActor("reranker_1", events)
    reranker_2 = FakeActor(
        "reranker_2",
        events,
        fail_capacity=fail_capacity,
    )

    originals: dict[str, Any] = {
        "get_embedding_actor": model_actors.get_embedding_actor,
        "get_reranker_actors": model_actors.get_reranker_actors,
        "ray_get": model_actors.ray_get,
    }

    def fake_get_embedding_actor(_config: Settings) -> FakeActor:
        return embedding

    def fake_get_reranker_actors(
        _config: Settings,
        *,
        retry_unavailable: bool = False,
    ) -> list[tuple[str, FakeActor]]:
        if not retry_unavailable:
            raise AssertionError("Startup should retry previously unavailable actors.")
        return [
            ("reranker_1", reranker_1),
            ("reranker_2", reranker_2),
        ]

    def fake_ray_get(
        ref: FakeRef,
        _config: Settings,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        events.append(("await", ref.name, timeout_seconds))
        if ref.error is not None:
            raise ref.error
        return ref.result

    model_actors.get_embedding_actor = fake_get_embedding_actor
    model_actors.get_reranker_actors = fake_get_reranker_actors
    model_actors.ray_get = fake_ray_get
    report: dict[str, object] | None = None
    error: Exception | None = None
    try:
        report = model_actors.warmup_model_actors(_config())
    except Exception as exc:
        error = exc
    finally:
        for name, original in originals.items():
            setattr(model_actors, name, original)
    return report, error, events


def assert_startup_order() -> None:
    report, error, events = _run_with_fakes()
    if error is not None:
        raise error
    if report is None or report.get("ready") is not True:
        raise AssertionError("Successful model warmup should produce a ready report.")
    if report["load_order"] != ["reranker_1", "reranker_2", "embedding"]:
        raise AssertionError(f"Unexpected model load order: {report['load_order']}")

    submitted = [
        event[1]
        for event in events
        if event[0] == "submit"
    ]
    expected = [
        "reranker_1.load",
        "reranker_2.load",
        "embedding.load",
        "embedding.capacity",
        "reranker_1.capacity",
        "reranker_2.capacity",
        "embedding.release",
        "reranker_1.release",
        "reranker_2.release",
        "embedding.performance",
        "reranker_1.performance",
        "reranker_2.performance",
    ]
    if submitted != expected:
        raise AssertionError(f"Unexpected startup stage order: {submitted}")

    first_capacity_await = events.index(("await", "embedding.capacity", 30.0))
    last_capacity_submit = events.index(
        ("submit", "reranker_2.capacity", (6, 14, 8))
    )
    if last_capacity_submit > first_capacity_await:
        raise AssertionError("Maximum-capacity probes must be submitted concurrently.")

    performance_submissions = {
        event[1]: event[2]
        for event in events
        if event[0] == "submit" and str(event[1]).endswith(".performance")
    }
    if performance_submissions != {
        "embedding.performance": (4, 8, 2),
        "reranker_1.performance": (6, 14, 4, 2),
        "reranker_2.performance": (6, 14, 4, 2),
    }:
        raise AssertionError(
            "Unexpected representative warmup envelopes: "
            f"{performance_submissions}"
        )

    print("Ray model startup order -> ok")


def assert_capacity_failure_blocks_ready() -> None:
    report, error, events = _run_with_fakes(fail_capacity=True)
    if report is not None or not isinstance(error, model_actors.ModelWarmupError):
        raise AssertionError("Capacity failure must prevent a ready report.")
    submitted = [
        event[1]
        for event in events
        if event[0] == "submit"
    ]
    if "embedding.release" not in submitted or "reranker_2.release" not in submitted:
        raise AssertionError("Capacity failure should still release actor caches.")
    if any(name.endswith(".performance") for name in submitted):
        raise AssertionError("Performance warmup must not run after capacity failure.")

    print("Ray model capacity failure gate -> ok")


def main() -> None:
    assert_startup_order()
    assert_capacity_failure_blocks_ready()


if __name__ == "__main__":
    main()
