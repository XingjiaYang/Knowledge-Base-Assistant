from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RAY_ENABLED", "0")

from app.config import Settings
import app.reranker as reranker_module
from app.reranker import Reranker
from app.vector_store import SearchResult


class FakeRerankerModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append((query, len(documents), top_n))
        if "Headings: Guide > Setup" not in documents[0]:
            if query != "health check":
                raise AssertionError("Document text should include heading context.")
            return [{"index": 0, "relevance_score": 1.0}]

        return [
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]


class FakeBatchRerankerModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append((query, len(documents), top_n))
        results = []
        for index, document in enumerate(documents):
            marker = document.rsplit("score=", 1)[-1]
            results.append(
                {
                    "index": index,
                    "relevance_score": float(marker),
                }
            )
        return sorted(
            results,
            key=lambda result: result["relevance_score"],
            reverse=True,
        )


class FakeDeviceModel:
    def __init__(self) -> None:
        self.devices: list[str] = []
        self.eval_called = False

    def to(self, device: str) -> "FakeDeviceModel":
        self.devices.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA failed")
        return self

    def eval(self) -> "FakeDeviceModel":
        self.eval_called = True
        return self

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        return [{"index": 0, "relevance_score": 1.0}]


class FakeRuntimeCudaRerankerModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.devices: list[str] = []
        self.calls = 0

    def to(self, device: str) -> "FakeRuntimeCudaRerankerModel":
        self.device = device
        self.devices.append(device)
        return self

    def eval(self) -> "FakeRuntimeCudaRerankerModel":
        return self

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        self.calls += 1
        if self.device == "cuda":
            raise RuntimeError("CUDA inference failed")
        return [{"index": 0, "relevance_score": 1.0}]


class MissingRayActorReranker(Reranker):
    def _reranker_actor(self) -> object | None:
        return None


def assert_reranker_orders_by_model_score() -> None:
    model = FakeRerankerModel()
    reranker = Reranker(Settings(), model=model)
    contexts = [
        SearchResult(
            text="First candidate",
            source="data/docs/guide.md",
            chunk_id=1,
            score=0.99,
            headings=("Guide", "Setup"),
        ),
        SearchResult(
            text="Best candidate",
            source="data/docs/guide.md",
            chunk_id=2,
            score=0.70,
        ),
        SearchResult(
            text="Middle candidate",
            source="data/docs/guide.md",
            chunk_id=3,
            score=0.80,
        ),
    ]

    reranked = reranker.rerank("How do I set up the system?", contexts, top_k=2)
    if [context.chunk_id for context in reranked] != [2, 3]:
        raise AssertionError("Reranker should sort by rerank score.")
    if reranked[0].rerank_score is None or reranked[0].score != 0.70:
        raise AssertionError("Reranker should preserve vector score and add rerank score.")
    if model.calls != [("How do I set up the system?", 3, 2)]:
        raise AssertionError("Reranker should pass query, documents, and top_n.")

    print("Reranker ordering -> ok")


def assert_reranker_batches_large_recalls() -> None:
    model = FakeBatchRerankerModel()
    reranker = Reranker(
        Settings(reranker_max_documents_per_call=2),
        model=model,
    )
    contexts = [
        SearchResult(
            text=f"candidate score={score}",
            source="data/docs/guide.md",
            chunk_id=index,
            score=0.1,
        )
        for index, score in enumerate([0.1, 0.9, 0.4, 0.7, 0.2])
    ]

    reranked = reranker.rerank("batch query", contexts, top_k=3)
    if [context.chunk_id for context in reranked] != [1, 3, 2]:
        raise AssertionError("Batched reranking should sort globally by score.")
    if model.calls != [
        ("batch query", 2, None),
        ("batch query", 2, None),
        ("batch query", 1, None),
    ]:
        raise AssertionError("Large recalls should rerank in bounded batches.")

    print("Reranker batching -> ok")


def assert_warmup_uses_rerank() -> None:
    model = FakeRerankerModel()
    reranker = Reranker(Settings(), model=model)
    reranker.warmup()
    if model.calls != [("health check", 1, 1)]:
        raise AssertionError("Warmup should run a one-document rerank call.")

    print("Reranker warmup -> ok")


def assert_reranker_falls_back_to_cpu_when_cuda_move_fails() -> None:
    original_transformers = sys.modules.get("transformers")
    original_preferred_device = reranker_module.preferred_torch_device
    loaded_models: list[FakeDeviceModel] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeDeviceModel:
            model = FakeDeviceModel()
            loaded_models.append(model)
            return model

    sys.modules["transformers"] = SimpleNamespace(AutoModel=FakeAutoModel)
    reranker_module.preferred_torch_device = lambda *_args: "cuda"
    try:
        reranker = Reranker(Settings())
        model = reranker.model
    finally:
        reranker_module.preferred_torch_device = original_preferred_device
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers

    if model is not loaded_models[0]:
        raise AssertionError("Reranker should keep the loaded model instance.")
    if loaded_models[0].devices != ["cuda", "cpu"]:
        raise AssertionError("Reranker should fall back to CPU after CUDA failure.")
    if not loaded_models[0].eval_called:
        raise AssertionError("Reranker model should be put in eval mode.")

    print("Reranker CUDA fallback -> ok")


def assert_reranker_falls_back_to_cpu_when_cuda_inference_fails() -> None:
    original_transformers = sys.modules.get("transformers")
    original_preferred_device = reranker_module.preferred_torch_device
    loaded_models: list[FakeRuntimeCudaRerankerModel] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(
            *args: object,
            **kwargs: object,
        ) -> FakeRuntimeCudaRerankerModel:
            model = FakeRuntimeCudaRerankerModel()
            loaded_models.append(model)
            return model

    sys.modules["transformers"] = SimpleNamespace(AutoModel=FakeAutoModel)
    reranker_module.preferred_torch_device = lambda *_args: "cuda"
    try:
        reranker = Reranker(Settings())
        contexts = [
            SearchResult(
                text="candidate",
                source="data/docs/guide.md",
                chunk_id=1,
                score=0.9,
            )
        ]
        reranked = reranker.rerank("query", contexts, top_k=1)
    finally:
        reranker_module.preferred_torch_device = original_preferred_device
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers

    if [context.chunk_id for context in reranked] != [1]:
        raise AssertionError("Reranker should retry on CPU after CUDA inference failure.")
    if loaded_models[0].devices != ["cuda", "cpu"]:
        raise AssertionError("Reranker should move to CPU after CUDA inference failure.")
    if loaded_models[0].calls != 2:
        raise AssertionError("Reranker should retry exactly once after CUDA failure.")

    print("Reranker CUDA inference fallback -> ok")


def assert_ray_reranker_does_not_fallback_locally_when_disabled() -> None:
    reranker = MissingRayActorReranker(
        Settings(ray_enabled=True, ray_local_fallback=False),
    )
    contexts = [
        SearchResult(
            text="candidate",
            source="data/docs/guide.md",
            chunk_id=1,
            score=0.9,
        )
    ]
    try:
        reranker.rerank("query", contexts, top_k=1)
    except RuntimeError as exc:
        if "RAY_LOCAL_FALLBACK=0" not in str(exc):
            raise
    else:
        raise AssertionError(
            "Reranker should not load a local model when Ray fallback is disabled."
        )

    print("Ray reranker local fallback disabled -> ok")


def main() -> None:
    assert_reranker_orders_by_model_score()
    assert_reranker_batches_large_recalls()
    assert_warmup_uses_rerank()
    assert_reranker_falls_back_to_cpu_when_cuda_move_fails()
    assert_reranker_falls_back_to_cpu_when_cuda_inference_fails()
    assert_ray_reranker_does_not_fallback_locally_when_disabled()


if __name__ == "__main__":
    main()
