from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
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


def main() -> None:
    assert_reranker_orders_by_model_score()
    assert_reranker_batches_large_recalls()
    assert_warmup_uses_rerank()


if __name__ == "__main__":
    main()
