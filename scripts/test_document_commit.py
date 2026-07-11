from __future__ import annotations

from pathlib import Path
import sys
from threading import Event, Thread
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.document_commit import RetrievalRequestGate
from app.rag import RAGPipeline
from app.vector_store import SearchResult


def assert_gate_queues_and_drains_retrievals() -> None:
    gate = RetrievalRequestGate()
    active_started = Event()
    release_active = Event()
    exclusive_started = Event()
    release_exclusive = Event()
    queued_entered = Event()

    def active_retrieval() -> None:
        with gate.request():
            active_started.set()
            release_active.wait(2)

    def exclusive_commit() -> None:
        with gate.exclusive(drain_timeout_seconds=2):
            exclusive_started.set()
            release_exclusive.wait(2)

    def queued_retrieval() -> None:
        with gate.request():
            queued_entered.set()

    active = Thread(target=active_retrieval)
    commit = Thread(target=exclusive_commit)
    queued = Thread(target=queued_retrieval)
    active.start()
    if not active_started.wait(1):
        raise AssertionError("Active retrieval did not enter the gate.")
    commit.start()
    time.sleep(0.05)
    queued.start()
    time.sleep(0.05)
    snapshot = gate.snapshot()
    if not snapshot.blocked or snapshot.active_requests != 1:
        raise AssertionError(f"Commit gate did not drain correctly: {snapshot}")
    if snapshot.queued_requests != 1 or queued_entered.is_set():
        raise AssertionError("New retrieval should remain queued during commit.")

    release_active.set()
    if not exclusive_started.wait(1):
        raise AssertionError("Commit did not begin after active retrieval drained.")
    if queued_entered.is_set():
        raise AssertionError("Queued retrieval entered before pointer swap completed.")
    release_exclusive.set()
    if not queued_entered.wait(1):
        raise AssertionError("Queued retrieval was not released after commit.")

    for thread in (active, commit, queued):
        thread.join(1)
        if thread.is_alive():
            raise AssertionError("Document commit gate test thread did not stop.")

    print("Retrieval commit gate queue/drain -> ok")


def assert_reranker_and_llm_run_outside_gate() -> None:
    gate = RetrievalRequestGate()

    class GateAwareVectorStore:
        def search_bm25(self, _query: str, top_k: int) -> list[SearchResult]:
            del top_k
            if gate.snapshot().active_requests != 1:
                raise AssertionError("BM25 recall must run inside the retrieval gate.")
            return [SearchResult("context", "doc.md", 0, 1.0)]

        def search(self, _query: str, top_k: int) -> list[SearchResult]:
            del top_k
            if gate.snapshot().active_requests != 1:
                raise AssertionError("Qdrant recall must run inside the retrieval gate.")
            return [SearchResult("context", "doc.md", 0, 0.9)]

    class GateAwareReranker:
        def rerank(
            self,
            _query: str,
            contexts: list[SearchResult],
            top_k: int | None = None,
        ) -> list[SearchResult]:
            if gate.snapshot().active_requests:
                raise AssertionError("Reranking must run after releasing the gate.")
            return contexts[:top_k]

    class GateAwareLLM:
        def chat(self, _messages: list[dict[str, str]]) -> str:
            if gate.snapshot().active_requests:
                raise AssertionError("LLM generation must run outside the gate.")
            return "ok"

    pipeline = RAGPipeline(
        Settings(reranker_enabled=True),
        vector_store=GateAwareVectorStore(),  # type: ignore[arg-type]
        llm_client=GateAwareLLM(),  # type: ignore[arg-type]
        reranker=GateAwareReranker(),  # type: ignore[arg-type]
        retrieval_gate=gate,
    )
    answer = pipeline.answer("question", rag_only=True)
    if answer.answer != "ok" or not answer.contexts:
        raise AssertionError("Gate-aware RAG pipeline returned an invalid answer.")

    print("Reranker/LLM outside retrieval gate -> ok")


def main() -> None:
    assert_gate_queues_and_drains_retrievals()
    assert_reranker_and_llm_run_outside_gate()


if __name__ == "__main__":
    main()
