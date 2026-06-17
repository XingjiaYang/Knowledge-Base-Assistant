from __future__ import annotations

from pathlib import Path
import sys
from threading import Lock, Thread
import time
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.rag import RAGPipeline
import app.vector_store as vector_store_module
from app.vector_store import SearchResult, VectorStore


class FakeEmbeddingModel:
    def get_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeRuntimeCudaEmbeddingModel(FakeEmbeddingModel):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.devices = [device]
        self.calls = 0

    def to(self, device: str) -> "FakeRuntimeCudaEmbeddingModel":
        self.device = device
        self.devices.append(device)
        return self

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self.calls += 1
        if self.device == "cuda":
            raise RuntimeError("CUDA encode failed")
        return super().encode(texts, normalize_embeddings=normalize_embeddings)


class CountingQdrantClient(QdrantClient):
    def __init__(self) -> None:
        super().__init__(":memory:")
        self.create_payload_index_calls = 0

    def create_payload_index(self, *args: object, **kwargs: object) -> object:
        self.create_payload_index_calls += 1
        return super().create_payload_index(*args, **kwargs)


class SequencingVectorStore(VectorStore):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []

    def _points_for_file(self, file_path: Path) -> list[object]:
        self.events.append(f"points:{file_path.name}")
        return super()._points_for_file(file_path)

    def _delete_file_points(self, file_path: Path) -> None:
        self.events.append(f"delete:{file_path.name}")
        super()._delete_file_points(file_path)


class FakeLLMClient:
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "ok"


def _records(client: QdrantClient, collection_name: str) -> list[Record]:
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    return records


def assert_incremental_ingest_replaces_source_points() -> None:
    with TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        file_path = docs_dir / "guide.md"
        collection_name = "incremental-ingest-test"
        client = QdrantClient(":memory:")
        config = Settings(
            docs_dir=docs_dir,
            collection_name=collection_name,
            chunk_size=80,
            chunk_overlap=10,
        )
        store = VectorStore(
            config,
            client=client,
            model=FakeEmbeddingModel(),
        )

        file_path.write_text(
            "# Guide\n\n" + " ".join(f"old-content-{idx}" for idx in range(40)),
            encoding="utf-8",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Payload indexes have no effect in the local Qdrant.*",
            )
            first_count = store.ingest_markdown_dir()
        if first_count < 2:
            raise AssertionError("Expected the initial document to produce multiple chunks.")
        first_records = _records(client, collection_name)
        first_ids = {record.id for record in first_records}
        if any(Path(record.payload["source"]).is_absolute() for record in first_records):
            raise AssertionError("Public source payload should not expose server paths.")
        if not all(record.payload.get("source_key") for record in first_records):
            raise AssertionError("Source key should be stored for internal replacement.")

        legacy_display_id = "00000000-0000-0000-0000-000000000001"
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=legacy_display_id,
                    vector=[1.0, 0.0],
                    payload={
                        "source": store._display_source(file_path),
                        "chunk_id": 99,
                        "text": "legacy display-source stale chunk",
                    },
                )
            ],
            wait=True,
        )

        file_path.write_text("# Guide\n\nCurrent content only.", encoding="utf-8")
        second_count = store.ingest_markdown_dir()
        records = _records(client, collection_name)
        if len(records) != second_count or second_count != 1:
            raise AssertionError("Updated document should replace all previous chunks.")
        if any("old-content" in str(record.payload) for record in records):
            raise AssertionError("Updated document left stale chunk content in Qdrant.")
        if any("legacy display-source" in str(record.payload) for record in records):
            raise AssertionError("Display-source legacy chunks should be deleted.")
        if records[0].id not in first_ids:
            raise AssertionError("Chunk point IDs should remain stable across content updates.")

        result = store._public_source(str(file_path))
        if Path(result).is_absolute() or str(docs_dir) in result:
            raise AssertionError("Legacy absolute sources should be sanitized on read.")

        file_path.write_text("", encoding="utf-8")
        empty_count = store.ingest_markdown_dir()
        if empty_count != 0 or _records(client, collection_name):
            raise AssertionError("Empty document should remove all previous source chunks.")

    print("Incremental ingest source replacement -> ok")


def assert_vector_store_lazy_loads_dependencies() -> None:
    store = VectorStore(Settings())
    if store._client is not None or store._model is not None:
        raise AssertionError("VectorStore should not connect or load models at init.")

    RAGPipeline(Settings(), vector_store=store, llm_client=FakeLLMClient())
    if store._client is not None or store._model is not None:
        raise AssertionError("RAGPipeline init should not force VectorStore loading.")

    print("VectorStore lazy loading -> ok")


def assert_vector_store_lazy_loads_once_under_concurrency() -> None:
    original_qdrant_client = vector_store_module.QdrantClient
    original_sentence_transformer = vector_store_module.SentenceTransformer
    counts = {"client": 0, "model": 0}
    count_lock = Lock()

    def fake_qdrant_client(*args: object, **kwargs: object) -> QdrantClient:
        time.sleep(0.01)
        with count_lock:
            counts["client"] += 1
        return QdrantClient(":memory:")

    def fake_sentence_transformer(*args: object, **kwargs: object) -> FakeEmbeddingModel:
        time.sleep(0.01)
        with count_lock:
            counts["model"] += 1
        return FakeEmbeddingModel()

    try:
        vector_store_module.QdrantClient = fake_qdrant_client
        vector_store_module.SentenceTransformer = fake_sentence_transformer
        store = VectorStore(Settings())

        clients: list[object] = []
        models: list[object] = []
        threads = [
            Thread(target=lambda: clients.append(store.client))
            for _ in range(8)
        ] + [
            Thread(target=lambda: models.append(store.model))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        vector_store_module.QdrantClient = original_qdrant_client
        vector_store_module.SentenceTransformer = original_sentence_transformer

    if counts != {"client": 1, "model": 1}:
        raise AssertionError(f"Lazy dependencies should load once: {counts}")
    if len({id(client) for client in clients}) != 1:
        raise AssertionError("Concurrent client access should reuse one instance.")
    if len({id(model) for model in models}) != 1:
        raise AssertionError("Concurrent model access should reuse one instance.")

    print("VectorStore concurrent lazy loading -> ok")


def assert_embedding_model_respects_cpu_override() -> None:
    original_sentence_transformer = vector_store_module.SentenceTransformer
    calls: list[dict[str, object]] = []

    def fake_sentence_transformer(*args: object, **kwargs: object) -> FakeEmbeddingModel:
        calls.append(dict(kwargs))
        return FakeEmbeddingModel()

    try:
        vector_store_module.SentenceTransformer = fake_sentence_transformer
        store = VectorStore(Settings(cuda_enabled=False))
        _ = store.model
    finally:
        vector_store_module.SentenceTransformer = original_sentence_transformer

    if calls != [{"device": "cpu"}]:
        raise AssertionError(f"Embedding model should load on CPU override: {calls}")

    print("Embedding model CPU override -> ok")


def assert_embedding_encode_falls_back_to_cpu_on_cuda_failure() -> None:
    original_sentence_transformer = vector_store_module.SentenceTransformer
    original_preferred_device = vector_store_module.preferred_torch_device
    models: list[FakeRuntimeCudaEmbeddingModel] = []

    def fake_sentence_transformer(
        *args: object,
        **kwargs: object,
    ) -> FakeRuntimeCudaEmbeddingModel:
        model = FakeRuntimeCudaEmbeddingModel(device=str(kwargs.get("device", "cpu")))
        models.append(model)
        return model

    try:
        vector_store_module.SentenceTransformer = fake_sentence_transformer
        vector_store_module.preferred_torch_device = lambda *_args: "cuda"
        store = VectorStore(Settings())
        embedding = store.encode("query")
    finally:
        vector_store_module.preferred_torch_device = original_preferred_device
        vector_store_module.SentenceTransformer = original_sentence_transformer

    if np.asarray(embedding).tolist() != [1.0, 0.0]:
        raise AssertionError("Embedding encode should retry successfully on CPU.")
    if models[0].devices != ["cuda", "cpu"]:
        raise AssertionError("Embedding model should move to CPU after CUDA failure.")
    if models[0].calls != 2:
        raise AssertionError("Embedding encode should retry exactly once.")

    print("Embedding CUDA encode fallback -> ok")


def assert_collection_setup_avoids_redundant_indexes() -> None:
    client = CountingQdrantClient()
    config = Settings(collection_name="index-test")
    store = VectorStore(
        config,
        client=client,
        model=FakeEmbeddingModel(),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Payload indexes have no effect in the local Qdrant.*",
        )
        store.ensure_collection()
        store.ensure_collection()
        if client.create_payload_index_calls != 1:
            raise AssertionError("Payload index should not be recreated every call.")

        store.ensure_collection(recreate=True)
        if client.create_payload_index_calls != 2:
            raise AssertionError("Payload index should be recreated after collection reset.")

    print("Payload index setup -> ok")


def assert_ingest_streams_files_and_skips_recreate_deletes() -> None:
    with TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        (docs_dir / "a.md").write_text("# A\n\nalpha beta gamma", encoding="utf-8")
        (docs_dir / "b.md").write_text("# B\n\none two three", encoding="utf-8")
        nested_dir = docs_dir / "topic"
        nested_dir.mkdir()
        (nested_dir / "c.md").write_text("# C\n\nnested topic", encoding="utf-8")
        client = CountingQdrantClient()
        store = SequencingVectorStore(
            Settings(
                docs_dir=docs_dir,
                collection_name="streaming-ingest-test",
                chunk_size=80,
                chunk_overlap=10,
            ),
            client=client,
            model=FakeEmbeddingModel(),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Payload indexes have no effect in the local Qdrant.*",
            )
            store.ingest_markdown_dir(recreate=True)
        if any(event.startswith("delete:") for event in store.events):
            raise AssertionError("Recreated collections should not delete per-file points.")

        store.events.clear()
        store.ingest_markdown_dir(recreate=False)
        expected_prefix = [
            "points:a.md",
            "delete:a.md",
            "points:b.md",
            "delete:b.md",
            "points:c.md",
            "delete:c.md",
        ]
        if store.events[:6] != expected_prefix:
            raise AssertionError(f"Ingest should process files incrementally: {store.events}")

        sources = {
            str(record.payload.get("source", ""))
            for record in _records(client, "streaming-ingest-test")
        }
        if "topic/c.md" not in sources:
            raise AssertionError(f"Recursive ingest should preserve nested source paths: {sources}")

    print("Streaming ingest behavior -> ok")


def assert_search_score_threshold() -> None:
    with TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        file_path = docs_dir / "guide.md"
        file_path.write_text("# Guide\n\nCurrent content only.", encoding="utf-8")
        client = QdrantClient(":memory:")
        store = VectorStore(
            Settings(
                docs_dir=docs_dir,
                collection_name="score-threshold-test",
                chunk_size=80,
                chunk_overlap=10,
                retrieve_score_threshold=1.1,
            ),
            client=client,
            model=FakeEmbeddingModel(),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Payload indexes have no effect in the local Qdrant.*",
            )
            store.ingest_markdown_dir(recreate=True)

        if store.search("anything"):
            raise AssertionError("Score threshold should filter low-scoring results.")

    print("Search score threshold -> ok")


def assert_bm25_recalls_keyword_matches() -> None:
    with TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        (docs_dir / "alpha.md").write_text(
            "# Alpha\n\nThe wall slogan says 巨大历史鲫鱼 and historical opportunity.",
            encoding="utf-8",
        )
        (docs_dir / "beta.md").write_text(
            "# Beta\n\nThis unrelated document discusses ordinary operations.",
            encoding="utf-8",
        )
        store = VectorStore(
            Settings(
                docs_dir=docs_dir,
                chunk_size=120,
                chunk_overlap=10,
                bm25_top_k=2,
            ),
            client=QdrantClient(":memory:"),
            model=FakeEmbeddingModel(),
        )

        results = store.search_bm25("巨大历史鲫鱼是什么梗？", top_k=1)
        if len(results) != 1 or results[0].source != "alpha.md":
            raise AssertionError(f"BM25 should recall the keyword-matched doc: {results}")
        if results[0].bm25_score is None or results[0].bm25_score <= 0:
            raise AssertionError("BM25 result should expose a positive BM25 score.")
        if results[0].retrieval_source != "bm25":
            raise AssertionError("BM25 result should record its retrieval source.")

    print("BM25 keyword recall -> ok")


def assert_rrf_fuses_vector_and_bm25_results() -> None:
    vector_results = [
        SearchResult(
            text="vector first",
            source="a.md",
            chunk_id=0,
            score=0.9,
            vector_score=0.9,
            retrieval_source="vector",
        ),
        SearchResult(
            text="shared",
            source="b.md",
            chunk_id=0,
            score=0.8,
            vector_score=0.8,
            retrieval_source="vector",
        ),
    ]
    bm25_results = [
        SearchResult(
            text="shared",
            source="b.md",
            chunk_id=0,
            score=4.0,
            bm25_score=4.0,
            retrieval_source="bm25",
        ),
        SearchResult(
            text="bm25 only",
            source="c.md",
            chunk_id=0,
            score=3.0,
            bm25_score=3.0,
            retrieval_source="bm25",
        ),
    ]

    fused = VectorStore._rrf_fuse(vector_results, bm25_results, top_k=2)
    if [result.source for result in fused] != ["b.md", "a.md"]:
        raise AssertionError(f"RRF should lift shared results: {fused}")
    shared = fused[0]
    if (
        shared.retrieval_source != "hybrid"
        or shared.vector_score != 0.8
        or shared.bm25_score != 4.0
        or shared.rrf_score is None
    ):
        raise AssertionError("RRF result should preserve hybrid score metadata.")

    print("RRF fusion -> ok")


def main() -> None:
    assert_vector_store_lazy_loads_dependencies()
    assert_vector_store_lazy_loads_once_under_concurrency()
    assert_embedding_model_respects_cpu_override()
    assert_embedding_encode_falls_back_to_cpu_on_cuda_failure()
    assert_collection_setup_avoids_redundant_indexes()
    assert_ingest_streams_files_and_skips_recreate_deletes()
    assert_search_score_threshold()
    assert_bm25_recalls_keyword_matches()
    assert_rrf_fuses_vector_and_bm25_results()
    assert_incremental_ingest_replaces_source_points()


if __name__ == "__main__":
    main()
