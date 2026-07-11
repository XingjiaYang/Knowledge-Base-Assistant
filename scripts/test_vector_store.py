from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from threading import Lock, Thread
import time
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5
import warnings

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RAY_ENABLED", "0")

from app.config import Settings
from app.document_commit import PreparedIndexCommit, RetrievalRequestGate
from app.rag import RAGPipeline
from app.s3_documents import S3DocumentRecord, VersionedS3DocumentIndexer
import app.vector_store as vector_store_module
from app.vector_store import SearchResult, VectorStore


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.encode_kwargs: list[dict[str, object]] = []
        self.input_batch_sizes: list[int] = []

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        **kwargs: object,
    ) -> np.ndarray:
        self.encode_kwargs.append(dict(kwargs))
        self.input_batch_sizes.append(1 if isinstance(texts, str) else len(texts))
        if isinstance(texts, str):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeChunkTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


FAKE_CHUNK_TOKENIZER = FakeChunkTokenizer()


class FakeRuntimeCudaEmbeddingModel(FakeEmbeddingModel):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
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
        **kwargs: object,
    ) -> np.ndarray:
        self.calls += 1
        if self.device == "cuda":
            raise RuntimeError("CUDA encode failed")
        return super().encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )


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


class MissingRayActorVectorStore(VectorStore):
    def _embedding_actor(self) -> object | None:
        return None


class FakeLLMClient:
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "ok"


class FakeS3DocumentStore:
    def __init__(self, documents: dict[str, tuple[str, str]]) -> None:
        self.active_manifest: dict[str, object] | None = None
        self.version_manifests: list[dict[str, object]] = []
        self.prune_calls: list[int] = []
        self.fail_active_manifest_write = False
        self.set_documents(documents)

    def set_documents(self, documents: dict[str, tuple[str, str]]) -> None:
        self.texts: dict[str, str] = {}
        self.records: list[S3DocumentRecord] = []
        for source, (version_id, text) in sorted(documents.items()):
            key = f"docs/{source}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.texts[key] = text
            self.records.append(
                S3DocumentRecord(
                    source_doc_id=str(uuid5(NAMESPACE_URL, f"s3://bucket/{key}")),
                    key=key,
                    source=source,
                    version_id=version_id,
                    etag=version_id,
                    size=len(text.encode("utf-8")),
                    last_modified="2026-01-01T00:00:00+00:00",
                    content_hash=content_hash,
                )
            )

    def ensure_versioning_enabled(self) -> None:
        return None

    def list_current_markdown(self) -> list[S3DocumentRecord]:
        return list(self.records)

    def read_markdown(self, record: S3DocumentRecord) -> str:
        return self.texts[record.key]

    def load_active_manifest(self) -> dict[str, object] | None:
        return self.active_manifest

    def load_version_manifest(self, index_version: str) -> dict[str, object] | None:
        for manifest in reversed(self.version_manifests):
            if manifest.get("index_version") == index_version:
                return manifest
        return None

    def write_active_manifest(self, manifest: dict[str, object]) -> None:
        if self.fail_active_manifest_write:
            raise RuntimeError("active manifest write failed")
        self.active_manifest = dict(manifest)

    def delete_active_manifest(self) -> None:
        self.active_manifest = None

    def write_version_manifest(self, manifest: dict[str, object]) -> None:
        self.version_manifests.append(dict(manifest))

    def prune_document_versions(self, retain_versions: int) -> dict[str, int]:
        self.prune_calls.append(retain_versions)
        return {
            "keys": len(self.records),
            "deleted_versions": 0,
            "retain_versions": retain_versions,
        }

    def find_manifest_by_collection(self, collection_name: str) -> dict[str, object] | None:
        for manifest in reversed(self.version_manifests):
            if manifest.get("qdrant_collection") == collection_name:
                return manifest
        return None


class FailingQdrantClient:
    def __getattr__(self, name: str) -> object:
        raise RuntimeError(f"Qdrant should not be used during this test: {name}")


class InProcessDocumentCommitter:
    def __init__(
        self,
        indexer: VersionedS3DocumentIndexer,
        vector_store: VectorStore,
    ) -> None:
        self.indexer = indexer
        self.vector_store = vector_store
        self.gate = RetrievalRequestGate()
        self.events: list[str] = []

    def prepare(self, index_version: str) -> int:
        self.events.append(f"prepare:{index_version}")
        return self.vector_store.prepare_bm25_candidate(index_version)

    def commit(self, prepared: PreparedIndexCommit) -> str:
        self.events.append(f"commit:{prepared.index_version}")
        with self.gate.exclusive(drain_timeout_seconds=1):
            result = self.indexer.commit_prepared_version(prepared)
        return str(result["previous_collection"])

    def discard(self, index_version: str) -> None:
        self.events.append(f"discard:{index_version}")
        self.vector_store.discard_bm25_candidate(index_version)


def _records(client: QdrantClient, collection_name: str) -> list[Record]:
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    return records


def _alias_target(client: QdrantClient, alias_name: str) -> str:
    for alias in client.get_aliases().aliases:
        if alias.alias_name == alias_name:
            return alias.collection_name
    return ""


def _source_doc_ids(client: QdrantClient, collection_name: str) -> set[str]:
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    return {
        str(record.payload.get("source_doc_id", ""))
        for record in records
        if record.payload
    }


def assert_incremental_ingest_replaces_source_points() -> None:
    with TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        file_path = docs_dir / "guide.md"
        collection_name = "incremental-ingest-test"
        client = QdrantClient(":memory:")
        config = Settings(
            docs_dir=docs_dir,
            collection_name=collection_name,
            chunk_body_target_tokens=50,
            chunk_body_max_tokens=60,
            chunk_overlap_target_tokens=10,
            chunk_overlap_max_tokens=10,
        )
        store = VectorStore(
            config,
            client=client,
            model=FakeEmbeddingModel(),
            chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
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
        if any(
            not record.payload
            or int(record.payload.get("body_token_count", 0)) <= 0
            or int(record.payload.get("token_count", 0)) <= 0
            for record in first_records
        ):
            raise AssertionError("Qdrant payloads should expose chunk token counts.")
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

    if calls != [{"device": "cpu", "trust_remote_code": True}]:
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


def assert_ray_embedding_does_not_fallback_locally_when_disabled() -> None:
    store = MissingRayActorVectorStore(
        Settings(ray_enabled=True, ray_local_fallback=False),
        use_ray=True,
    )
    try:
        store.encode("query")
    except RuntimeError as exc:
        if "RAY_LOCAL_FALLBACK=0" not in str(exc):
            raise
    else:
        raise AssertionError(
            "VectorStore should not load a local model when Ray fallback is disabled."
        )

    print("Ray embedding local fallback disabled -> ok")


def assert_jina_embedding_tasks_are_routed_by_use_case() -> None:
    model = FakeEmbeddingModel()
    store = VectorStore(
        Settings(
            embedding_query_task="retrieval",
            embedding_passage_task="retrieval",
            embedding_classification_task="classification",
            embedding_query_prompt_name="query",
            embedding_passage_prompt_name="document",
        ),
        model=model,
    )

    _ = store._embed_one("query")
    _ = store.encode("intent text")
    _ = store._encode(
        ["chunk text"],
        task=store.config.embedding_passage_task,
        prompt_name=store.config.embedding_passage_prompt_name,
    )

    expected = [
        {"task": "retrieval", "prompt_name": "query"},
        {"task": "classification"},
        {"task": "retrieval", "prompt_name": "document", "batch_size": 1},
    ]
    if model.encode_kwargs != expected:
        raise AssertionError(f"Embedding task routing mismatch: {model.encode_kwargs}")

    print("Jina embedding task routing -> ok")


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
                chunk_body_target_tokens=50,
                chunk_body_max_tokens=60,
                chunk_overlap_target_tokens=10,
                chunk_overlap_max_tokens=10,
            ),
            client=client,
            model=FakeEmbeddingModel(),
            chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
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
                chunk_body_target_tokens=50,
                chunk_body_max_tokens=60,
                chunk_overlap_target_tokens=10,
                chunk_overlap_max_tokens=10,
                retrieve_score_threshold=1.1,
            ),
            client=client,
            model=FakeEmbeddingModel(),
            chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
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
                chunk_body_target_tokens=80,
                chunk_body_max_tokens=100,
                chunk_overlap_target_tokens=10,
                chunk_overlap_max_tokens=10,
                bm25_top_k=2,
            ),
            client=QdrantClient(":memory:"),
            model=FakeEmbeddingModel(),
            chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
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


def assert_s3_versioned_ingest_switches_alias_and_removes_orphans() -> None:
    client = QdrantClient(":memory:")
    config = Settings(
        docs_source="s3",
        docs_s3_bucket="bucket",
        docs_s3_prefix="docs",
        docs_s3_require_versioning=False,
        collection_name="s3_docs_test",
        chunk_body_target_tokens=160,
        chunk_body_max_tokens=200,
        chunk_overlap_target_tokens=20,
        chunk_overlap_max_tokens=20,
        embedding_offline_batch_size=2,
    )
    document_store = FakeS3DocumentStore(
        {
            "a.md": ("a-v1", "# A\n\nalpha orphan text"),
            "b.md": ("b-v1", "# B\n\nbeta stable text"),
            "c.md": ("c-v1", "# C\n\ncharlie old text"),
        }
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = VectorStore(
        config,
        client=client,
        model=embedding_model,
        chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
        document_store=document_store,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Payload indexes have no effect in the local Qdrant.*",
        )
        vector_store.ensure_collection(recreate=True)
    client.upsert(
        collection_name="s3_docs_test",
        points=[
            PointStruct(
                id="00000000-0000-0000-0000-000000000010",
                vector=[1.0, 0.0],
                payload={
                    "source": "legacy.md",
                    "chunk_id": 0,
                    "text": "legacy pre-S3 chunk",
                },
            )
        ],
        wait=True,
    )
    indexer = VersionedS3DocumentIndexer(
        config,
        document_store=document_store,
        vector_store=vector_store,
    )

    first = indexer.ingest()
    first_collection = _alias_target(client, "s3_docs_test")
    if not first_collection or first_collection != first.qdrant_collection:
        raise AssertionError("S3 ingest should create and point the alias to v1.")
    if first.total_chunks != 3:
        raise AssertionError(f"Expected one chunk per initial document: {first}")
    if embedding_model.input_batch_sizes[-2:] != [2, 1]:
        raise AssertionError(
            "S3 ingest should batch chunks across document boundaries: "
            f"{embedding_model.input_batch_sizes}"
        )
    manifest = document_store.active_manifest or {}
    expected_chunk_config = {
        "chunking_version": vector_store_module.CHUNKING_VERSION,
        "chunk_tokenizer_model": "jinaai/jina-embeddings-v5-text-small",
        "chunk_tokenizer_trust_remote_code": True,
        "chunk_body_target_tokens": 160,
        "chunk_body_max_tokens": 200,
        "chunk_overlap_target_tokens": 20,
        "chunk_overlap_max_tokens": 20,
    }
    if any(manifest.get(key) != value for key, value in expected_chunk_config.items()):
        raise AssertionError(
            f"S3 manifest should fingerprint chunking behavior: {manifest}"
        )
    original_signature = vector_store._s3_docs_signature()  # noqa: SLF001
    document_store.active_manifest = {**manifest, "chunk_body_target_tokens": 161}
    changed_signature = vector_store._s3_docs_signature()  # noqa: SLF001
    document_store.active_manifest = manifest
    if changed_signature == original_signature:
        raise AssertionError("BM25 signature should change with chunk configuration.")
    collection_names = {collection.name for collection in client.get_collections().collections}
    if not any(name.startswith("s3_docs_test__vlegacy_") for name in collection_names):
        raise AssertionError("Existing physical collection should be archived before alias creation.")

    document_store.set_documents(
        {
            "b.md": ("b-v1", "# B\n\nbeta stable text"),
            "c.md": ("c-v2", "# C\n\ncharlie modified text"),
            "d.md": ("d-v1", "# D\n\ndelta new text"),
        }
    )
    second = indexer.ingest()
    second_collection = _alias_target(client, "s3_docs_test")
    if second_collection == first_collection or second_collection != second.qdrant_collection:
        raise AssertionError("S3 ingest should atomically switch the alias to v2.")
    if (
        second.added_docs != 1
        or second.modified_docs != 1
        or second.deleted_docs != 1
        or second.unchanged_docs != 1
        or second.copied_chunks != 1
        or second.embedded_chunks != 2
    ):
        raise AssertionError(f"Unexpected S3 diff/index stats: {second}")

    source_ids = _source_doc_ids(client, second_collection)
    current_by_source = {record.source: record.source_doc_id for record in document_store.records}
    deleted_a_id = str(uuid5(NAMESPACE_URL, "s3://bucket/docs/a.md"))
    if deleted_a_id in source_ids:
        raise AssertionError("Deleted S3 documents must not leave orphan chunks.")
    for source in ("b.md", "c.md", "d.md"):
        if current_by_source[source] not in source_ids:
            raise AssertionError(f"Current S3 document {source} is missing chunks.")

    deleted_results = vector_store.search_bm25("alpha orphan", top_k=5)
    if any(result.source == "a.md" for result in deleted_results):
        raise AssertionError("BM25 should rebuild from the active alias without deleted docs.")
    fresh_results = vector_store.search_bm25("delta new", top_k=1)
    if not fresh_results or fresh_results[0].source != "d.md":
        raise AssertionError(f"BM25 should search the active S3 version: {fresh_results}")
    offline_store = VectorStore(
        config,
        client=FailingQdrantClient(),  # type: ignore[arg-type]
        model=FakeEmbeddingModel(),
        chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
        document_store=document_store,
    )
    offline_results = offline_store.search_bm25("delta new", top_k=1)
    if not offline_results or offline_results[0].source != "d.md":
        raise AssertionError(
            "BM25 should build from S3 active manifest without Qdrant: "
            f"{offline_results}"
        )

    document_store.set_documents(
        {
            "b.md": ("b-v2", "# B\n\nbeta changed again"),
            "c.md": ("c-v2", "# C\n\ncharlie modified text"),
            "d.md": ("d-v1", "# D\n\ndelta new text"),
        }
    )
    third = indexer.ingest()
    third_collection = _alias_target(client, "s3_docs_test")
    if third_collection != third.qdrant_collection:
        raise AssertionError("Third S3 ingest should switch to the latest version.")
    retained_versions = [
        collection.name
        for collection in client.get_collections().collections
        if collection.name.startswith("s3_docs_test__v")
        and not collection.name.startswith("s3_docs_test__vlegacy_")
    ]
    if len(retained_versions) != 2:
        raise AssertionError(f"Qdrant should retain exactly two S3 versions: {retained_versions}")
    if document_store.prune_calls != [6, 5, 6, 5, 6, 5]:
        raise AssertionError(f"S3 pruning should use processing/stable limits: {document_store.prune_calls}")

    print("S3 versioned ingest alias/orphan behavior -> ok")


def assert_s3_remote_commit_swaps_prebuilt_bm25() -> None:
    client = QdrantClient(":memory:")
    config = Settings(
        docs_source="s3",
        docs_s3_bucket="bucket",
        docs_s3_prefix="docs",
        docs_s3_require_versioning=False,
        collection_name="s3_remote_commit_test",
        chunk_body_target_tokens=160,
        chunk_body_max_tokens=200,
        chunk_overlap_target_tokens=20,
        chunk_overlap_max_tokens=20,
        embedding_offline_batch_size=2,
    )
    document_store = FakeS3DocumentStore(
        {
            "a.md": ("a-v1", "# A\n\nalpha old text"),
            "b.md": ("b-v1", "# B\n\nbeta stable text"),
        }
    )
    offline_store = VectorStore(
        config,
        client=client,
        model=FakeEmbeddingModel(),
        chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
        document_store=document_store,
    )
    offline_indexer = VersionedS3DocumentIndexer(
        config,
        document_store=document_store,
        vector_store=offline_store,
    )
    first = offline_indexer.ingest()

    online_store = VectorStore(
        config,
        client=client,
        model=FakeEmbeddingModel(),
        chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
        document_store=document_store,
    )
    online_indexer = VersionedS3DocumentIndexer(
        config,
        document_store=document_store,
        vector_store=online_store,
    )
    online_store.rebuild_bm25_index(expected_index_version=first.index_version)
    old_index = online_store._bm25_index  # noqa: SLF001

    document_store.set_documents(
        {
            "b.md": ("b-v1", "# B\n\nbeta stable text"),
            "c.md": ("c-v1", "# C\n\ngamma candidate text"),
        }
    )
    committer = InProcessDocumentCommitter(online_indexer, online_store)
    second = offline_indexer.ingest(committer=committer)

    if _alias_target(client, config.collection_name) != second.qdrant_collection:
        raise AssertionError("Remote commit should switch the Qdrant alias.")
    if online_store.active_bm25_index_version() != second.index_version:
        raise AssertionError("Remote commit should activate the matching BM25S index.")
    if online_store.prepared_bm25_candidate_version():
        raise AssertionError("BM25S candidate should be cleared after pointer swap.")
    if old_index is None or old_index is online_store._bm25_index:  # noqa: SLF001
        raise AssertionError("BM25S commit should exchange active index references.")
    if not old_index.documents:
        raise AssertionError("The old BM25S index reference should remain usable.")
    if [event.split(":", 1)[0] for event in committer.events] != [
        "prepare",
        "commit",
    ]:
        raise AssertionError(f"Unexpected commit lifecycle: {committer.events}")
    results = online_store.search_bm25("gamma candidate", top_k=1)
    if not results or results[0].source != "c.md":
        raise AssertionError(f"Swapped BM25S index returned stale data: {results}")

    print("S3 candidate BM25S double-buffer commit -> ok")


def assert_s3_versioned_ingest_rolls_back_on_failure() -> None:
    client = QdrantClient(":memory:")
    config = Settings(
        docs_source="s3",
        docs_s3_bucket="bucket",
        docs_s3_prefix="docs",
        docs_s3_require_versioning=False,
        collection_name="s3_rollback_test",
        chunk_body_target_tokens=160,
        chunk_body_max_tokens=200,
        chunk_overlap_target_tokens=20,
        chunk_overlap_max_tokens=20,
    )
    document_store = FakeS3DocumentStore(
        {"a.md": ("a-v1", "# A\n\nalpha initial")}
    )
    vector_store = VectorStore(
        config,
        client=client,
        model=FakeEmbeddingModel(),
        chunk_tokenizer=FAKE_CHUNK_TOKENIZER,
        document_store=document_store,
    )
    indexer = VersionedS3DocumentIndexer(
        config,
        document_store=document_store,
        vector_store=vector_store,
    )

    first = indexer.ingest()
    first_collection = _alias_target(client, "s3_rollback_test")
    if first_collection != first.qdrant_collection:
        raise AssertionError("Initial S3 rollback test ingest should set the alias.")

    document_store.set_documents({"a.md": ("a-v2", "# A\n\nalpha changed")})
    document_store.fail_active_manifest_write = True
    try:
        indexer.ingest()
    except RuntimeError as exc:
        if "active manifest write failed" not in str(exc):
            raise
    else:
        raise AssertionError("S3 ingest should fail when active manifest write fails.")

    if _alias_target(client, "s3_rollback_test") != first_collection:
        raise AssertionError("Failed S3 ingest should roll alias back to the previous collection.")
    version_collections = [
        collection.name
        for collection in client.get_collections().collections
        if collection.name.startswith("s3_rollback_test__v")
    ]
    if version_collections != [first_collection]:
        raise AssertionError(f"Failed S3 ingest should delete the new collection: {version_collections}")

    print("S3 versioned ingest rollback -> ok")


def main() -> None:
    assert_vector_store_lazy_loads_dependencies()
    assert_vector_store_lazy_loads_once_under_concurrency()
    assert_embedding_model_respects_cpu_override()
    assert_embedding_encode_falls_back_to_cpu_on_cuda_failure()
    assert_ray_embedding_does_not_fallback_locally_when_disabled()
    assert_jina_embedding_tasks_are_routed_by_use_case()
    assert_collection_setup_avoids_redundant_indexes()
    assert_ingest_streams_files_and_skips_recreate_deletes()
    assert_search_score_threshold()
    assert_bm25_recalls_keyword_matches()
    assert_rrf_fuses_vector_and_bm25_results()
    assert_incremental_ingest_replaces_source_points()
    assert_s3_versioned_ingest_switches_alias_and_removes_orphans()
    assert_s3_remote_commit_swaps_prebuilt_bm25()
    assert_s3_versioned_ingest_rolls_back_on_failure()


if __name__ == "__main__":
    main()
