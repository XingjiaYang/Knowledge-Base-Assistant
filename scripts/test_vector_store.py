from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.rag import RAGPipeline
from app.vector_store import VectorStore


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

        file_path.write_text("# Guide\n\nCurrent content only.", encoding="utf-8")
        second_count = store.ingest_markdown_dir()
        records = _records(client, collection_name)
        if len(records) != second_count or second_count != 1:
            raise AssertionError("Updated document should replace all previous chunks.")
        if any("old-content" in str(record.payload) for record in records):
            raise AssertionError("Updated document left stale chunk content in Qdrant.")
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
        expected_prefix = ["points:a.md", "delete:a.md", "points:b.md", "delete:b.md"]
        if store.events[:4] != expected_prefix:
            raise AssertionError(f"Ingest should process files incrementally: {store.events}")

    print("Streaming ingest behavior -> ok")


def main() -> None:
    assert_vector_store_lazy_loads_dependencies()
    assert_collection_setup_avoids_redundant_indexes()
    assert_ingest_streams_files_and_skips_recreate_deletes()
    assert_incremental_ingest_replaces_source_points()


if __name__ == "__main__":
    main()
