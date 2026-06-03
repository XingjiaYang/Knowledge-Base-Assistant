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
        first_ids = {record.id for record in _records(client, collection_name)}

        file_path.write_text("# Guide\n\nCurrent content only.", encoding="utf-8")
        second_count = store.ingest_markdown_dir()
        records = _records(client, collection_name)
        if len(records) != second_count or second_count != 1:
            raise AssertionError("Updated document should replace all previous chunks.")
        if any("old-content" in str(record.payload) for record in records):
            raise AssertionError("Updated document left stale chunk content in Qdrant.")
        if records[0].id not in first_ids:
            raise AssertionError("Chunk point IDs should remain stable across content updates.")

        file_path.write_text("", encoding="utf-8")
        empty_count = store.ingest_markdown_dir()
        if empty_count != 0 or _records(client, collection_name):
            raise AssertionError("Empty document should remove all previous source chunks.")

    print("Incremental ingest source replacement -> ok")


def main() -> None:
    assert_incremental_ingest_replaces_source_points()


if __name__ == "__main__":
    main()
