import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.s3_documents import VersionedS3DocumentIndexer
from app.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown documents into Qdrant.")
    parser.add_argument(
        "--source",
        choices=("local", "s3"),
        default=settings.docs_source,
        help="Document source to ingest. Defaults to DOCS_SOURCE.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Local mode: delete and recreate the collection. "
            "S3 mode: build a new version without copying unchanged chunks."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="S3 mode only: diff manifests without writing Qdrant or S3 manifests.",
    )
    args = parser.parse_args()

    if args.source == "s3":
        result = VersionedS3DocumentIndexer(settings).ingest(
            recreate=args.recreate,
            dry_run=args.dry_run,
        )
        if result.skipped:
            print(
                "No S3 document changes detected; active collection remains "
                f"{result.qdrant_collection}."
            )
            return
        mode = "Would build" if args.dry_run else "Built"
        print(
            f"{mode} document index version {result.index_version}: "
            f"alias={result.qdrant_alias} collection={result.qdrant_collection} "
            f"added_docs={result.added_docs} modified_docs={result.modified_docs} "
            f"deleted_docs={result.deleted_docs} unchanged_docs={result.unchanged_docs} "
            f"copied_chunks={result.copied_chunks} "
            f"embedded_chunks={result.embedded_chunks} "
            f"total_chunks={result.total_chunks}"
        )
        return

    store = VectorStore(settings)
    inserted = store.ingest_markdown_dir(recreate=args.recreate)
    if inserted == 0:
        print(f"No chunks generated from Markdown documents in {settings.docs_dir}")
        return
    print(f"Ingested {inserted} chunks into collection: {settings.collection_name}")


if __name__ == "__main__":
    main()
