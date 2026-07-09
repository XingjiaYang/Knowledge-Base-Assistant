import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.s3_documents import S3DocumentStore, VersionedS3DocumentIndexer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize S3-backed docs once for the current Docker image build."
    )
    parser.add_argument(
        "--build-id",
        required=True,
        help="Docker image build id baked into /app/.image_build_id.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=settings.docs_dir,
        help="Local Markdown directory to sync into S3 before indexing.",
    )
    parser.add_argument(
        "--delete-removed",
        action="store_true",
        help="Delete S3 Markdown objects that no longer exist locally.",
    )
    args = parser.parse_args()

    if settings.docs_source != "s3":
        print(f"Skipping build document initialization because DOCS_SOURCE={settings.docs_source}.")
        return

    document_store = S3DocumentStore(settings)
    marker = document_store.load_build_init_marker() or {}
    if marker.get("build_id") == args.build_id:
        print(f"S3 docs already initialized for image build {args.build_id}.")
        return

    sync_result = document_store.sync_local_directory(
        args.docs_dir,
        delete_removed=args.delete_removed,
    )
    ingest_result = VersionedS3DocumentIndexer(
        settings,
        document_store=document_store,
    ).ingest()
    document_store.write_build_init_marker(
        {
            "build_id": args.build_id,
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "docs_dir": str(args.docs_dir),
            "delete_removed": args.delete_removed,
            "sync": sync_result,
            "index_version": ingest_result.index_version,
            "qdrant_alias": ingest_result.qdrant_alias,
            "qdrant_collection": ingest_result.qdrant_collection,
            "skipped": ingest_result.skipped,
        }
    )
    if ingest_result.skipped:
        print(
            f"S3 docs synced for image build {args.build_id}; "
            "no index changes were detected."
        )
        return
    print(
        f"S3 docs initialized for image build {args.build_id}: "
        f"index_version={ingest_result.index_version} "
        f"collection={ingest_result.qdrant_collection} "
        f"added_docs={ingest_result.added_docs} "
        f"modified_docs={ingest_result.modified_docs} "
        f"deleted_docs={ingest_result.deleted_docs}"
    )


if __name__ == "__main__":
    main()
