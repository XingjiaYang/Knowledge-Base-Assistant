import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.document_commit import HTTPDocumentIndexCommitter
from app.model_actors import destroy_embedding_actor
from app.s3_documents import S3DocumentStore, VersionedS3DocumentIndexer
from app.vector_store import CHUNKING_VERSION


def _index_config_fingerprint() -> str:
    payload = {
        "embedding_model": settings.embedding_model,
        "embedding_passage_task": settings.embedding_passage_task,
        "embedding_passage_prompt_name": settings.embedding_passage_prompt_name,
        "chunking_version": CHUNKING_VERSION,
        "chunk_tokenizer_model": settings.chunk_tokenizer_model,
        "chunk_tokenizer_trust_remote_code": (
            settings.chunk_tokenizer_trust_remote_code
        ),
        "chunk_body_target_tokens": settings.chunk_body_target_tokens,
        "chunk_body_max_tokens": settings.chunk_body_max_tokens,
        "chunk_overlap_target_tokens": settings.chunk_overlap_target_tokens,
        "chunk_overlap_max_tokens": settings.chunk_overlap_max_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize S3-backed docs once for the current Docker image build."
    )
    build_id_group = parser.add_mutually_exclusive_group(required=True)
    build_id_group.add_argument(
        "--build-id",
        help="Docker image build id baked into /app/.image_build_id.",
    )
    build_id_group.add_argument(
        "--build-id-file",
        type=Path,
        help="Read the Docker image build id from this file.",
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
        default=os.getenv("DOCS_INIT_DELETE_REMOVED", "0").lower()
        in {"1", "true", "yes", "on"},
        help="Delete S3 Markdown objects that no longer exist locally.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.getenv("DOCS_BOOTSTRAP_FORCE", "0").lower()
        in {"1", "true", "yes", "on"},
        help="Run S3 sync/index diff even when startup fingerprints are unchanged.",
    )
    parser.add_argument(
        "--destroy-embedding-actor-on-exit",
        action="store_true",
        help=(
            "Permanently remove the configured detached Ray embedding actor before "
            "the one-shot indexer exits. Use only with a dedicated offline actor name."
        ),
    )
    args = parser.parse_args()
    build_id = (
        args.build_id_file.read_text(encoding="utf-8").strip()
        if args.build_id_file is not None
        else str(args.build_id).strip()
    )
    if not build_id:
        raise ValueError("Docker image build id must not be empty.")

    try:
        _initialize_documents(args, build_id)
    finally:
        if args.destroy_embedding_actor_on_exit:
            destroyed = destroy_embedding_actor(settings)
            print(
                "Offline embedding actor cleanup: "
                f"name={settings.ray_embedding_actor_name} destroyed={destroyed}"
            )


def _initialize_documents(args: argparse.Namespace, build_id: str) -> None:
    if settings.docs_source != "s3":
        print(
            "Skipping build document initialization because "
            f"DOCS_SOURCE={settings.docs_source}."
        )
        return

    document_store = S3DocumentStore(settings)
    marker = document_store.load_build_init_marker() or {}
    index_config_fingerprint = _index_config_fingerprint()
    if (
        not args.force
        and marker.get("build_id") == build_id
        and marker.get("index_config_fingerprint") == index_config_fingerprint
    ):
        print(f"S3 docs already initialized for image build {build_id}.")
        return

    sync_result = document_store.sync_local_directory(
        args.docs_dir,
        delete_removed=args.delete_removed,
    )
    committer = (
        HTTPDocumentIndexCommitter(
            settings.docs_commit_url,
            settings.docs_commit_token,
            timeout_seconds=settings.docs_commit_http_timeout_seconds,
        )
        if settings.docs_commit_url.strip()
        else None
    )
    ingest_result = VersionedS3DocumentIndexer(
        settings,
        document_store=document_store,
    ).ingest(committer=committer)
    document_store.write_build_init_marker(
        {
            "build_id": build_id,
            "index_config_fingerprint": index_config_fingerprint,
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
            f"S3 docs synced for image build {build_id}; "
            "no index changes were detected."
        )
        return
    print(
        f"S3 docs initialized for image build {build_id}: "
        f"index_version={ingest_result.index_version} "
        f"collection={ingest_result.qdrant_collection} "
        f"added_docs={ingest_result.added_docs} "
        f"modified_docs={ingest_result.modified_docs} "
        f"deleted_docs={ingest_result.deleted_docs}"
    )


if __name__ == "__main__":
    main()
