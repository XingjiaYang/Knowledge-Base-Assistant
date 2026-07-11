from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import mimetypes
from pathlib import Path
import posixpath
import re
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import Settings, settings
from app.document_commit import DocumentIndexCommitter, PreparedIndexCommit
from app.vector_store import (
    CHUNKING_VERSION,
    MarkdownChunk,
    VectorStore,
    chunk_markdown,
)


logger = logging.getLogger(__name__)

_MANIFEST_SCHEMA_VERSION = 4
_S3_METADATA_SHA256 = "content-sha256"


@dataclass(frozen=True)
class S3DocumentRecord:
    source_doc_id: str
    key: str
    source: str
    version_id: str
    etag: str
    size: int
    last_modified: str
    content_hash: str = ""


@dataclass(frozen=True)
class S3IngestResult:
    index_version: str
    qdrant_alias: str
    qdrant_collection: str
    previous_collection: str
    added_docs: int
    modified_docs: int
    deleted_docs: int
    unchanged_docs: int
    copied_chunks: int
    embedded_chunks: int
    total_chunks: int
    skipped: bool = False


class S3DocumentStore:
    def __init__(self, config: Settings = settings, client: object | None = None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> object:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for DOCS_SOURCE=s3. "
                    "Install requirements.api.txt."
                ) from exc

            client_kwargs: dict[str, object] = {
                "service_name": "s3",
                "region_name": self.config.docs_s3_region,
            }
            if self.config.docs_s3_endpoint_url:
                client_kwargs["endpoint_url"] = self.config.docs_s3_endpoint_url
            if self.config.docs_s3_access_key_id:
                client_kwargs["aws_access_key_id"] = self.config.docs_s3_access_key_id
            if self.config.docs_s3_secret_access_key:
                client_kwargs[
                    "aws_secret_access_key"
                ] = self.config.docs_s3_secret_access_key
            if self.config.docs_s3_session_token:
                client_kwargs["aws_session_token"] = self.config.docs_s3_session_token
            if self.config.docs_s3_force_path_style:
                client_kwargs["config"] = Config(
                    s3={"addressing_style": "path"},
                )
            self._client = boto3.client(**client_kwargs)
        return self._client

    @property
    def bucket(self) -> str:
        return self.config.docs_s3_bucket

    def ensure_versioning_enabled(self) -> None:
        if not self.config.docs_s3_require_versioning:
            return

        response = self.client.get_bucket_versioning(Bucket=self.bucket)
        if response.get("Status") != "Enabled":
            raise RuntimeError(
                "S3 bucket versioning must be Enabled for document ingest."
            )

    def list_current_markdown(self) -> list[S3DocumentRecord]:
        prefix = self._normalized_prefix(self.config.docs_s3_prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        records: list[S3DocumentRecord] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key.endswith(".md"):
                    continue
                records.append(self._record_for_key(key))
        records.sort(key=lambda record: record.key)
        return records

    def read_markdown(self, record: S3DocumentRecord) -> str:
        kwargs: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": record.key,
        }
        if record.version_id:
            kwargs["VersionId"] = record.version_id
        response = self.client.get_object(**kwargs)
        body = response["Body"].read()
        return body.decode("utf-8")

    def load_active_manifest(self) -> dict[str, object] | None:
        return self._read_json_if_exists(self.active_manifest_key())

    def load_version_manifest(self, index_version: str) -> dict[str, object] | None:
        return self._read_json_if_exists(self.version_manifest_key(index_version))

    def write_active_manifest(self, manifest: dict[str, object]) -> None:
        self._put_json(self.active_manifest_key(), manifest)

    def delete_active_manifest(self) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self.active_manifest_key())

    def write_version_manifest(self, manifest: dict[str, object]) -> None:
        self._put_json(self.version_manifest_key(str(manifest["index_version"])), manifest)

    def load_build_init_marker(self) -> dict[str, object] | None:
        return self._read_json_if_exists(self.build_init_marker_key())

    def write_build_init_marker(self, marker: dict[str, object]) -> None:
        self._put_json(self.build_init_marker_key(), marker)

    def prune_document_versions(self, retain_versions: int) -> dict[str, int]:
        retain_versions = max(1, retain_versions)
        prefix = self._normalized_prefix(self.config.docs_s3_prefix)
        paginator = self.client.get_paginator("list_object_versions")
        versions_by_key: dict[str, list[dict[str, object]]] = {}
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for version in page.get("Versions", []):
                key = str(version.get("Key", ""))
                if key.endswith(".md"):
                    item = dict(version)
                    item["_delete_marker"] = False
                    versions_by_key.setdefault(key, []).append(item)
            for marker in page.get("DeleteMarkers", []):
                key = str(marker.get("Key", ""))
                if key.endswith(".md"):
                    item = dict(marker)
                    item["_delete_marker"] = True
                    versions_by_key.setdefault(key, []).append(item)

        deleted = 0
        for key, versions in versions_by_key.items():
            versions.sort(
                key=self._version_sort_key,
                reverse=True,
            )
            keep = list(versions[:retain_versions])
            newest_usable = next(
                (version for version in versions if not version.get("_delete_marker")),
                None,
            )
            if newest_usable is not None and newest_usable not in keep:
                if keep:
                    keep[-1] = newest_usable
                else:
                    keep.append(newest_usable)
            keep_ids = {str(version.get("VersionId", "")) for version in keep}

            for version in versions:
                if str(version.get("VersionId", "")) in keep_ids:
                    continue
                version_id = str(version.get("VersionId", ""))
                if not version_id:
                    continue
                self.client.delete_object(
                    Bucket=self.bucket,
                    Key=key,
                    VersionId=version_id,
                )
                deleted += 1

        return {
            "keys": len(versions_by_key),
            "deleted_versions": deleted,
            "retain_versions": retain_versions,
        }

    @staticmethod
    def _version_sort_key(version: dict[str, object]) -> float:
        last_modified = version.get("LastModified")
        if hasattr(last_modified, "timestamp"):
            return float(last_modified.timestamp())
        if isinstance(last_modified, str):
            try:
                return datetime.fromisoformat(last_modified).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    def find_manifest_by_collection(self, collection_name: str) -> dict[str, object] | None:
        prefix = self._versions_prefix()
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key.endswith(".json"):
                    continue
                manifest = self._read_json_if_exists(key)
                if manifest and manifest.get("qdrant_collection") == collection_name:
                    return manifest
        return None

    def sync_local_directory(
        self,
        docs_dir: Path,
        *,
        delete_removed: bool,
    ) -> dict[str, int]:
        self.ensure_versioning_enabled()
        docs_dir = docs_dir.resolve()
        if not docs_dir.exists():
            raise RuntimeError(f"Docs directory does not exist: {docs_dir}")

        local_files = {
            path.relative_to(docs_dir).as_posix(): path
            for path in sorted(docs_dir.rglob("*.md"))
            if path.is_file()
        }
        remote_records = {
            self._relative_source(record.key): record
            for record in self.list_current_markdown()
        }

        uploaded = 0
        skipped = 0
        deleted = 0

        for relative_path, path in local_files.items():
            data = path.read_bytes()
            content_hash = hashlib.sha256(data).hexdigest()
            remote = remote_records.get(relative_path)
            if remote and remote.content_hash == content_hash:
                skipped += 1
                continue

            key = self._data_key(relative_path)
            content_type = mimetypes.guess_type(path.name)[0] or "text/markdown"
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata={_S3_METADATA_SHA256: content_hash},
            )
            uploaded += 1

        if delete_removed:
            for relative_path, record in remote_records.items():
                if relative_path in local_files:
                    continue
                self.client.delete_object(Bucket=self.bucket, Key=record.key)
                deleted += 1

        return {
            "uploaded": uploaded,
            "skipped": skipped,
            "deleted": deleted,
            "local_files": len(local_files),
            "remote_files": len(remote_records),
        }

    def active_manifest_key(self) -> str:
        return posixpath.join(
            self._normalized_prefix(self.config.docs_s3_manifest_prefix),
            "active.json",
        )

    def build_init_marker_key(self) -> str:
        return posixpath.join(
            self._normalized_prefix(self.config.docs_s3_manifest_prefix),
            "build-init.json",
        )

    def version_manifest_key(self, index_version: str) -> str:
        return posixpath.join(self._versions_prefix(), f"{index_version}.json")

    def _versions_prefix(self) -> str:
        return posixpath.join(
            self._normalized_prefix(self.config.docs_s3_manifest_prefix),
            "versions",
        )

    def _record_for_key(self, key: str) -> S3DocumentRecord:
        response = self.client.head_object(Bucket=self.bucket, Key=key)
        metadata = response.get("Metadata") or {}
        version_id = str(response.get("VersionId") or "")
        if self.config.docs_s3_require_versioning and not version_id:
            raise RuntimeError(
                f"S3 object {key} has no VersionId; bucket versioning is required."
            )

        last_modified = response.get("LastModified")
        if hasattr(last_modified, "astimezone"):
            last_modified_text = last_modified.astimezone(timezone.utc).isoformat()
        else:
            last_modified_text = ""
        return S3DocumentRecord(
            source_doc_id=str(uuid5(NAMESPACE_URL, f"s3://{self.bucket}/{key}")),
            key=key,
            source=self._relative_source(key),
            version_id=version_id,
            etag=str(response.get("ETag", "")).strip('"'),
            size=int(response.get("ContentLength", 0) or 0),
            last_modified=last_modified_text,
            content_hash=str(metadata.get(_S3_METADATA_SHA256, "")),
        )

    def _data_key(self, relative_path: str) -> str:
        prefix = self._normalized_prefix(self.config.docs_s3_prefix).rstrip("/")
        relative_path = relative_path.strip("/")
        if not prefix:
            return relative_path
        return posixpath.join(prefix, relative_path)

    def _relative_source(self, key: str) -> str:
        prefix = self._normalized_prefix(self.config.docs_s3_prefix).rstrip("/")
        if prefix and key.startswith(f"{prefix}/"):
            return key[len(prefix) + 1 :]
        return key

    @staticmethod
    def _normalized_prefix(prefix: str) -> str:
        return prefix.strip("/")

    def _read_json_if_exists(self, key: str) -> dict[str, object] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    def _put_json(self, key: str, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType="application/json",
        )


class VersionedS3DocumentIndexer:
    def __init__(
        self,
        config: Settings = settings,
        *,
        document_store: S3DocumentStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config
        self.document_store = document_store or S3DocumentStore(config)
        self.vector_store = vector_store or VectorStore(config)

    def ingest(
        self,
        *,
        recreate: bool = False,
        dry_run: bool = False,
        committer: DocumentIndexCommitter | None = None,
    ) -> S3IngestResult:
        self.document_store.ensure_versioning_enabled()
        current_records = self.document_store.list_current_markdown()
        active_manifest = self._load_active_manifest()
        previous_collection = (
            str(active_manifest.get("qdrant_collection", ""))
            if active_manifest
            else ""
        )
        previous_docs = self._manifest_docs(active_manifest)
        current_docs = {record.source_doc_id: record for record in current_records}

        config_changed = self._index_config_changed(active_manifest)
        if recreate or config_changed:
            unchanged_ids: set[str] = set()
            added_records = current_records
            modified_records: list[S3DocumentRecord] = []
        else:
            unchanged_ids = {
                doc_id
                for doc_id, record in current_docs.items()
                if self._record_matches_previous(record, previous_docs.get(doc_id))
            }
            added_records = [
                record
                for doc_id, record in current_docs.items()
                if doc_id not in previous_docs
            ]
            modified_records = [
                record
                for doc_id, record in current_docs.items()
                if doc_id in previous_docs and doc_id not in unchanged_ids
            ]

        deleted_doc_ids = set(previous_docs) - set(current_docs)
        has_changes = bool(
            recreate
            or config_changed
            or added_records
            or modified_records
            or deleted_doc_ids
            or not previous_collection
        )
        active_collection = self._active_collection_name()
        if not has_changes and active_collection:
            return S3IngestResult(
                index_version=str(active_manifest.get("index_version", "")) if active_manifest else "",
                qdrant_alias=self.config.collection_name,
                qdrant_collection=active_collection,
                previous_collection=previous_collection,
                added_docs=0,
                modified_docs=0,
                deleted_docs=0,
                unchanged_docs=len(unchanged_ids),
                copied_chunks=0,
                embedded_chunks=0,
                total_chunks=self._collection_count(active_collection),
                skipped=True,
            )

        index_version = self._new_index_version(current_records)
        new_collection = self._version_collection_name(index_version)
        manifest = self._manifest_payload(
            index_version=index_version,
            qdrant_collection=new_collection,
            previous_collection=previous_collection,
            current_records=current_records,
            diff={
                "added_doc_ids": [record.source_doc_id for record in added_records],
                "modified_doc_ids": [record.source_doc_id for record in modified_records],
                "deleted_doc_ids": sorted(deleted_doc_ids),
                "unchanged_doc_ids": sorted(unchanged_ids),
                "config_changed": config_changed,
                "recreate": recreate,
            },
        )
        manifest["status"] = "processing"

        if dry_run:
            return S3IngestResult(
                index_version=index_version,
                qdrant_alias=self.config.collection_name,
                qdrant_collection=new_collection,
                previous_collection=previous_collection,
                added_docs=len(added_records),
                modified_docs=len(modified_records),
                deleted_docs=len(deleted_doc_ids),
                unchanged_docs=len(unchanged_ids),
                copied_chunks=0,
                embedded_chunks=0,
                total_chunks=0,
                skipped=False,
            )

        self.document_store.write_version_manifest(manifest)
        prepare_executor: ThreadPoolExecutor | None = None
        bm25_future: Future[int] | None = None
        if committer is not None:
            prepare_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="bm25-candidate",
            )
            bm25_future = prepare_executor.submit(committer.prepare, index_version)

        self.document_store.prune_document_versions(
            self.config.docs_s3_processing_retain_versions
        )
        self._prune_qdrant_versions(
            retain_versions=self.config.qdrant_processing_retain_versions,
            active_collection=active_collection,
        )

        new_collection_created = False
        alias_switched = False
        rollback_collection = active_collection
        try:
            self._create_version_collection(new_collection)
            new_collection_created = True
            copied_chunks = self._copy_unchanged_chunks(
                source_collection=previous_collection,
                target_collection=new_collection,
                unchanged_doc_ids=unchanged_ids,
                index_version=index_version,
            )
            embedded_chunks = self._index_records(
                new_collection,
                self._records_selected_from_manifest(
                    manifest,
                    {
                        record.source_doc_id
                        for record in (*added_records, *modified_records)
                    },
                ),
                index_version=index_version,
            )
            total_chunks = self._collection_count(new_collection)
            expected_chunks = copied_chunks + embedded_chunks
            if total_chunks != expected_chunks:
                raise RuntimeError(
                    "Indexed chunk count mismatch: "
                    f"expected={expected_chunks} actual={total_chunks}."
                )

            manifest.update(
                {
                    "status": "candidate",
                    "copied_chunks": copied_chunks,
                    "embedded_chunks": embedded_chunks,
                    "total_chunks": total_chunks,
                }
            )
            self.document_store.write_version_manifest(manifest)
            if committer is None:
                rollback_collection = self._switch_alias(new_collection)
                alias_switched = True
                active_payload = {**manifest, "status": "active"}
                self.document_store.write_active_manifest(active_payload)
                self.document_store.write_version_manifest(active_payload)
            else:
                if bm25_future is None:
                    raise RuntimeError("BM25S candidate preparation was not started.")
                manifest["bm25_chunks"] = int(bm25_future.result())
                self.document_store.write_version_manifest(manifest)
                rollback_collection = committer.commit(
                    PreparedIndexCommit(
                        index_version=index_version,
                        candidate_collection=new_collection,
                        previous_collection=active_collection,
                        expected_total_chunks=total_chunks,
                    )
                )
                alias_switched = True
        except Exception:
            logger.exception(
                "S3 document ingest failed; rolling back to %s.",
                rollback_collection or "no previous document alias",
            )
            active_now = self._active_collection_name()
            active_now_manifest = self.document_store.load_active_manifest() or {}
            commit_visible = (
                committer is not None
                and active_now == new_collection
                and active_now_manifest.get("qdrant_collection") == new_collection
                and active_now_manifest.get("index_version") == index_version
            )
            if commit_visible:
                logger.warning(
                    "Treating document commit as successful after an ambiguous "
                    "coordinator response: index_version=%s.",
                    index_version,
                )
                alias_switched = True
            elif committer is None and alias_switched:
                self._rollback_alias(rollback_collection)
                try:
                    if active_manifest:
                        self.document_store.write_active_manifest(active_manifest)
                    else:
                        self.document_store.delete_active_manifest()
                except Exception:
                    logger.exception("Failed to restore the previous S3 manifest.")
            else:
                if committer is not None:
                    if bm25_future is not None:
                        try:
                            bm25_future.result()
                        except Exception:
                            pass
                    try:
                        committer.discard(index_version)
                    except Exception:
                        logger.exception(
                            "Failed to discard Main BM25S candidate %s.",
                            index_version,
                        )
                failed_payload = {**manifest, "status": "failed"}
                self.document_store.write_version_manifest(failed_payload)
            if new_collection_created and not commit_visible:
                self._delete_collection_if_exists(new_collection)
            if not commit_visible:
                raise
        finally:
            if prepare_executor is not None:
                prepare_executor.shutdown(wait=True, cancel_futures=True)

        try:
            self.document_store.prune_document_versions(
                self.config.docs_s3_retain_versions
            )
            self._prune_qdrant_versions(
                retain_versions=self.config.qdrant_retain_versions,
                active_collection=new_collection,
            )
        except Exception:
            logger.exception("Post-ingest document retention cleanup failed.")

        with self.vector_store._bm25_lock:  # noqa: SLF001 - same ownership boundary
            self.vector_store._bm25_index = None  # noqa: SLF001
            self.vector_store._bm25_candidate = None  # noqa: SLF001

        return S3IngestResult(
            index_version=index_version,
            qdrant_alias=self.config.collection_name,
            qdrant_collection=new_collection,
            previous_collection=previous_collection,
            added_docs=len(added_records),
            modified_docs=len(modified_records),
            deleted_docs=len(deleted_doc_ids),
            unchanged_docs=len(unchanged_ids),
            copied_chunks=copied_chunks,
            embedded_chunks=embedded_chunks,
            total_chunks=total_chunks,
            skipped=False,
        )

    def reconcile_active_manifest(self) -> dict[str, object] | None:
        active_collection = self._active_collection_name()
        if not active_collection:
            return None
        current = self.document_store.load_active_manifest()
        if current and current.get("qdrant_collection") == active_collection:
            return current
        recovered = self.document_store.find_manifest_by_collection(active_collection)
        if not recovered:
            raise RuntimeError(
                "No S3 manifest matches the active Qdrant collection: "
                f"{active_collection}."
            )
        active_payload = {**recovered, "status": "active"}
        self.document_store.write_active_manifest(active_payload)
        self.document_store.write_version_manifest(active_payload)
        return active_payload

    def commit_prepared_version(
        self,
        prepared: PreparedIndexCommit,
    ) -> dict[str, object]:
        manifest = self.document_store.load_version_manifest(
            prepared.index_version
        )
        if not manifest:
            raise RuntimeError(
                f"Prepared S3 manifest is missing: {prepared.index_version}."
            )
        if (
            manifest.get("index_version") != prepared.index_version
            or manifest.get("qdrant_collection") != prepared.candidate_collection
        ):
            raise RuntimeError("Prepared S3 manifest does not match the commit request.")
        if str(manifest.get("status", "")) not in {"candidate", "active"}:
            raise RuntimeError("Prepared S3 manifest is not ready for commit.")

        actual_chunks = self._collection_count(prepared.candidate_collection)
        if actual_chunks != prepared.expected_total_chunks:
            raise RuntimeError(
                "Candidate Qdrant chunk count changed before commit: "
                f"expected={prepared.expected_total_chunks} actual={actual_chunks}."
            )
        active_collection = self._active_collection_name()
        if active_collection not in {
            prepared.previous_collection,
            prepared.candidate_collection,
        }:
            raise RuntimeError(
                "Active Qdrant alias changed during document preparation: "
                f"expected={prepared.previous_collection} actual={active_collection}."
            )

        if (
            active_collection == prepared.candidate_collection
            and self.vector_store.active_bm25_index_version()
            == prepared.index_version
        ):
            return {
                "previous_collection": prepared.previous_collection,
                "active_collection": prepared.candidate_collection,
                "index_version": prepared.index_version,
                "bm25_chunks": int(manifest.get("bm25_chunks", 0) or 0),
                "idempotent": True,
            }

        previous_manifest = self.document_store.load_active_manifest()
        if (
            not previous_manifest
            or previous_manifest.get("qdrant_collection")
            != prepared.previous_collection
        ):
            previous_manifest = self.document_store.find_manifest_by_collection(
                prepared.previous_collection
            )

        alias_changed = active_collection != prepared.candidate_collection
        try:
            if alias_changed:
                self._switch_alias(prepared.candidate_collection)
            active_payload = {**manifest, "status": "active"}
            self.document_store.write_active_manifest(active_payload)
            if self._active_collection_name() != prepared.candidate_collection:
                raise RuntimeError("Qdrant alias verification failed after commit.")
            committed_manifest = self.document_store.load_active_manifest() or {}
            if (
                committed_manifest.get("index_version") != prepared.index_version
                or committed_manifest.get("qdrant_collection")
                != prepared.candidate_collection
            ):
                raise RuntimeError("Active S3 manifest verification failed after commit.")
            self.document_store.write_version_manifest(active_payload)
            bm25_chunks = self.vector_store.activate_bm25_candidate(
                prepared.index_version
            )
        except Exception:
            logger.exception(
                "Atomic document commit failed; restoring collection %s.",
                prepared.previous_collection or "<none>",
            )
            if self._active_collection_name() == prepared.candidate_collection:
                self._rollback_alias(prepared.previous_collection)
            if previous_manifest:
                self.document_store.write_active_manifest(previous_manifest)
            else:
                self.document_store.delete_active_manifest()
            raise

        return {
            "previous_collection": prepared.previous_collection,
            "active_collection": prepared.candidate_collection,
            "index_version": prepared.index_version,
            "bm25_chunks": bm25_chunks,
            "idempotent": False,
        }

    def _load_active_manifest(self) -> dict[str, object] | None:
        manifest = self.document_store.load_active_manifest()
        active_collection = self._active_collection_name()
        if (
            manifest
            and active_collection
            and manifest.get("qdrant_collection") != active_collection
        ):
            recovered = self.document_store.find_manifest_by_collection(
                active_collection
            )
            if recovered:
                logger.warning(
                    "Recovered S3 document manifest from active Qdrant alias target %s.",
                    active_collection,
                )
                return recovered
        if not manifest and active_collection:
            recovered = self.document_store.find_manifest_by_collection(active_collection)
            if recovered:
                return recovered
        return manifest

    @staticmethod
    def _manifest_docs(manifest: dict[str, object] | None) -> dict[str, dict[str, object]]:
        if not manifest:
            return {}
        docs = manifest.get("documents", [])
        if not isinstance(docs, list):
            return {}
        return {
            str(doc.get("source_doc_id")): doc
            for doc in docs
            if isinstance(doc, dict) and doc.get("source_doc_id")
        }

    @staticmethod
    def _records_selected_from_manifest(
        manifest: dict[str, object],
        source_doc_ids: set[str],
    ) -> list[S3DocumentRecord]:
        records: list[S3DocumentRecord] = []
        for item in manifest.get("documents", []):
            if not isinstance(item, dict):
                continue
            source_doc_id = str(item.get("source_doc_id", ""))
            if source_doc_id not in source_doc_ids:
                continue
            records.append(
                S3DocumentRecord(
                    source_doc_id=source_doc_id,
                    key=str(item.get("key", "")),
                    source=str(item.get("source", "")),
                    version_id=str(item.get("version_id", "")),
                    etag=str(item.get("etag", "")),
                    size=int(item.get("size", 0) or 0),
                    last_modified=str(item.get("last_modified", "")),
                    content_hash=str(item.get("content_hash", "")),
                )
            )
        return records

    def _index_config_changed(self, manifest: dict[str, object] | None) -> bool:
        if not manifest:
            return False
        return (
            manifest.get("embedding_model") != self.config.embedding_model
            or manifest.get("chunking_version") != CHUNKING_VERSION
            or manifest.get("chunk_tokenizer_model")
            != self.config.chunk_tokenizer_model
            or bool(manifest.get("chunk_tokenizer_trust_remote_code", False))
            != self.config.chunk_tokenizer_trust_remote_code
            or int(manifest.get("chunk_body_target_tokens", 0) or 0)
            != self.config.chunk_body_target_tokens
            or int(manifest.get("chunk_body_max_tokens", 0) or 0)
            != self.config.chunk_body_max_tokens
            or int(manifest.get("chunk_overlap_target_tokens", -1) or -1)
            != self.config.chunk_overlap_target_tokens
            or int(manifest.get("chunk_overlap_max_tokens", -1) or -1)
            != self.config.chunk_overlap_max_tokens
            or manifest.get("embedding_passage_task")
            != self.config.embedding_passage_task
            or manifest.get("embedding_passage_prompt_name")
            != self.config.embedding_passage_prompt_name
        )

    @staticmethod
    def _record_matches_previous(
        record: S3DocumentRecord,
        previous: dict[str, object] | None,
    ) -> bool:
        if previous is None:
            return False
        return (
            previous.get("key") == record.key
            and previous.get("version_id") == record.version_id
            and previous.get("etag") == record.etag
            and int(previous.get("size", -1) or -1) == record.size
            and str(previous.get("content_hash", "")) == record.content_hash
        )

    def _new_index_version(self, current_records: list[S3DocumentRecord]) -> str:
        payload = {
            "time_ns": time.time_ns(),
            "records": [asdict(record) for record in current_records],
            "embedding_model": self.config.embedding_model,
            "chunking_version": CHUNKING_VERSION,
            "chunk_tokenizer_model": self.config.chunk_tokenizer_model,
            "chunk_tokenizer_trust_remote_code": (
                self.config.chunk_tokenizer_trust_remote_code
            ),
            "chunk_body_target_tokens": self.config.chunk_body_target_tokens,
            "chunk_body_max_tokens": self.config.chunk_body_max_tokens,
            "chunk_overlap_target_tokens": (
                self.config.chunk_overlap_target_tokens
            ),
            "chunk_overlap_max_tokens": self.config.chunk_overlap_max_tokens,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{digest}"

    def _version_collection_name(self, index_version: str) -> str:
        return f"{self._version_collection_base()}__v{index_version}"

    def _manifest_payload(
        self,
        *,
        index_version: str,
        qdrant_collection: str,
        previous_collection: str,
        current_records: list[S3DocumentRecord],
        diff: dict[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "index_version": index_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "bucket": self.config.docs_s3_bucket,
            "prefix": self.config.docs_s3_prefix,
            "qdrant_alias": self.config.collection_name,
            "qdrant_collection": qdrant_collection,
            "previous_collection": previous_collection,
            "embedding_model": self.config.embedding_model,
            "embedding_passage_task": self.config.embedding_passage_task,
            "embedding_passage_prompt_name": self.config.embedding_passage_prompt_name,
            "chunking_version": CHUNKING_VERSION,
            "chunk_tokenizer_model": self.config.chunk_tokenizer_model,
            "chunk_tokenizer_trust_remote_code": (
                self.config.chunk_tokenizer_trust_remote_code
            ),
            "chunk_body_target_tokens": self.config.chunk_body_target_tokens,
            "chunk_body_max_tokens": self.config.chunk_body_max_tokens,
            "chunk_overlap_target_tokens": (
                self.config.chunk_overlap_target_tokens
            ),
            "chunk_overlap_max_tokens": self.config.chunk_overlap_max_tokens,
            "documents": [asdict(record) for record in current_records],
            "diff": diff,
        }

    def _create_version_collection(self, collection_name: str) -> None:
        collection_names = {
            collection.name for collection in self.vector_store.client.get_collections().collections
        }
        if collection_name in collection_names:
            self.vector_store.client.delete_collection(collection_name=collection_name)

        self.vector_store.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_store.vector_size,
                distance=Distance.COSINE,
            ),
        )
        for field_name in ("source_doc_id", "version_id", "source_key"):
            try:
                self.vector_store.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception:
                logger.exception(
                    "Failed to create payload index %s on %s.",
                    field_name,
                    collection_name,
                )

    def _copy_unchanged_chunks(
        self,
        *,
        source_collection: str,
        target_collection: str,
        unchanged_doc_ids: set[str],
        index_version: str,
    ) -> int:
        if not source_collection or not unchanged_doc_ids:
            return 0

        collection_names = {
            collection.name for collection in self.vector_store.client.get_collections().collections
        }
        if source_collection not in collection_names:
            return 0

        copied = 0
        offset = None
        batch: list[PointStruct] = []
        while True:
            records, offset = self.vector_store.client.scroll(
                collection_name=source_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                payload = dict(record.payload or {})
                if str(payload.get("source_doc_id", "")) not in unchanged_doc_ids:
                    continue
                payload["index_version"] = index_version
                if record.vector is None:
                    raise RuntimeError("Cannot copy a Qdrant point without vector data.")
                batch.append(
                    PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=payload,
                    )
                )
                copied += 1
                if len(batch) >= 256:
                    self.vector_store.client.upsert(
                        collection_name=target_collection,
                        points=batch,
                        wait=True,
                    )
                    batch = []
            if offset is None:
                break

        if batch:
            self.vector_store.client.upsert(
                collection_name=target_collection,
                points=batch,
                wait=True,
            )
        return copied

    def _index_records(
        self,
        collection_name: str,
        records: list[S3DocumentRecord],
        *,
        index_version: str,
    ) -> int:
        inserted = 0
        pending: list[tuple[S3DocumentRecord, int, MarkdownChunk]] = []
        for record in records:
            text = self.document_store.read_markdown(record)
            chunks = chunk_markdown(
                text,
                tokenizer=self.vector_store.chunk_tokenizer,
                body_target_tokens=self.config.chunk_body_target_tokens,
                body_max_tokens=self.config.chunk_body_max_tokens,
                overlap_target_tokens=self.config.chunk_overlap_target_tokens,
                overlap_max_tokens=self.config.chunk_overlap_max_tokens,
            )
            if not chunks:
                continue
            for idx, chunk in enumerate(chunks):
                pending.append((record, idx, chunk))
                if len(pending) >= self.config.embedding_offline_batch_size:
                    inserted += self._embed_and_upsert_chunks(
                        collection_name,
                        pending,
                        index_version=index_version,
                    )
                    pending = []

        if pending:
            inserted += self._embed_and_upsert_chunks(
                collection_name,
                pending,
                index_version=index_version,
            )
        return inserted

    def _embed_and_upsert_chunks(
        self,
        collection_name: str,
        pending: list[tuple[S3DocumentRecord, int, MarkdownChunk]],
        *,
        index_version: str,
    ) -> int:
        embeddings = self.vector_store._encode_matrix(  # noqa: SLF001
            [chunk.embedding_text for _record, _idx, chunk in pending],
            normalize_embeddings=True,
            task=self.config.embedding_passage_task,
            prompt_name=self.config.embedding_passage_prompt_name,
        )
        if len(embeddings) != len(pending):
            raise RuntimeError("Offline embedding batch returned an unexpected row count.")

        points: list[PointStruct] = []
        for (record, idx, chunk), embedding in zip(
            pending,
            embeddings,
            strict=True,
        ):
            source_key = f"s3://{self.config.docs_s3_bucket}/{record.key}"
            version_id = record.version_id or record.etag or record.content_hash
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{record.source_doc_id}:{version_id}:{idx}",
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "source": record.source,
                        "source_key": source_key,
                        "source_doc_id": record.source_doc_id,
                        "version_id": version_id,
                        "s3_bucket": self.config.docs_s3_bucket,
                        "s3_key": record.key,
                        "s3_version_id": record.version_id,
                        "etag": record.etag,
                        "content_hash": record.content_hash,
                        "object_size": record.size,
                        "source_last_modified": record.last_modified,
                        "index_version": index_version,
                        "chunk_id": idx,
                        "text": chunk.text,
                        "content_type": chunk.content_type,
                        "h1": chunk.h1,
                        "h2": chunk.h2,
                        "h3": chunk.h3,
                        "headings": list(chunk.headings),
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "body_token_count": chunk.body_token_count,
                        "prefix_overlap_token_count": (
                            chunk.prefix_overlap_token_count
                        ),
                        "suffix_overlap_token_count": (
                            chunk.suffix_overlap_token_count
                        ),
                        "token_count": chunk.token_count,
                    },
                )
            )

        self.vector_store.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def _switch_alias(self, new_collection: str) -> str:
        active_collection = self._active_collection_name()
        archived_collection = ""
        if not active_collection:
            archived_collection = self._archive_conflicting_collection()
            if archived_collection:
                logger.warning(
                    "Archived existing physical document collection %s to %s "
                    "before creating the stable alias.",
                    self.config.collection_name,
                    archived_collection,
                )
        operations: list[object] = []
        if active_collection:
            operations.append(
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=self.config.collection_name)
                )
            )
        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=new_collection,
                    alias_name=self.config.collection_name,
                )
            )
        )
        try:
            self.vector_store.client.update_collection_aliases(
                change_aliases_operations=operations
            )
        except Exception:
            if archived_collection:
                self.vector_store.client.update_collection_aliases(
                    change_aliases_operations=[
                        CreateAliasOperation(
                            create_alias=CreateAlias(
                                collection_name=archived_collection,
                                alias_name=self.config.collection_name,
                            )
                        )
                    ]
                )
            raise
        return active_collection or archived_collection

    def _rollback_alias(self, collection_name: str) -> None:
        active_collection = self._active_collection_name()
        operations: list[object] = []
        if active_collection:
            operations.append(
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=self.config.collection_name)
                )
            )
        if collection_name:
            operations.append(
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=collection_name,
                        alias_name=self.config.collection_name,
                    )
                )
            )
        if operations:
            self.vector_store.client.update_collection_aliases(
                change_aliases_operations=operations
            )

    def _archive_conflicting_collection(self) -> str:
        collection_names = {
            collection.name
            for collection in self.vector_store.client.get_collections().collections
        }
        if self.config.collection_name not in collection_names:
            return ""

        archive_name = (
            f"{self._version_collection_name('legacy')}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        vector_size = self._collection_vector_size(self.config.collection_name)
        self.vector_store.client.create_collection(
            collection_name=archive_name,
            vectors_config=VectorParams(
                size=vector_size or self.vector_store.vector_size,
                distance=Distance.COSINE,
            ),
        )
        self._copy_collection_points(
            source_collection=self.config.collection_name,
            target_collection=archive_name,
        )
        self.vector_store.client.delete_collection(
            collection_name=self.config.collection_name
        )
        return archive_name

    def _copy_collection_points(
        self,
        *,
        source_collection: str,
        target_collection: str,
    ) -> int:
        copied = 0
        offset = None
        batch: list[PointStruct] = []
        while True:
            records, offset = self.vector_store.client.scroll(
                collection_name=source_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                if record.vector is None:
                    continue
                batch.append(
                    PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=dict(record.payload or {}),
                    )
                )
                copied += 1
                if len(batch) >= 256:
                    self.vector_store.client.upsert(
                        collection_name=target_collection,
                        points=batch,
                        wait=True,
                    )
                    batch = []
            if offset is None:
                break
        if batch:
            self.vector_store.client.upsert(
                collection_name=target_collection,
                points=batch,
                wait=True,
            )
        return copied

    def _collection_vector_size(self, collection_name: str) -> int:
        info = self.vector_store.client.get_collection(collection_name=collection_name)
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            for value in vectors.values():
                if hasattr(value, "size"):
                    return int(value.size)
        return 0

    def _prune_qdrant_versions(
        self,
        *,
        retain_versions: int,
        active_collection: str,
    ) -> list[str]:
        retain_versions = max(1, retain_versions)
        version_collections = self._version_collections()
        keep: set[str] = set()
        if active_collection:
            keep.add(active_collection)
        for collection_name in version_collections:
            if len(keep) >= retain_versions:
                break
            keep.add(collection_name)

        deleted: list[str] = []
        for collection_name in version_collections:
            if collection_name in keep:
                continue
            self.vector_store.client.delete_collection(collection_name=collection_name)
            deleted.append(collection_name)
        return deleted

    def _version_collections(self) -> list[str]:
        prefix = f"{self._version_collection_base()}__v"
        collections = [
            collection.name
            for collection in self.vector_store.client.get_collections().collections
            if collection.name.startswith(prefix)
            and not collection.name.startswith(f"{prefix}legacy_")
        ]
        return sorted(collections, reverse=True)

    def _delete_collection_if_exists(self, collection_name: str) -> None:
        collection_names = {
            collection.name
            for collection in self.vector_store.client.get_collections().collections
        }
        if collection_name in collection_names:
            self.vector_store.client.delete_collection(collection_name=collection_name)

    def _version_collection_base(self) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", self.config.collection_name).strip(
            "_"
        ) or "docs"

    def _active_collection_name(self) -> str:
        try:
            aliases = self.vector_store.client.get_aliases().aliases
        except Exception:
            return ""
        for alias in aliases:
            if alias.alias_name == self.config.collection_name:
                return alias.collection_name
        return ""

    def _collection_count(self, collection_name: str) -> int:
        response = self.vector_store.client.count(
            collection_name=collection_name,
            exact=True,
        )
        return int(response.count)


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        code = str(error.get("Code", ""))
        return code in {"NoSuchKey", "404", "NotFound"}
    code = str(getattr(exc, "response_code", ""))
    return code == "404"
