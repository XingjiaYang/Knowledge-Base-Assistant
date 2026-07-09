from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import logging
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import PROJECT_ROOT, Settings, settings
from app.device import preferred_torch_device
from app.transformers_compat import patch_all_tied_weights_keys


logger = logging.getLogger(__name__)

_PYTHON_EXTENSIONS = {".py"}
_CPP_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
_CODE_EXTENSIONS = _PYTHON_EXTENSIONS | _CPP_EXTENSIONS
_TREE_SITTER_LANGUAGES = {"python": "python", "cpp": "cpp"}
_QDRANT_BATCH_SIZE = 64
_CPP_MACRO_FUNCTION_NAMES = {
    "TEST",
    "TEST_F",
    "TEST_P",
    "TYPED_TEST",
    "TYPED_TEST_P",
    "BENCHMARK",
}
_REPOSITORY_MARKERS = {
    ".git",
    "pyproject.toml",
    "setup.py",
    "CMakeLists.txt",
    "package.json",
    "go.mod",
    "Cargo.toml",
}


@dataclass(frozen=True)
class CodeFileRecord:
    id: UUID
    repository_id: str
    repository_name: str
    path: str
    source_root: str
    language: str
    full_content: str
    content_sha256: str
    line_count: int

    @property
    def embedding_text(self) -> str:
        return "\n\n".join(
            [
                f"Repository: {self.repository_name}",
                f"Path: {self.path}",
                f"Language: {self.language}",
                self.full_content,
            ]
        ).strip()


@dataclass(frozen=True)
class CodeFunctionRecord:
    id: UUID
    file_id: UUID
    repository_id: str
    repository_name: str
    file_path: str
    language: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    body: str
    docstring: str
    start_line: int
    end_line: int

    @property
    def embedding_text(self) -> str:
        parts = [
            f"Repository: {self.repository_name}",
            f"Path: {self.file_path}",
            f"Language: {self.language}",
            f"Kind: {self.kind}",
            f"Name: {self.qualified_name}",
            self.signature,
        ]
        if self.docstring:
            parts.append(self.docstring)
        parts.append(self.body)
        return "\n\n".join(part for part in parts if part).strip()


@dataclass(frozen=True)
class CodeIndexStats:
    files: int = 0
    functions: int = 0
    call_edges: int = 0


@dataclass(frozen=True)
class CodeRepository:
    id: str
    name: str
    source_dir: Path
    source_root: str


def discover_code_repositories(config: Settings = settings) -> list[CodeRepository]:
    root = config.code_root_dir.expanduser().resolve()
    source = config.code_source_dir.expanduser().resolve()
    source_is_explicit = (
        config.code_source_dir_explicit
        or source != root
    )

    if source_is_explicit:
        return [code_repository_for_source(source, config)] if source.exists() else []

    if not root.exists():
        return []

    if _looks_like_repository(root):
        return [code_repository_for_source(root, config)]

    repositories = [
        code_repository_for_source(child, config)
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir()
        and not child.name.startswith(".")
        and _contains_supported_code(child)
    ]
    if repositories:
        return repositories

    return [code_repository_for_source(source, config)] if source.exists() else []


def code_repository_for_source(
    source_dir: Path,
    config: Settings = settings,
) -> CodeRepository:
    source = source_dir.expanduser().resolve()
    root = config.code_root_dir.expanduser().resolve()
    try:
        relative = source.relative_to(root)
        repository_id = relative.as_posix() if relative.parts else source.name
    except ValueError:
        repository_id = source.name
    repository_id = repository_id.strip("/") or source.name or "code"
    return CodeRepository(
        id=repository_id,
        name=source.name or repository_id,
        source_dir=source,
        source_root=_display_path(source),
    )


class CodeEmbedder:
    """CodeBERT mean-pooling embedder for code and natural-language queries."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        model: object | None = None,
        tokenizer: object | None = None,
    ) -> None:
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._device = "cpu" if model is not None else None
        self._lock = Lock()
        self._vector_size: int | None = None

    def warmup(self) -> int:
        """Load tokenizer/model into memory and return the embedding dimension."""
        _ = self.tokenizer
        return self.vector_size

    @property
    def tokenizer(self) -> object:
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.config.code_embedding_model,
                    )
        return self._tokenizer

    @property
    def model(self) -> object:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from transformers import AutoModel

                    patch_all_tied_weights_keys()
                    device = preferred_torch_device(
                        self.config.cuda_enabled,
                        "Code embedding model",
                    )
                    logger.info(
                        "Loading code embedding model %s on %s.",
                        self.config.code_embedding_model,
                        device,
                    )
                    model = AutoModel.from_pretrained(self.config.code_embedding_model)
                    self._device = self._move_model_to_device(model, device)
                    if hasattr(model, "eval"):
                        model.eval()
                    self._model = model
        return self._model

    @staticmethod
    def _move_model_to_device(model: object, device: str) -> str:
        if not hasattr(model, "to"):
            return device
        try:
            model.to(device)
            return device
        except Exception as exc:
            if device != "cuda":
                raise
            logger.warning(
                "Code embedding model failed to move to CUDA; falling back to CPU: %s",
                exc,
            )
            model.to("cpu")
            return "cpu"

    @property
    def vector_size(self) -> int:
        if self._vector_size is not None:
            return self._vector_size

        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size:
            self._vector_size = int(hidden_size)
            return self._vector_size

        vector = self.embed(["dimension probe"])[0]
        self._vector_size = len(vector)
        return self._vector_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        import torch

        vectors: list[list[float]] = []
        batch_size = self.config.code_embedding_batch_size
        model = self.model
        device = self._device or "cpu"
        for offset in range(0, len(texts), batch_size):
            batch = [text or "" for text in texts[offset : offset + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.code_embedding_max_tokens,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            try:
                with torch.no_grad():
                    output = model(**encoded)
                    pooled = self._mean_pool(output.last_hidden_state, encoded)
            except Exception as exc:
                if self._device != "cuda":
                    raise
                logger.warning(
                    "Code embedding failed during CUDA inference; falling back to CPU: %s",
                    exc,
                )
                self._device = self._move_model_to_device(self.model, "cpu")
                model = self.model
                encoded = {
                    key: value.to("cpu") if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                with torch.no_grad():
                    output = model(**encoded)
                    pooled = self._mean_pool(output.last_hidden_state, encoded)

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().float().tolist())
        return vectors

    @staticmethod
    def _mean_pool(last_hidden_state: object, encoded: dict[str, object]) -> object:
        import torch

        attention_mask = encoded["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class CodeIndexer:
    def __init__(
        self,
        config: Settings = settings,
        *,
        client: QdrantClient | None = None,
        embedder: CodeEmbedder | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._embedder = embedder or CodeEmbedder(config)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.config.qdrant_url)
        return self._client

    @property
    def embedder(self) -> CodeEmbedder:
        return self._embedder

    def index_source_tree(
        self,
        source_dir: Path | None = None,
        *,
        recreate: bool = False,
        repository: CodeRepository | None = None,
    ) -> CodeIndexStats:
        root = (source_dir or self.config.code_source_dir).resolve()
        repository = repository or code_repository_for_source(root, self.config)
        if not root.exists():
            logger.warning("Code source directory does not exist: %s", root)
            return CodeIndexStats()

        self.ensure_collections(recreate=recreate)
        if recreate:
            self._clear_postgres_tables()

        parsed_files = [
            parse_code_file(path, root, repository)
            for path in sorted(root.rglob("*"))
            if path.is_file() and _language_for_path(path) is not None
        ]
        parsed_files = [parsed for parsed in parsed_files if parsed is not None]
        if not parsed_files:
            logger.info("No supported code files found under %s.", root)
            return CodeIndexStats()

        file_points: list[PointStruct] = []
        function_points: list[PointStruct] = []
        edge_count = 0

        with self._connect() as conn:
            for code_file, functions in parsed_files:
                if not recreate:
                    self._delete_existing_file(conn, code_file)
                    self._delete_qdrant_file_points(code_file.id)
                self._upsert_file(conn, code_file)
                self._upsert_functions(conn, functions)
                edge_count += self._upsert_call_edges(conn, functions)

                file_points.append(self._file_point(code_file))
                function_points.extend(self._function_points(functions))

        self._upsert_points(self.config.code_files_collection, file_points)
        self._upsert_points(self.config.code_functions_collection, function_points)
        logger.info(
            "Indexed code tree %s: files=%s functions=%s call_edges=%s.",
            root,
            len(file_points),
            len(function_points),
            edge_count,
        )
        return CodeIndexStats(
            files=len(file_points),
            functions=len(function_points),
            call_edges=edge_count,
        )

    def index_repositories(
        self,
        repositories: Sequence[CodeRepository] | None = None,
        *,
        recreate: bool = False,
        clear_existing: bool = False,
    ) -> CodeIndexStats:
        selected = list(repositories or discover_code_repositories(self.config))
        if recreate:
            self.ensure_collections(recreate=True)
            self._clear_postgres_tables()
        else:
            self.ensure_collections(recreate=False)

        total = CodeIndexStats()
        for repository in selected:
            if clear_existing:
                self._delete_repository_index(repository.id)
            stats = self.index_source_tree(
                repository.source_dir,
                recreate=False,
                repository=repository,
            )
            total = CodeIndexStats(
                files=total.files + stats.files,
                functions=total.functions + stats.functions,
                call_edges=total.call_edges + stats.call_edges,
            )
        return total

    def ensure_collections(self, *, recreate: bool = False) -> None:
        collection_names = {c.name for c in self.client.get_collections().collections}
        for collection in (
            self.config.code_files_collection,
            self.config.code_functions_collection,
        ):
            if recreate and collection in collection_names:
                self.client.delete_collection(collection_name=collection)
                collection_names.remove(collection)

            if collection in collection_names:
                current_size = self._collection_vector_size(collection)
                expected_size = self.embedder.vector_size
                if current_size and current_size != expected_size:
                    logger.warning(
                        "Code collection %s vector size mismatch: current=%s "
                        "expected=%s. Recreating collection.",
                        collection,
                        current_size,
                        expected_size,
                    )
                    self.client.delete_collection(collection_name=collection)
                    collection_names.remove(collection)

            if collection not in collection_names:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.embedder.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

        for collection in (
            self.config.code_files_collection,
            self.config.code_functions_collection,
        ):
            self._ensure_payload_index(collection, "repository_id")
            self._ensure_payload_index(collection, "path")
            self._ensure_payload_index(collection, "language")
        self._ensure_payload_index(self.config.code_functions_collection, "file_id")
        self._ensure_payload_index(self.config.code_functions_collection, "kind")

    def _collection_vector_size(self, collection: str) -> int | None:
        try:
            info = self.client.get_collection(collection_name=collection)
        except Exception:
            return None
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            for value in vectors.values():
                if hasattr(value, "size"):
                    return int(value.size)
        return None

    def _ensure_payload_index(self, collection: str, field_name: str) -> None:
        try:
            info = self.client.get_collection(collection_name=collection)
            payload_schema = getattr(info, "payload_schema", {}) or {}
            if field_name in payload_schema:
                return
        except Exception:
            pass

        self.client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def _file_point(self, record: CodeFileRecord) -> PointStruct:
        text = _limit_text(record.embedding_text, self.config.code_file_embedding_max_chars)
        vector = self.embedder.embed([text])[0]
        return PointStruct(
            id=str(record.id),
            vector=vector,
            payload={
                "file_id": str(record.id),
                "repository_id": record.repository_id,
                "repository_name": record.repository_name,
                "source_root": record.source_root,
                "path": record.path,
                "language": record.language,
                "line_count": record.line_count,
                "content_sha256": record.content_sha256,
                "text": _limit_text(record.full_content, self.config.code_payload_snippet_chars),
            },
        )

    def _function_points(
        self,
        records: Sequence[CodeFunctionRecord],
    ) -> list[PointStruct]:
        points: list[PointStruct] = []
        for batch in _batched(list(records), self.config.code_embedding_batch_size):
            texts = [
                _limit_text(
                    record.embedding_text,
                    self.config.code_function_embedding_max_chars,
                )
                for record in batch
            ]
            vectors = self.embedder.embed(texts)
            for record, vector in zip(batch, vectors):
                points.append(
                    PointStruct(
                        id=str(record.id),
                        vector=vector,
                        payload={
                            "function_id": str(record.id),
                            "file_id": str(record.file_id),
                            "repository_id": record.repository_id,
                            "repository_name": record.repository_name,
                            "path": record.file_path,
                            "language": record.language,
                            "name": record.name,
                            "qualified_name": record.qualified_name,
                            "kind": record.kind,
                            "signature": record.signature,
                            "docstring": record.docstring,
                            "start_line": record.start_line,
                            "end_line": record.end_line,
                            "text": _limit_text(
                                record.body,
                                self.config.code_payload_snippet_chars,
                            ),
                        },
                    )
                )
        return points

    def _upsert_points(self, collection: str, points: list[PointStruct]) -> None:
        for batch in _batched(points, _QDRANT_BATCH_SIZE):
            self.client.upsert(
                collection_name=collection,
                points=batch,
                wait=True,
            )

    def _delete_repository_index(self, repository_id: str) -> None:
        for collection in (
            self.config.code_files_collection,
            self.config.code_functions_collection,
        ):
            self.client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="repository_id",
                            match=MatchValue(value=repository_id),
                        )
                    ]
                ),
                wait=True,
            )

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM code_files WHERE repository_id = %s",
                (repository_id,),
            )

    def _delete_qdrant_file_points(self, file_id: UUID) -> None:
        for collection in (
            self.config.code_files_collection,
            self.config.code_functions_collection,
        ):
            self.client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=str(file_id)),
                        )
                    ]
                ),
                wait=True,
            )

    def _delete_existing_file(
        self,
        conn: psycopg.Connection[Any],
        record: CodeFileRecord,
    ) -> None:
        conn.execute(
            """
            DELETE FROM code_files
            WHERE id = %s
               OR (repository_id = %s AND path = %s)
            """,
            (record.id, record.repository_id, record.path),
        )

    def _clear_postgres_tables(self) -> None:
        with self._connect() as conn:
            conn.execute("TRUNCATE code_call_edges, code_functions, code_files")

    def _upsert_file(
        self,
        conn: psycopg.Connection[Any],
        record: CodeFileRecord,
    ) -> None:
        conn.execute(
            """
            INSERT INTO code_files (
                id,
                repository_id,
                repository_name,
                source_root,
                path,
                language,
                full_content,
                content_sha256,
                line_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET repository_id = EXCLUDED.repository_id,
                repository_name = EXCLUDED.repository_name,
                source_root = EXCLUDED.source_root,
                path = EXCLUDED.path,
                language = EXCLUDED.language,
                full_content = EXCLUDED.full_content,
                content_sha256 = EXCLUDED.content_sha256,
                line_count = EXCLUDED.line_count,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (
                record.id,
                record.repository_id,
                record.repository_name,
                record.source_root,
                record.path,
                record.language,
                record.full_content,
                record.content_sha256,
                record.line_count,
            ),
        )

    def _upsert_functions(
        self,
        conn: psycopg.Connection[Any],
        records: Sequence[CodeFunctionRecord],
    ) -> None:
        for record in records:
            conn.execute(
                """
                INSERT INTO code_functions (
                    id,
                    file_id,
                    name,
                    qualified_name,
                    kind,
                    signature,
                    body,
                    docstring,
                    start_line,
                    end_line,
                    embedding_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET file_id = EXCLUDED.file_id,
                    name = EXCLUDED.name,
                    qualified_name = EXCLUDED.qualified_name,
                    kind = EXCLUDED.kind,
                    signature = EXCLUDED.signature,
                    body = EXCLUDED.body,
                    docstring = EXCLUDED.docstring,
                    start_line = EXCLUDED.start_line,
                    end_line = EXCLUDED.end_line,
                    embedding_text = EXCLUDED.embedding_text,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    record.id,
                    record.file_id,
                    record.name,
                    record.qualified_name,
                    record.kind,
                    record.signature,
                    record.body,
                    record.docstring,
                    record.start_line,
                    record.end_line,
                    record.embedding_text,
                ),
            )

    def _upsert_call_edges(
        self,
        conn: psycopg.Connection[Any],
        records: Sequence[CodeFunctionRecord],
    ) -> int:
        from app.call_graph import extract_call_edges

        edges = extract_call_edges(records)
        for edge in edges:
            conn.execute(
                """
                INSERT INTO code_call_edges (
                    caller_id,
                    caller_name,
                    callee_name,
                    file_id,
                    call_line
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (caller_id, callee_name, call_line) DO NOTHING
                """,
                (
                    edge.caller_id,
                    edge.caller_name,
                    edge.callee_name,
                    edge.file_id,
                    edge.call_line,
                ),
            )
        return len(edges)

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self.config.database_url,
            row_factory=dict_row,
            connect_timeout=self.config.database_connect_timeout_seconds,
        )


def parse_code_file(
    file_path: Path,
    source_root: Path,
    repository: CodeRepository,
) -> tuple[CodeFileRecord, list[CodeFunctionRecord]] | None:
    language = _language_for_path(file_path)
    if language is None:
        return None

    content = file_path.read_text(encoding="utf-8", errors="replace")
    path = _relative_display_path(file_path, source_root)
    file_id = uuid5(NAMESPACE_URL, f"code-file:{repository.id}:{path}")
    record = CodeFileRecord(
        id=file_id,
        repository_id=repository.id,
        repository_name=repository.name,
        path=path,
        source_root=repository.source_root,
        language=language,
        full_content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
        line_count=len(content.splitlines()),
    )
    functions = parse_code_functions(record, content)
    return record, functions


def parse_code_functions(
    code_file: CodeFileRecord,
    content: str,
) -> list[CodeFunctionRecord]:
    parser = _get_parser(code_file.language)
    source = content.encode("utf-8", errors="replace")
    tree = parser.parse(source)
    if code_file.language == "python":
        return _extract_python_symbols(code_file, source, tree.root_node)
    return _extract_cpp_symbols(code_file, source, tree.root_node)


def _extract_python_symbols(
    code_file: CodeFileRecord,
    source: bytes,
    root: object,
) -> list[CodeFunctionRecord]:
    records: list[CodeFunctionRecord] = []

    def visit(node: object, scope: list[str]) -> None:
        node_type = getattr(node, "type", "")
        if node_type == "class_definition":
            class_name = _node_text(_child_by_field(node, "name"), source)
            if class_name:
                records.append(
                    _record_from_node(
                        code_file,
                        source,
                        node,
                        name=class_name,
                        qualified_name=".".join(scope + [class_name]),
                        kind="class",
                        signature=_python_signature(node, source),
                        docstring=_python_docstring(node, source),
                    )
                )
                body = _child_by_field(node, "body")
                if body is not None:
                    visit(body, scope + [class_name])
                    return
        elif node_type == "function_definition":
            name = _node_text(_child_by_field(node, "name"), source)
            if name:
                records.append(
                    _record_from_node(
                        code_file,
                        source,
                        node,
                        name=name,
                        qualified_name=".".join(scope + [name]),
                        kind="function",
                        signature=_python_signature(node, source),
                        docstring=_python_docstring(node, source),
                    )
                )
                body = _child_by_field(node, "body")
                if body is not None:
                    visit(body, scope + [name])
                    return

        for child in getattr(node, "children", []) or []:
            visit(child, scope)

    visit(root, [])
    return records


def _extract_cpp_symbols(
    code_file: CodeFileRecord,
    source: bytes,
    root: object,
) -> list[CodeFunctionRecord]:
    records: list[CodeFunctionRecord] = []

    def visit(node: object, scope: list[str]) -> None:
        node_type = getattr(node, "type", "")
        if node_type in {"class_specifier", "struct_specifier"}:
            class_name = _cpp_class_name(node, source)
            if class_name:
                records.append(
                    _record_from_node(
                        code_file,
                        source,
                        node,
                        name=class_name,
                        qualified_name="::".join(scope + [class_name]),
                        kind="class",
                        signature=_cpp_class_signature(node, source),
                        docstring="",
                    )
                )
                for child in getattr(node, "children", []) or []:
                    visit(child, scope + [class_name])
                return
        elif node_type == "function_definition":
            name = _cpp_function_name(node, source)
            if name and name not in _CPP_MACRO_FUNCTION_NAMES:
                qualified_name = name if "::" in name else "::".join(scope + [name])
                records.append(
                    _record_from_node(
                        code_file,
                        source,
                        node,
                        name=name.split("::")[-1],
                        qualified_name=qualified_name,
                        kind="function",
                        signature=_cpp_signature(node, source),
                        docstring="",
                    )
                )

        for child in getattr(node, "children", []) or []:
            visit(child, scope)

    visit(root, [])
    return records


def _record_from_node(
    code_file: CodeFileRecord,
    source: bytes,
    node: object,
    *,
    name: str,
    qualified_name: str,
    kind: str,
    signature: str,
    docstring: str,
) -> CodeFunctionRecord:
    start_line = int(getattr(node, "start_point")[0]) + 1
    end_line = int(getattr(node, "end_point")[0]) + 1
    body = _node_text(node, source)
    function_id = uuid5(
        NAMESPACE_URL,
        "code-function:"
        f"{code_file.repository_id}:{code_file.path}:"
        f"{qualified_name}:{kind}:{start_line}:{end_line}",
    )
    return CodeFunctionRecord(
        id=function_id,
        file_id=code_file.id,
        repository_id=code_file.repository_id,
        repository_name=code_file.repository_name,
        file_path=code_file.path,
        language=code_file.language,
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        signature=signature.strip(),
        body=body,
        docstring=docstring.strip(),
        start_line=start_line,
        end_line=end_line,
    )


def _python_signature(node: object, source: bytes) -> str:
    body = _child_by_field(node, "body")
    if body is None:
        return _first_line(_node_text(node, source))
    return _bytes_text(source, int(getattr(node, "start_byte")), int(getattr(body, "start_byte"))).rstrip()


def _python_docstring(node: object, source: bytes) -> str:
    body = _child_by_field(node, "body")
    if body is None:
        return ""
    for child in getattr(body, "children", []) or []:
        if getattr(child, "type", "") != "expression_statement":
            continue
        grand_children = getattr(child, "children", []) or []
        if not grand_children:
            continue
        first = grand_children[0]
        if getattr(first, "type", "") in {"string", "concatenated_string"}:
            return _strip_string_literal(_node_text(first, source))
        break
    return ""


def _cpp_signature(node: object, source: bytes) -> str:
    body = _child_by_field(node, "body")
    if body is None:
        return _first_line(_node_text(node, source))
    return _bytes_text(source, int(getattr(node, "start_byte")), int(getattr(body, "start_byte"))).rstrip()


def _cpp_class_signature(node: object, source: bytes) -> str:
    body = _child_by_field(node, "body")
    if body is None:
        return _first_line(_node_text(node, source))
    return _bytes_text(source, int(getattr(node, "start_byte")), int(getattr(body, "start_byte"))).rstrip()


def _cpp_class_name(node: object, source: bytes) -> str:
    field = _child_by_field(node, "name")
    if field is not None:
        return _node_text(field, source)
    for child in getattr(node, "children", []) or []:
        if getattr(child, "type", "") in {"type_identifier", "identifier"}:
            return _node_text(child, source)
    return ""


def _cpp_function_name(node: object, source: bytes) -> str:
    declarator = _child_by_field(node, "declarator")
    if declarator is None:
        return ""
    name_node = _cpp_declarator_name(declarator)
    return _node_text(name_node, source).strip()


def _cpp_declarator_name(node: object | None) -> object | None:
    if node is None:
        return None
    if getattr(node, "type", "") in {
        "qualified_identifier",
        "identifier",
        "field_identifier",
        "operator_name",
        "destructor_name",
    }:
        return node

    for field_name in ("declarator", "name"):
        child = _child_by_field(node, field_name)
        found = _cpp_declarator_name(child)
        if found is not None:
            return found

    return None


def _get_parser(language: str) -> object:
    try:
        from tree_sitter_language_pack import get_parser
    except Exception as exc:
        raise RuntimeError(
            "tree-sitter-language-pack is required for code indexing."
        ) from exc

    parser_name = _TREE_SITTER_LANGUAGES.get(language)
    if not parser_name:
        raise ValueError(f"Unsupported code language: {language}")
    return get_parser(parser_name)


def _child_by_field(node: object | None, field_name: str) -> object | None:
    if node is None or not hasattr(node, "child_by_field_name"):
        return None
    return node.child_by_field_name(field_name)


def _node_text(node: object | None, source: bytes) -> str:
    if node is None:
        return ""
    return _bytes_text(
        source,
        int(getattr(node, "start_byte")),
        int(getattr(node, "end_byte")),
    )


def _bytes_text(source: bytes, start: int, end: int) -> str:
    return source[start:end].decode("utf-8", errors="replace")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _strip_string_literal(text: str) -> str:
    stripped = text.strip()
    prefixes = "rRuUbBfF"
    while stripped and stripped[0] in prefixes:
        stripped = stripped[1:].lstrip()
    for quote in ('"""', "'''", '"', "'"):
        if stripped.startswith(quote) and stripped.endswith(quote):
            return stripped[len(quote) : -len(quote)].strip()
    return stripped


def _looks_like_repository(path: Path) -> bool:
    if any((path / marker).exists() for marker in _REPOSITORY_MARKERS):
        return True
    return any(
        child.is_file() and _language_for_path(child) is not None
        for child in path.iterdir()
    )


def _contains_supported_code(path: Path, *, max_files: int = 2000) -> bool:
    checked = 0
    for candidate in path.rglob("*"):
        try:
            relative_parts = candidate.relative_to(path).parts
        except ValueError:
            relative_parts = candidate.parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if not candidate.is_file():
            continue
        checked += 1
        if _language_for_path(candidate) is not None:
            return True
        if checked >= max_files:
            return False
    return False


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _relative_display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return _display_path(path)


def _language_for_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in _PYTHON_EXTENSIONS:
        return "python"
    if suffix in _CPP_EXTENSIONS:
        return "cpp"
    return None


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _batched[T](items: Sequence[T], size: int) -> Iterable[list[T]]:
    for offset in range(0, len(items), size):
        yield list(items[offset : offset + size])
