from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass, replace
import logging
import math
from pathlib import Path
import re
from threading import Lock
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from markdown_it import MarkdownIt
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
from sentence_transformers import SentenceTransformer

from app.config import PROJECT_ROOT, Settings, settings
from app.device import preferred_torch_device


logger = logging.getLogger(__name__)

ContentType = Literal["text", "code", "table"]


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    chunk_id: int
    score: float
    rerank_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    retrieval_source: str = "vector"
    content_type: str = "text"
    h1: str = ""
    h2: str = ""
    h3: str = ""
    headings: tuple[str, ...] = ()
    start_line: int = 0
    end_line: int = 0


@dataclass(frozen=True)
class MarkdownBlock:
    text: str
    content_type: ContentType
    headings: tuple[str, ...]
    h1: str
    h2: str
    h3: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class MarkdownChunk:
    text: str
    embedding_text: str
    content_type: ContentType
    headings: tuple[str, ...]
    h1: str
    h2: str
    h3: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _BM25Document:
    result: SearchResult
    term_counts: dict[str, int]
    length: int


@dataclass(frozen=True)
class _BM25Index:
    signature: tuple[tuple[str, int, int], ...]
    documents: tuple[_BM25Document, ...]
    document_frequencies: dict[str, int]
    average_length: float


_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_DIVIDER_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CJK_BLOCK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_MARKDOWN = MarkdownIt("gfm-like", {"linkify": False})
_BM25_K1 = 1.5
_BM25_B = 0.75
_RRF_RANK_CONSTANT = 60


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    return [
        chunk.embedding_text
        for chunk in chunk_markdown(text, chunk_size=chunk_size, overlap=overlap)
    ]


def chunk_markdown(text: str, chunk_size: int, overlap: int) -> list[MarkdownChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if overlap and overlap >= chunk_size - 1:
        raise ValueError("overlap must leave room for non-overlap text")

    blocks = _markdown_blocks(text)
    if not blocks:
        return []

    chunks: list[MarkdownChunk] = []
    effective_chunk_size = _effective_chunk_size(chunk_size, overlap)

    for block in blocks:
        if block.content_type == "code":
            parts = _split_code_block(block.text, chunk_size)
        elif block.content_type == "table":
            parts = _split_table_block(block.text, chunk_size)
        else:
            parts = _split_with_overlap(block.text, effective_chunk_size, overlap)

        chunks.extend(_chunk_from_block(block, part) for part in parts if part)

    return chunks


def _effective_chunk_size(chunk_size: int, overlap: int) -> int:
    if not overlap:
        return chunk_size
    # Reserve one separator character for the overlapped tail that is prepended.
    return chunk_size - overlap - 1


def _markdown_blocks(text: str) -> list[MarkdownBlock]:
    lines = text.splitlines()
    tokens = _MARKDOWN.parse(text)
    heading_stack: list[tuple[int, str]] = []
    blocks: list[MarkdownBlock] = []

    for idx, token in enumerate(tokens):
        if token.type == "heading_open":
            inline_token = tokens[idx + 1] if idx + 1 < len(tokens) else None
            level = int(token.tag[1]) if token.tag.startswith("h") else 1
            heading = inline_token.content.strip() if inline_token else ""
            if not heading:
                continue
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            continue

        if token.level != 0 or token.map is None:
            continue

        raw = _slice_lines(lines, token.map[0], token.map[1]).strip()
        if not raw:
            continue

        headings, h1, h2, h3 = _heading_metadata(heading_stack)
        blocks.append(
            MarkdownBlock(
                text=raw,
                content_type=_content_type(token.type),
                headings=headings,
                h1=h1,
                h2=h2,
                h3=h3,
                start_line=token.map[0] + 1,
                end_line=token.map[1],
            )
        )

    return blocks


def _heading_metadata(
    heading_stack: list[tuple[int, str]],
) -> tuple[tuple[str, ...], str, str, str]:
    headings = tuple(heading for _, heading in heading_stack)
    by_level = {level: heading for level, heading in heading_stack if level <= 3}
    return headings, by_level.get(1, ""), by_level.get(2, ""), by_level.get(3, "")


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end])


def _content_type(token_type: str) -> ContentType:
    if token_type in {"fence", "code_block"}:
        return "code"
    if token_type == "table_open":
        return "table"
    return "text"


def _chunk_from_block(block: MarkdownBlock, text: str) -> MarkdownChunk:
    return MarkdownChunk(
        text=text,
        embedding_text=_embedding_text(block, text),
        content_type=block.content_type,
        headings=block.headings,
        h1=block.h1,
        h2=block.h2,
        h3=block.h3,
        start_line=block.start_line,
        end_line=block.end_line,
    )


def _embedding_text(block: MarkdownBlock, text: str) -> str:
    parts = []
    if block.headings:
        parts.append(f"Headings: {' > '.join(block.headings)}")
    parts.append(f"Content type: {block.content_type}")
    parts.append(text)
    return "\n\n".join(parts).strip()


def _split_with_overlap(
    text: str,
    effective_chunk_size: int,
    overlap: int,
) -> list[str]:
    effective_chunk_size = max(1, effective_chunk_size)
    parts = _word_boundary_chunks(text, effective_chunk_size)
    if not parts or overlap <= 0:
        return parts

    chunks: list[str] = []
    previous_tail = ""
    for part in parts:
        chunk = " ".join(
            segment
            for segment in [previous_tail, part.strip()]
            if segment
        )
        chunks.append(chunk)
        previous_tail = _tail_text(part, overlap)
    return chunks


def _split_code_block(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    lines = text.splitlines()
    if len(lines) < 2 or not _FENCE_RE.match(lines[0]):
        return _word_boundary_chunks(text, chunk_size)

    opening = lines[0]
    closing = lines[-1] if _FENCE_RE.match(lines[-1]) else lines[0]
    body_lines = lines[1:-1] if closing == lines[-1] else lines[1:]
    body_budget = max(1, chunk_size - len(opening) - len(closing) - 2)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append("\n".join([opening, *current, closing]).strip())
        current = []
        current_len = 0

    for line in body_lines:
        line_len = len(line) + 1
        if line_len > body_budget:
            flush()
            for part in _word_boundary_chunks(line, body_budget):
                chunks.append("\n".join([opening, part, closing]).strip())
            continue

        if current and current_len + line_len > body_budget:
            flush()

        current.append(line)
        current_len += line_len

    flush()
    return chunks


def _split_table_block(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    lines = text.splitlines()
    if len(lines) < 3 or not _TABLE_DIVIDER_RE.match(lines[1]):
        return _word_boundary_chunks(text, chunk_size)

    header = lines[:2]
    rows = lines[2:]
    header_text = "\n".join(header)
    row_budget = max(1, chunk_size - len(header_text) - 1)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append("\n".join([*header, *current]).strip())
        current = []
        current_len = 0

    for row in rows:
        row_len = len(row) + 1
        if row_len > row_budget:
            flush()
            for part in _word_boundary_chunks(row, row_budget):
                chunks.append("\n".join([*header, part]).strip())
            continue

        if current and current_len + row_len > row_budget:
            flush()

        current.append(row)
        current_len += row_len

    flush()
    return chunks


def _word_boundary_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for token in re.findall(r"\S+\s*", text):
        if current and current_len + len(token) > max_chars:
            chunks.append("".join(current).strip())
            current = []
            current_len = 0

        while len(token) > max_chars:
            if current:
                chunks.append("".join(current).strip())
                current = []
                current_len = 0
            chunks.append(token[:max_chars].strip())
            token = token[max_chars:]

        current.append(token)
        current_len += len(token)

    if current:
        chunks.append("".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _join_chunk_parts(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip()).strip()


def _tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    tokens = re.findall(r"\S+\s*", text.strip())
    tail_tokens: list[str] = []
    current_len = 0
    for token in reversed(tokens):
        token_len = len(token)
        if token_len > max_chars and not tail_tokens:
            return token[-max_chars:].strip()
        if current_len + token_len > max_chars:
            break
        tail_tokens.append(token)
        current_len += token_len

    return "".join(reversed(tail_tokens)).strip()


def _bm25_tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _ASCII_TOKEN_RE.findall(lowered)
    for block in _CJK_BLOCK_RE.findall(text):
        tokens.extend(block)
        tokens.extend(
            block[index : index + 2]
            for index in range(max(0, len(block) - 1))
        )
    return [token for token in tokens if token]


class VectorStore:
    def __init__(
        self,
        config: Settings = settings,
        client: QdrantClient | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._model = model
        self._model_device = "cpu" if model is not None else None
        self._client_lock = Lock()
        self._model_lock = Lock()
        self._bm25_lock = Lock()
        self._bm25_index: _BM25Index | None = None
        self._ensured_payload_indexes: set[str] = set()

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    logger.info("Connecting to Qdrant at %s.", self.config.qdrant_url)
                    self._client = QdrantClient(url=self.config.qdrant_url)
        return self._client

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    device = preferred_torch_device(
                        self.config.cuda_enabled,
                        "Embedding model",
                    )
                    logger.info(
                        "Loading embedding model %s on %s.",
                        self.config.embedding_model,
                        device,
                    )
                    try:
                        self._model = SentenceTransformer(
                            self.config.embedding_model,
                            device=device,
                        )
                        self._model_device = device
                    except Exception as exc:
                        if device != "cuda":
                            raise
                        logger.warning(
                            "Embedding model failed to load on CUDA; "
                            "falling back to CPU: %s",
                            exc,
                        )
                        self._model = SentenceTransformer(
                            self.config.embedding_model,
                            device="cpu",
                        )
                        self._model_device = "cpu"
        return self._model

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
            self._ensured_payload_indexes.clear()
        with self._bm25_lock:
            self._bm25_index = None

        if client is not None and hasattr(client, "close"):
            client.close()

    @property
    def vector_size(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            size = self.model.get_embedding_dimension()
        else:
            size = self.model.get_sentence_embedding_dimension()
        if size is None:
            raise RuntimeError("Embedding model did not report a vector dimension")
        return int(size)

    def ensure_collection(self, recreate: bool = False) -> None:
        logger.info(
            "Ensuring Qdrant collection %s (recreate=%s).",
            self.config.collection_name,
            recreate,
        )
        collection_names = {c.name for c in self.client.get_collections().collections}

        if recreate and self.config.collection_name in collection_names:
            self.client.delete_collection(collection_name=self.config.collection_name)
            collection_names.remove(self.config.collection_name)
            self._ensured_payload_indexes.clear()

        if self.config.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self._ensure_payload_index("source_key")

    def ingest_markdown_dir(
        self,
        docs_dir: Path | None = None,
        recreate: bool = False,
    ) -> int:
        docs_path = docs_dir or self.config.docs_dir
        self.ensure_collection(recreate=recreate)
        logger.info("Ingesting Markdown documents from %s.", docs_path)

        inserted = 0
        for file_path in sorted(docs_path.rglob("*.md")):
            points = self._points_for_file(file_path)
            if not recreate:
                self._delete_file_points(file_path)
            if not points:
                continue

            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )
            inserted += len(points)
        logger.info(
            "Ingested %s chunks into %s.",
            inserted,
            self.config.collection_name,
        )
        with self._bm25_lock:
            self._bm25_index = None
        return inserted

    def _ensure_payload_index(self, field_name: str) -> None:
        if field_name in self._ensured_payload_indexes:
            return

        if self._payload_index_exists(field_name):
            self._ensured_payload_indexes.add(field_name)
            return

        self.client.create_payload_index(
            collection_name=self.config.collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        self._ensured_payload_indexes.add(field_name)

    def _payload_index_exists(self, field_name: str) -> bool:
        try:
            collection_info = self.client.get_collection(
                collection_name=self.config.collection_name,
            )
        except Exception:
            return False

        payload_schema = getattr(collection_info, "payload_schema", {}) or {}
        return field_name in payload_schema

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        limit = top_k or self.config.retrieve_top_k
        score_threshold = self.config.retrieve_score_threshold
        query_vector = self._embed_one(query)
        query_filter = self._metadata_filter(metadata_filter)
        logger.info(
            "Running vector search: top_k=%s score_threshold=%s "
            "metadata_filter_keys=%s.",
            limit,
            score_threshold or "disabled",
            sorted((metadata_filter or {}).keys()),
        )

        try:
            if hasattr(self.client, "query_points"):
                query_args = {
                    "collection_name": self.config.collection_name,
                    "query": query_vector,
                    "limit": limit,
                    "with_payload": True,
                }
                if score_threshold > 0:
                    query_args["score_threshold"] = score_threshold
                if query_filter is not None:
                    query_args["query_filter"] = query_filter
                response = self.client.query_points(**query_args)
                hits = response.points
            else:
                search_args = {
                    "collection_name": self.config.collection_name,
                    "query_vector": query_vector,
                    "limit": limit,
                    "with_payload": True,
                }
                if score_threshold > 0:
                    search_args["score_threshold"] = score_threshold
                if query_filter is not None:
                    search_args["query_filter"] = query_filter
                hits = self.client.search(**search_args)
        except Exception:
            logger.exception("Qdrant vector search failed.")
            raise

        results: list[SearchResult] = []
        for hit in hits:
            score = float(hit.score)
            if score_threshold > 0 and score < score_threshold:
                continue
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    text=str(payload.get("text", "")),
                    source=self._public_source(str(payload.get("source", ""))),
                    chunk_id=int(payload.get("chunk_id", -1)),
                    score=score,
                    vector_score=score,
                    retrieval_source="vector",
                    content_type=str(payload.get("content_type", "text")),
                    h1=str(payload.get("h1", "")),
                    h2=str(payload.get("h2", "")),
                    h3=str(payload.get("h3", "")),
                    headings=tuple(payload.get("headings", ())),
                    start_line=int(payload.get("start_line", 0)),
                    end_line=int(payload.get("end_line", 0)),
                )
            )
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        limit = max(1, top_k or self.config.bm25_top_k)
        index = self._ensure_bm25_index()
        if not index.documents:
            return []

        query_terms = set(_bm25_tokens(query))
        if not query_terms:
            return []

        metadata_filter = metadata_filter or {}
        document_count = len(index.documents)
        scored: list[tuple[float, SearchResult]] = []
        for document in index.documents:
            if not self._matches_result_metadata(document.result, metadata_filter):
                continue

            score = 0.0
            for term in query_terms:
                frequency = document.term_counts.get(term, 0)
                if not frequency:
                    continue

                document_frequency = index.document_frequencies.get(term, 0)
                idf = math.log(
                    1
                    + (
                        (document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                )
                denominator = frequency + _BM25_K1 * (
                    1
                    - _BM25_B
                    + _BM25_B * document.length / index.average_length
                )
                score += idf * (frequency * (_BM25_K1 + 1)) / denominator

            if score > 0:
                scored.append(
                    (
                        score,
                        replace(
                            document.result,
                            score=score,
                            bm25_score=score,
                            retrieval_source="bm25",
                        ),
                    )
                )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].source,
                -item[1].chunk_id,
            ),
            reverse=True,
        )
        logger.info("BM25 recall returned %s contexts.", min(len(scored), limit))
        return [result for _, result in scored[:limit]]

    def hybrid_search(
        self,
        query: str,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        rrf_top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        bm25_limit = max(1, bm25_top_k or self.config.bm25_top_k)
        vector_limit = max(1, vector_top_k or self.config.recall_top_k)
        fused_limit = max(1, rrf_top_k or self.config.rrf_top_k)

        bm25_results = self.search_bm25(
            query,
            top_k=bm25_limit,
            metadata_filter=metadata_filter,
        )
        vector_results = self.search(
            query,
            top_k=vector_limit,
            metadata_filter=metadata_filter,
        )
        fused = self._rrf_fuse(
            vector_results,
            bm25_results,
            top_k=fused_limit,
        )
        logger.info(
            "Hybrid recall fused bm25=%s vector=%s into rrf=%s contexts.",
            len(bm25_results),
            len(vector_results),
            len(fused),
        )
        return fused

    def _points_for_file(self, file_path: Path) -> list[PointStruct]:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return []

        embeddings = self._encode(
            [chunk.embedding_text for chunk in chunks],
            normalize_embeddings=True,
        )
        points: list[PointStruct] = []
        source_key = self._source_key(file_path)
        source = self._display_source(file_path)
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid5(NAMESPACE_URL, f"{source_key}:{idx}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "source": source,
                        "source_key": source_key,
                        "chunk_id": idx,
                        "text": chunk.text,
                        "content_type": chunk.content_type,
                        "h1": chunk.h1,
                        "h2": chunk.h2,
                        "h3": chunk.h3,
                        "headings": list(chunk.headings),
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                )
            )
        return points

    def _delete_file_points(self, file_path: Path) -> None:
        for key, value in (
            ("source_key", self._source_key(file_path)),
            ("source", self._display_source(file_path)),
            ("source", str(file_path)),
        ):
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    ]
                ),
                wait=True,
            )

    @staticmethod
    def _source_key(file_path: Path) -> str:
        return str(file_path.resolve())

    def _display_source(self, file_path: Path) -> str:
        resolved = file_path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            pass

        try:
            return resolved.relative_to(self.config.docs_dir.resolve()).as_posix()
        except ValueError:
            return file_path.name

    @staticmethod
    def _public_source(source: str) -> str:
        if not source:
            return ""

        source_path = Path(source)
        if not source_path.is_absolute():
            return source.replace("\\", "/")

        try:
            return source_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return source_path.name

    def _embed_one(self, text: str) -> list[float]:
        embedding = self._encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool = True,
    ) -> object:
        return self._encode(texts, normalize_embeddings=normalize_embeddings)

    def _encode(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool = True,
    ) -> object:
        try:
            return self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
            )
        except Exception as exc:
            if self._model_device != "cuda":
                raise
            logger.warning(
                "Embedding model failed during CUDA encode; falling back to CPU: %s",
                exc,
            )
            self._move_model_to_cpu()
            return self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
            )

    def _move_model_to_cpu(self) -> None:
        if self._model is not None and hasattr(self._model, "to"):
            self._model.to("cpu")
        self._model_device = "cpu"

    def _ensure_bm25_index(self) -> _BM25Index:
        signature = self._docs_signature()
        if self._bm25_index is not None and self._bm25_index.signature == signature:
            return self._bm25_index

        with self._bm25_lock:
            if (
                self._bm25_index is not None
                and self._bm25_index.signature == signature
            ):
                return self._bm25_index
            self._bm25_index = self._build_bm25_index(signature)
            return self._bm25_index

    def _build_bm25_index(
        self,
        signature: tuple[tuple[str, int, int], ...],
    ) -> _BM25Index:
        documents: list[_BM25Document] = []
        document_frequencies: Counter[str] = Counter()

        for file_path in sorted(self.config.docs_dir.rglob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_markdown(
                text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            source = self._display_source(file_path)
            for idx, chunk in enumerate(chunks):
                tokens = _bm25_tokens(chunk.embedding_text)
                if not tokens:
                    continue

                term_counts = Counter(tokens)
                document_frequencies.update(term_counts.keys())
                documents.append(
                    _BM25Document(
                        result=SearchResult(
                            text=chunk.text,
                            source=source,
                            chunk_id=idx,
                            score=0.0,
                            content_type=chunk.content_type,
                            h1=chunk.h1,
                            h2=chunk.h2,
                            h3=chunk.h3,
                            headings=chunk.headings,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            retrieval_source="bm25",
                        ),
                        term_counts=dict(term_counts),
                        length=len(tokens),
                    )
                )

        average_length = (
            sum(document.length for document in documents) / len(documents)
            if documents
            else 1.0
        )
        logger.info(
            "Built BM25 index from %s chunks under %s.",
            len(documents),
            self.config.docs_dir,
        )
        return _BM25Index(
            signature=signature,
            documents=tuple(documents),
            document_frequencies=dict(document_frequencies),
            average_length=max(average_length, 1.0),
        )

    def _docs_signature(self) -> tuple[tuple[str, int, int], ...]:
        docs_dir = self.config.docs_dir
        if not docs_dir.exists():
            return ()

        signature: list[tuple[str, int, int]] = []
        for file_path in sorted(docs_dir.rglob("*.md")):
            stat = file_path.stat()
            signature.append((str(file_path.resolve()), stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    @staticmethod
    def _rrf_fuse(
        vector_results: Sequence[SearchResult],
        bm25_results: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        by_key: dict[tuple[str, int], SearchResult] = {}
        rrf_scores: dict[tuple[str, int], float] = {}
        vector_scores: dict[tuple[str, int], float] = {}
        bm25_scores: dict[tuple[str, int], float] = {}
        sources: dict[tuple[str, int], set[str]] = {}

        for label, results in (("vector", vector_results), ("bm25", bm25_results)):
            for rank, result in enumerate(results, start=1):
                key = VectorStore._result_key(result)
                if key not in by_key or label == "vector":
                    by_key[key] = result
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (
                    1.0 / (_RRF_RANK_CONSTANT + rank)
                )
                sources.setdefault(key, set()).add(label)
                if label == "vector":
                    vector_scores[key] = (
                        result.vector_score
                        if result.vector_score is not None
                        else result.score
                    )
                else:
                    bm25_scores[key] = (
                        result.bm25_score
                        if result.bm25_score is not None
                        else result.score
                    )

        keys = sorted(
            rrf_scores,
            key=lambda key: (
                rrf_scores[key],
                vector_scores.get(key, float("-inf")),
                bm25_scores.get(key, float("-inf")),
                by_key[key].source,
                -by_key[key].chunk_id,
            ),
            reverse=True,
        )

        fused: list[SearchResult] = []
        for key in keys[: max(1, top_k)]:
            source_labels = sources.get(key, set())
            retrieval_source = "hybrid" if len(source_labels) > 1 else next(iter(source_labels))
            rrf_score = rrf_scores[key]
            fused.append(
                replace(
                    by_key[key],
                    score=rrf_score,
                    vector_score=vector_scores.get(key),
                    bm25_score=bm25_scores.get(key),
                    rrf_score=rrf_score,
                    retrieval_source=retrieval_source,
                )
            )
        return fused

    @staticmethod
    def _result_key(result: SearchResult) -> tuple[str, int]:
        return result.source, result.chunk_id

    @staticmethod
    def _matches_result_metadata(
        result: SearchResult,
        metadata_filter: dict[str, str],
    ) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if not expected:
                continue
            actual = getattr(result, key, "")
            if actual != expected:
                return False
        return True

    @staticmethod
    def _metadata_filter(
        metadata_filter: dict[str, str] | None,
    ) -> Filter | None:
        if not metadata_filter:
            return None

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in metadata_filter.items()
            if value
        ]
        if not conditions:
            return None
        return Filter(must=conditions)
