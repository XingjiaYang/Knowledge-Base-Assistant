from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import re
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


logger = logging.getLogger(__name__)

ContentType = Literal["text", "code", "table"]


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    chunk_id: int
    score: float
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


_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_DIVIDER_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_MARKDOWN = MarkdownIt("gfm-like", {"linkify": False})


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
            part
            for part in [previous_tail, part.strip()]
            if part
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
    return text[-max_chars:].strip()


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

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.info("Connecting to Qdrant at %s.", self.config.qdrant_url)
            self._client = QdrantClient(url=self.config.qdrant_url)
        return self._client

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model %s.", self.config.embedding_model)
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

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

        if self.config.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self.client.create_payload_index(
            collection_name=self.config.collection_name,
            field_name="source_key",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def ingest_markdown_dir(
        self,
        docs_dir: Path | None = None,
        recreate: bool = False,
    ) -> int:
        docs_path = docs_dir or self.config.docs_dir
        self.ensure_collection(recreate=recreate)
        logger.info("Ingesting Markdown documents from %s.", docs_path)

        file_points = [
            (file_path, self._points_for_file(file_path))
            for file_path in sorted(docs_path.glob("*.md"))
        ]

        inserted = 0
        for file_path, points in file_points:
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
        return inserted

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        limit = top_k or self.config.retrieve_top_k
        query_vector = self._embed_one(query)
        query_filter = self._metadata_filter(metadata_filter)
        logger.info(
            "Running vector search: top_k=%s metadata_filter_keys=%s.",
            limit,
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
                if query_filter is not None:
                    search_args["query_filter"] = query_filter
                hits = self.client.search(**search_args)
        except Exception:
            logger.exception("Qdrant vector search failed.")
            raise

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    text=str(payload.get("text", "")),
                    source=self._public_source(str(payload.get("source", ""))),
                    chunk_id=int(payload.get("chunk_id", -1)),
                    score=float(hit.score),
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

    def _points_for_file(self, file_path: Path) -> list[PointStruct]:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return []

        embeddings = self.model.encode(
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

    @staticmethod
    def _display_source(file_path: Path) -> str:
        resolved = file_path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
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
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool = True,
    ) -> object:
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

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
