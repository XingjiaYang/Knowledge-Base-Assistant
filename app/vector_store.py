from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
import logging
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Literal, Protocol
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
import yaml

from app.config import PROJECT_ROOT, Settings, settings
from app.device import preferred_torch_device
from app.transformers_compat import patch_all_tied_weights_keys


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
class VectorSearchOutcome:
    results: list[SearchResult]
    embedding_ms: float = 0.0
    qdrant_ms: float = 0.0
    total_ms: float = 0.0


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
    body_token_count: int
    prefix_overlap_token_count: int
    suffix_overlap_token_count: int
    token_count: int


@dataclass(frozen=True)
class _BM25Index:
    signature: tuple[tuple[str, int, int], ...]
    retriever: Any
    documents: tuple[SearchResult, ...]


_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_DIVIDER_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CJK_BLOCK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_SENTENCE_RE = re.compile(
    r".+?(?:(?:[。！？]+[”’」』）】]*)|"
    r"(?:[.!?]+[\"'”’\)\]]*(?=\s|$))|\n+|$)",
    re.DOTALL,
)
_CLAUSE_RE = re.compile(
    r".+?(?:(?:[，；：、]+)|(?:[,;:]+(?=\s|$))|$)",
    re.DOTALL,
)
_MARKDOWN = MarkdownIt("gfm-like", {"linkify": False})
_RRF_RANK_CONSTANT = 60
CHUNKING_VERSION = "markdown-qwen3-token-section-overlap-v4"


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str: ...


class _NonFiniteEmbeddingError(RuntimeError):
    pass


def chunk_text(
    text: str,
    *,
    tokenizer: TokenizerLike,
    body_target_tokens: int,
    body_max_tokens: int,
    overlap_target_tokens: int,
    overlap_max_tokens: int,
) -> list[str]:
    return [
        chunk.embedding_text
        for chunk in chunk_markdown(
            text,
            tokenizer=tokenizer,
            body_target_tokens=body_target_tokens,
            body_max_tokens=body_max_tokens,
            overlap_target_tokens=overlap_target_tokens,
            overlap_max_tokens=overlap_max_tokens,
        )
    ]


def chunk_markdown(
    text: str,
    *,
    tokenizer: TokenizerLike,
    body_target_tokens: int,
    body_max_tokens: int,
    overlap_target_tokens: int,
    overlap_max_tokens: int,
) -> list[MarkdownChunk]:
    if body_max_tokens <= 0:
        raise ValueError("body_max_tokens must be greater than 0")
    if not 0 < body_target_tokens <= body_max_tokens:
        raise ValueError(
            "body_target_tokens must be between 1 and body_max_tokens"
        )
    if overlap_target_tokens < 0:
        raise ValueError("overlap_target_tokens must be greater than or equal to 0")
    if overlap_max_tokens < overlap_target_tokens:
        raise ValueError(
            "overlap_max_tokens must be greater than or equal to "
            "overlap_target_tokens"
        )

    body, frontmatter, line_offset = _extract_frontmatter(text)
    document_heading = str(
        frontmatter.get("title") or frontmatter.get("id") or ""
    ).strip()
    blocks = _markdown_blocks(
        body,
        initial_heading=document_heading,
        line_offset=line_offset,
    )
    if not blocks:
        return []

    cores = _pack_markdown_blocks(
        blocks,
        tokenizer=tokenizer,
        target_tokens=body_target_tokens,
        max_tokens=body_max_tokens,
    )
    return _chunks_with_sentence_overlap(
        cores,
        tokenizer=tokenizer,
        body_max_tokens=body_max_tokens,
        overlap_target_tokens=overlap_target_tokens,
        overlap_max_tokens=overlap_max_tokens,
    )


def _extract_frontmatter(text: str) -> tuple[str, dict[str, object], int]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text, {}, 0

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        ),
        -1,
    )
    if closing_index < 0:
        return text, {}, 0

    try:
        payload = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
    except yaml.YAMLError:
        logger.warning("Ignoring invalid Markdown YAML frontmatter.", exc_info=True)
        return text, {}, 0
    if not isinstance(payload, dict):
        logger.warning("Ignoring Markdown frontmatter that is not a mapping.")
        return text, {}, 0

    body = "\n".join(lines[closing_index + 1 :])
    metadata = {str(key): value for key, value in payload.items()}
    return body, metadata, closing_index + 1


def _markdown_blocks(
    text: str,
    *,
    initial_heading: str = "",
    line_offset: int = 0,
) -> list[MarkdownBlock]:
    lines = text.splitlines()
    tokens = _MARKDOWN.parse(text)
    heading_stack: list[tuple[int, str]] = (
        [(1, initial_heading)] if initial_heading else []
    )
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
                start_line=token.map[0] + 1 + line_offset,
                end_line=token.map[1] + line_offset,
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
    if token_type == "fence":
        return "code"
    if token_type == "table_open":
        return "table"
    # Long-form corpora commonly indent prose for visual layout. Only explicit
    # fenced blocks are treated as code so indentation does not fragment prose.
    return "text"


def _pack_markdown_blocks(
    blocks: list[MarkdownBlock],
    *,
    tokenizer: TokenizerLike,
    target_tokens: int,
    max_tokens: int,
) -> list[MarkdownBlock]:
    packed: list[MarkdownBlock] = []
    group: list[MarkdownBlock] = []
    group_key: tuple[tuple[str, ...], ContentType] | None = None

    def flush_group() -> None:
        nonlocal group
        if not group:
            return
        packed.extend(
            _pack_block_group(
                group,
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
            )
        )
        group = []

    for block in blocks:
        key = (block.headings, block.content_type)
        if group_key is not None and key != group_key:
            flush_group()
        group_key = key
        group.extend(
            _expand_block(
                block,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
        )
    flush_group()
    return packed


def _expand_block(
    block: MarkdownBlock,
    *,
    tokenizer: TokenizerLike,
    max_tokens: int,
) -> list[MarkdownBlock]:
    if _token_count(tokenizer, block.text) <= max_tokens:
        return [block]
    if block.content_type == "code":
        parts = _split_code_block(block.text, max_tokens, tokenizer)
    elif block.content_type == "table":
        parts = _split_table_block(block.text, max_tokens, tokenizer)
    else:
        parts = _long_paragraph_units(block.text, max_tokens, tokenizer)
    return [replace(block, text=part) for part in parts if part.strip()]


def _pack_block_group(
    blocks: list[MarkdownBlock],
    *,
    tokenizer: TokenizerLike,
    target_tokens: int,
    max_tokens: int,
) -> list[MarkdownBlock]:
    groups: list[list[MarkdownBlock]] = []
    current: list[MarkdownBlock] = []

    for block in blocks:
        current_text = _join_chunk_parts([item.text for item in current])
        candidate = _join_chunk_parts([*(item.text for item in current), block.text])
        current_tokens = _token_count(tokenizer, current_text)
        candidate_tokens = _token_count(tokenizer, candidate)
        should_flush = bool(current) and candidate_tokens > max_tokens
        if (
            current
            and candidate_tokens > target_tokens
            and abs(current_tokens - target_tokens)
            <= abs(candidate_tokens - target_tokens)
        ):
            should_flush = True
        if should_flush:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)

    cores = [_block_group_to_core(group) for group in groups if group]
    oversized = [
        _token_count(tokenizer, core.text)
        for core in cores
        if _token_count(tokenizer, core.text) > max_tokens
    ]
    if oversized:
        raise RuntimeError(f"Packed Markdown bodies exceeded token budget: {oversized}")
    return cores


def _block_group_to_core(group: list[MarkdownBlock]) -> MarkdownBlock:
    first = group[0]
    return replace(
        first,
        text=_join_chunk_parts([block.text for block in group]),
        start_line=min(block.start_line for block in group),
        end_line=max(block.end_line for block in group),
    )


def _long_paragraph_units(
    text: str,
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    units: list[str] = []
    for sentence in _sentence_units(text):
        if _token_count(tokenizer, sentence) <= max_tokens:
            units.append(sentence)
            continue
        clauses = _clause_units(sentence)
        if len(clauses) == 1 and clauses[0] == sentence:
            units.extend(
                _word_boundary_token_chunks(sentence, max_tokens, tokenizer)
            )
            continue
        for clause in clauses:
            if _token_count(tokenizer, clause) <= max_tokens:
                units.append(clause)
            else:
                units.extend(
                    _word_boundary_token_chunks(clause, max_tokens, tokenizer)
                )
    return [unit for unit in units if unit]


def _sentence_units(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in _SENTENCE_RE.finditer(text)
        if match.group(0).strip()
    ]


def _clause_units(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in _CLAUSE_RE.finditer(text)
        if match.group(0).strip()
    ]


def _word_boundary_token_chunks(
    text: str,
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for word in re.findall(r"\S+\s*", text):
        candidate = "".join([*current, word]).strip()
        if current and _token_count(tokenizer, candidate) > max_tokens:
            chunks.append("".join(current).strip())
            current = []
        if _token_count(tokenizer, word) > max_tokens:
            if current:
                chunks.append("".join(current).strip())
                current = []
            chunks.extend(_token_boundary_chunks(word, max_tokens, tokenizer))
            continue
        current.append(word)
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _chunks_with_sentence_overlap(
    cores: list[MarkdownBlock],
    *,
    tokenizer: TokenizerLike,
    body_max_tokens: int,
    overlap_target_tokens: int,
    overlap_max_tokens: int,
) -> list[MarkdownChunk]:
    chunks: list[MarkdownChunk] = []
    separator_tokens = _token_count(tokenizer, "\n\n")
    overlap_content_target = max(0, overlap_target_tokens - separator_tokens)
    overlap_content_max = max(0, overlap_max_tokens - separator_tokens)
    total_max_tokens = body_max_tokens + 2 * overlap_max_tokens
    for index, core in enumerate(cores):
        previous = cores[index - 1] if index > 0 else None
        following = cores[index + 1] if index + 1 < len(cores) else None
        prefix = ""
        suffix = ""
        if previous is not None and _can_overlap(previous, core):
            prefix = _sentence_overlap(
                previous.text,
                target_tokens=overlap_content_target,
                max_tokens=overlap_content_max,
                from_tail=True,
                tokenizer=tokenizer,
            )
        if following is not None and _can_overlap(core, following):
            suffix = _sentence_overlap(
                following.text,
                target_tokens=overlap_content_target,
                max_tokens=overlap_content_max,
                from_tail=False,
                tokenizer=tokenizer,
            )
        combined = _join_chunk_parts([prefix, core.text, suffix])
        body_tokens = _token_count(tokenizer, core.text)
        combined_tokens = _token_count(tokenizer, combined)
        if combined_tokens > total_max_tokens:
            raise RuntimeError(
                "Sentence overlap exceeded the derived maximum token budget: "
                f"{combined_tokens} > {total_max_tokens}."
            )
        chunks.append(
            _chunk_from_block(
                core,
                combined,
                body_token_count=body_tokens,
                prefix_overlap_token_count=_token_count(tokenizer, prefix),
                suffix_overlap_token_count=_token_count(tokenizer, suffix),
                token_count=combined_tokens,
            )
        )
    return chunks


def _can_overlap(left: MarkdownBlock, right: MarkdownBlock) -> bool:
    return (
        left.content_type == "text"
        and right.content_type == "text"
        and left.headings == right.headings
    )


def _sentence_overlap(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
    from_tail: bool,
    tokenizer: TokenizerLike,
) -> str:
    if target_tokens <= 0 or max_tokens <= 0:
        return ""
    units: list[str] = []
    for sentence in _sentence_units(text):
        if _token_count(tokenizer, sentence) <= max_tokens:
            units.append(sentence)
        else:
            units.extend(_long_paragraph_units(sentence, max_tokens, tokenizer))
    if not units:
        return ""
    ordered = list(reversed(units)) if from_tail else units
    selected: list[str] = []
    for unit in ordered:
        candidate = _join_chunk_parts([*selected, unit])
        if _token_count(tokenizer, candidate) > max_tokens:
            break
        selected.append(unit)
        if _token_count(tokenizer, candidate) >= target_tokens:
            break
    if from_tail:
        selected.reverse()
    return _join_chunk_parts(selected)


def _chunk_from_block(
    block: MarkdownBlock,
    text: str,
    *,
    body_token_count: int,
    prefix_overlap_token_count: int,
    suffix_overlap_token_count: int,
    token_count: int,
) -> MarkdownChunk:
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
        body_token_count=body_token_count,
        prefix_overlap_token_count=prefix_overlap_token_count,
        suffix_overlap_token_count=suffix_overlap_token_count,
        token_count=token_count,
    )


def _embedding_text(block: MarkdownBlock, text: str) -> str:
    parts = []
    if block.headings:
        parts.append(f"Headings: {' > '.join(block.headings)}")
    parts.append(f"Content type: {block.content_type}")
    parts.append(text)
    return "\n\n".join(parts).strip()


def _split_code_block(
    text: str,
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    if _token_count(tokenizer, text) <= max_tokens:
        return [text.strip()]

    lines = text.splitlines()
    if len(lines) < 2 or not _FENCE_RE.match(lines[0]):
        return _word_boundary_token_chunks(text, max_tokens, tokenizer)

    opening = lines[0]
    closing = lines[-1] if _FENCE_RE.match(lines[-1]) else lines[0]
    body_lines = lines[1:-1] if closing == lines[-1] else lines[1:]

    def render(parts: list[str]) -> str:
        return "\n".join([opening, *parts, closing]).strip()

    return _pack_wrapped_lines(body_lines, render, max_tokens, tokenizer)


def _split_table_block(
    text: str,
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    if _token_count(tokenizer, text) <= max_tokens:
        return [text.strip()]

    lines = text.splitlines()
    if len(lines) < 3 or not _TABLE_DIVIDER_RE.match(lines[1]):
        return _word_boundary_token_chunks(text, max_tokens, tokenizer)

    header = lines[:2]
    rows = lines[2:]

    def render(parts: list[str]) -> str:
        return "\n".join([*header, *parts]).strip()

    return _pack_wrapped_lines(rows, render, max_tokens, tokenizer)


def _pack_wrapped_lines(
    lines: list[str],
    render: Callable[[list[str]], str],
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        chunks.append(render(current))
        current = []

    for line in lines:
        candidate = render([*current, line])
        if _token_count(tokenizer, candidate) <= max_tokens:
            current.append(line)
            continue

        if current:
            flush()
        single = render([line])
        if _token_count(tokenizer, single) <= max_tokens:
            current.append(line)
            continue
        chunks.extend(
            _split_text_to_fit_wrapper(
                line,
                render=render,
                max_tokens=max_tokens,
                tokenizer=tokenizer,
            )
        )

    flush()
    return chunks


def _split_text_to_fit_wrapper(
    text: str,
    *,
    render: Callable[[list[str]], str],
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    token_ids = _token_ids(tokenizer, text)
    chunks: list[str] = []
    while token_ids:
        low = 1
        high = len(token_ids)
        best = 0
        best_text = ""
        while low <= high:
            middle = (low + high) // 2
            candidate_text = _decode_tokens(tokenizer, token_ids[:middle]).strip()
            if _token_count(tokenizer, render([candidate_text])) <= max_tokens:
                best = middle
                best_text = candidate_text
                low = middle + 1
            else:
                high = middle - 1
        if best <= 0:
            raise ValueError("Wrapper metadata leaves no room for content tokens.")
        chunks.append(render([best_text]))
        token_ids = token_ids[best:]
    return chunks


def _join_chunk_parts(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip()).strip()


def _token_ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded]


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    return len(_token_ids(tokenizer, text))


def _decode_tokens(tokenizer: TokenizerLike, token_ids: Sequence[int]) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=False))


def _token_boundary_chunks(
    text: str,
    max_tokens: int,
    tokenizer: TokenizerLike,
) -> list[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    token_ids = _token_ids(tokenizer, text)
    return [
        decoded.strip()
        for offset in range(0, len(token_ids), max_tokens)
        if (
            decoded := _decode_tokens(
                tokenizer,
                token_ids[offset : offset + max_tokens],
            )
        ).strip()
    ]


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
        chunk_tokenizer: TokenizerLike | None = None,
        use_ray: bool = True,
        document_store: object | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._model = model
        self._chunk_tokenizer = chunk_tokenizer
        self._model_device = "cpu" if model is not None else None
        self._use_ray = use_ray and model is None
        self._document_store = document_store
        self._client_lock = Lock()
        self._model_lock = Lock()
        self._chunk_tokenizer_lock = Lock()
        self._bm25_lock = Lock()
        self._bm25_index: _BM25Index | None = None
        self._bm25_candidate: tuple[str, _BM25Index] | None = None
        self._ensured_payload_indexes: set[str] = set()
        self._vector_size: int | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    logger.info("Connecting to Qdrant at %s.", self.config.qdrant_url)
                    self._client = QdrantClient(url=self.config.qdrant_url)
        return self._client

    @property
    def chunk_tokenizer(self) -> TokenizerLike:
        if self._chunk_tokenizer is None:
            with self._chunk_tokenizer_lock:
                if self._chunk_tokenizer is None:
                    from transformers import AutoTokenizer

                    logger.info(
                        "Loading chunk tokenizer %s.",
                        self.config.chunk_tokenizer_model,
                    )
                    self._chunk_tokenizer = AutoTokenizer.from_pretrained(
                        self.config.chunk_tokenizer_model,
                        trust_remote_code=(
                            self.config.chunk_tokenizer_trust_remote_code
                        ),
                        use_fast=True,
                    )
        return self._chunk_tokenizer

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
                    patch_all_tied_weights_keys()
                    model_kwargs = self._sentence_transformer_kwargs(device)
                    try:
                        self._model = SentenceTransformer(
                            self.config.embedding_model,
                            device=device,
                            trust_remote_code=self.config.embedding_trust_remote_code,
                            **model_kwargs,
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
                            trust_remote_code=self.config.embedding_trust_remote_code,
                            **self._sentence_transformer_kwargs("cpu"),
                        )
                        self._model_device = "cpu"
        return self._model

    def _sentence_transformer_kwargs(self, device: str) -> dict[str, object]:
        if "jina-embeddings-v5" not in self.config.embedding_model.lower():
            return {}

        kwargs: dict[str, object] = {}
        if device == "cuda":
            try:
                import torch
            except Exception:
                return kwargs

            kwargs["model_kwargs"] = {"dtype": torch.bfloat16}
        return kwargs

    def _embedding_actor(self) -> object | None:
        if not self._use_ray:
            return None
        try:
            from app.model_actors import get_embedding_actor

            return get_embedding_actor(self.config)
        except Exception:
            logger.exception("Embedding Ray actor setup failed.")
            return None

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
            self._ensured_payload_indexes.clear()
        with self._bm25_lock:
            self._bm25_index = None
            self._bm25_candidate = None

        if client is not None and hasattr(client, "close"):
            client.close()

    @property
    def vector_size(self) -> int:
        if self._vector_size is not None:
            return self._vector_size

        if self._use_ray:
            actor = self._embedding_actor()
            if actor is not None:
                try:
                    from app.model_actors import mark_ray_unavailable, ray_get

                    actual_size = int(ray_get(actor.vector_size.remote(), self.config))
                    if actual_size <= 0:
                        raise RuntimeError(
                            "Embedding actor returned an invalid vector dimension."
                        )
                    self._vector_size = actual_size
                    return actual_size
                except Exception:
                    logger.exception(
                        "Embedding Ray actor failed during dimension probe; "
                        "local_model_fallback=%s.",
                        self.config.ray_local_fallback,
                    )
                    mark_ray_unavailable(self.config.ray_embedding_actor_name)
                    if not self.config.ray_local_fallback:
                        raise RuntimeError(
                            "Embedding Ray actor failed during dimension probe and "
                            "RAY_LOCAL_FALLBACK=0."
                        )
            elif not self.config.ray_local_fallback:
                raise RuntimeError(
                    "Embedding Ray actor is unavailable and RAY_LOCAL_FALLBACK=0."
                )

        reported_size: int | None = None
        if hasattr(self.model, "get_embedding_dimension"):
            reported_size = self.model.get_embedding_dimension()
        else:
            reported_size = self.model.get_sentence_embedding_dimension()

        actual_size = self._probe_vector_size()
        if reported_size is not None and int(reported_size) != actual_size:
            logger.warning(
                "Embedding model reported dimension %s but produced %s; "
                "using produced dimension.",
                reported_size,
                actual_size,
            )
        if actual_size <= 0:
            raise RuntimeError("Embedding model did not report a vector dimension")
        self._vector_size = actual_size
        return actual_size

    def _probe_vector_size(self) -> int:
        embeddings = self._encode_matrix(
            "dimension probe",
            normalize_embeddings=True,
            task=self.config.embedding_passage_task,
            prompt_name=self.config.embedding_passage_prompt_name,
        )
        if len(embeddings) != 1 or not embeddings[0]:
            raise RuntimeError("Embedding model returned an empty probe vector.")
        return len(embeddings[0])

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

        if self.config.collection_name in collection_names:
            current_size = self._collection_vector_size()
            expected_size = self.vector_size
            if current_size and current_size != expected_size:
                logger.warning(
                    "Qdrant collection %s vector size mismatch: current=%s "
                    "expected=%s. Recreating collection.",
                    self.config.collection_name,
                    current_size,
                    expected_size,
                )
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

    def _collection_vector_size(self) -> int | None:
        try:
            collection_info = self.client.get_collection(
                collection_name=self.config.collection_name,
            )
        except Exception:
            return None

        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            for value in vectors.values():
                if hasattr(value, "size"):
                    return int(value.size)
        return None

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

            self._log_point_vector_shape(file_path, points)
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
            self._bm25_candidate = None
        return inserted

    @staticmethod
    def _log_point_vector_shape(file_path: Path, points: list[PointStruct]) -> None:
        if not points:
            return

        vector = points[0].vector
        vector_len = len(vector) if isinstance(vector, list) else 0
        first_value = vector[0] if vector_len else None
        logger.info(
            "Prepared %s vectors for %s: vector_type=%s vector_len=%s "
            "first_value_type=%s.",
            len(points),
            file_path,
            type(vector).__name__,
            vector_len,
            type(first_value).__name__,
        )

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
        return self.search_with_timing(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        ).results

    def search_with_timing(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> VectorSearchOutcome:
        total_start = time.perf_counter()
        limit = top_k or self.config.retrieve_top_k
        score_threshold = self.config.retrieve_score_threshold
        embedding_start = time.perf_counter()
        query_vector = self._embed_one(query)
        embedding_ms = (time.perf_counter() - embedding_start) * 1000
        query_filter = self._metadata_filter(metadata_filter)
        logger.info(
            "Running vector search: top_k=%s score_threshold=%s "
            "metadata_filter_keys=%s.",
            limit,
            score_threshold or "disabled",
            sorted((metadata_filter or {}).keys()),
        )

        try:
            qdrant_start = time.perf_counter()
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
            qdrant_ms = (time.perf_counter() - qdrant_start) * 1000
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
        return VectorSearchOutcome(
            results=results,
            embedding_ms=embedding_ms,
            qdrant_ms=qdrant_ms,
            total_ms=(time.perf_counter() - total_start) * 1000,
        )

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

        query_tokens = _bm25_tokens(query)
        if not query_tokens or index.retriever is None:
            return []

        metadata_filter = metadata_filter or {}
        search_limit = len(index.documents) if metadata_filter else limit
        response = index.retriever.retrieve(
            [query_tokens],
            k=min(max(1, search_limit), len(index.documents)),
            sorted=True,
            return_as="tuple",
            show_progress=False,
        )
        documents = list(response.documents[0])
        scores = list(response.scores[0])

        results: list[SearchResult] = []
        for document, score_value in zip(documents, scores):
            score = float(score_value)
            if score <= 0:
                continue
            if not isinstance(document, SearchResult):
                raise RuntimeError("BM25S returned an unexpected document payload.")
            if not self._matches_result_metadata(document, metadata_filter):
                continue
            results.append(
                replace(
                    document,
                    score=score,
                    bm25_score=score,
                    retrieval_source="bm25",
                )
            )
            if len(results) >= limit:
                break

        logger.info("BM25 recall returned %s contexts.", len(results))
        return results

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
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks = chunk_markdown(
            text,
            tokenizer=self.chunk_tokenizer,
            body_target_tokens=self.config.chunk_body_target_tokens,
            body_max_tokens=self.config.chunk_body_max_tokens,
            overlap_target_tokens=self.config.chunk_overlap_target_tokens,
            overlap_max_tokens=self.config.chunk_overlap_max_tokens,
        )
        if not chunks:
            return []

        embeddings = self._encode_matrix(
            [chunk.embedding_text for chunk in chunks],
            normalize_embeddings=True,
            task=self.config.embedding_passage_task,
            prompt_name=self.config.embedding_passage_prompt_name,
        )
        points: list[PointStruct] = []
        source_key = self._source_key(file_path)
        source = self._display_source(file_path)
        source_doc_id = str(uuid5(NAMESPACE_URL, source_key))
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid5(NAMESPACE_URL, f"{source_key}:{idx}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "source": source,
                        "source_key": source_key,
                        "source_doc_id": source_doc_id,
                        "version_id": content_hash,
                        "content_hash": content_hash,
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
        embeddings = self._encode_matrix(
            text,
            normalize_embeddings=True,
            task=self.config.embedding_query_task,
            prompt_name=self.config.embedding_query_prompt_name,
        )
        if len(embeddings) != 1:
            raise RuntimeError("Embedding model returned an unexpected query shape.")
        return embeddings[0]

    def encode(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool = True,
        task: str | None = None,
        prompt_name: str | None = None,
    ) -> object:
        embeddings = self._encode_matrix(
            texts,
            normalize_embeddings=normalize_embeddings,
            task=task or self.config.embedding_classification_task,
            prompt_name=(
                prompt_name
                if prompt_name is not None
                else self.config.embedding_classification_prompt_name
            ),
        )
        if isinstance(texts, str):
            if len(embeddings) != 1:
                raise RuntimeError("Embedding model returned an unexpected query shape.")
            return embeddings[0]
        return embeddings

    def _encode_matrix(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool,
        task: str | None,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        if self._use_ray:
            actor = self._embedding_actor()
            if actor is not None:
                try:
                    from app.model_actors import mark_ray_unavailable, ray_get

                    payload = texts if isinstance(texts, str) else list(texts)
                    return ray_get(
                        actor.encode_matrix.remote(
                            payload,
                            normalize_embeddings,
                            task,
                            prompt_name,
                        ),
                        self.config,
                    )
                except Exception:
                    logger.exception(
                        "Embedding Ray actor failed during encode; "
                        "local_model_fallback=%s.",
                        self.config.ray_local_fallback,
                    )
                    mark_ray_unavailable(self.config.ray_embedding_actor_name)
                    if not self.config.ray_local_fallback:
                        raise RuntimeError(
                            "Embedding Ray actor failed during encode and "
                            "RAY_LOCAL_FALLBACK=0."
                        )
            elif not self.config.ray_local_fallback:
                raise RuntimeError(
                    "Embedding Ray actor is unavailable and RAY_LOCAL_FALLBACK=0."
                )

        raw_embeddings = self._encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            task=task,
            prompt_name=prompt_name,
        )
        try:
            return self._embedding_matrix(raw_embeddings)
        except _NonFiniteEmbeddingError as exc:
            if self._model_device == "cuda":
                logger.warning(
                    "Embedding model returned non-finite values on CUDA; "
                    "falling back to CPU: %s",
                    exc,
                )
                self._move_model_to_cpu()
                raw_embeddings = self._encode(
                    texts,
                    normalize_embeddings=normalize_embeddings,
                    task=task,
                    prompt_name=prompt_name,
                )
                try:
                    return self._embedding_matrix(raw_embeddings)
                except _NonFiniteEmbeddingError:
                    pass

            logger.warning(
                "Embedding model returned non-finite values; sanitizing "
                "non-finite vector entries to zero."
            )
            return self._embedding_matrix(
                raw_embeddings,
                sanitize_nonfinite=True,
            )

    def _encode(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool = True,
        task: str | None = None,
        prompt_name: str | None = None,
    ) -> object:
        encode_kwargs = self._encode_kwargs(task, prompt_name=prompt_name)
        if not isinstance(texts, str):
            encode_kwargs["batch_size"] = min(
                len(texts),
                self.config.embedding_offline_batch_size,
            )
        try:
            return self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                **encode_kwargs,
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
                **encode_kwargs,
            )

    @classmethod
    def _embedding_matrix(
        cls,
        embeddings: object,
        sanitize_nonfinite: bool = False,
    ) -> list[list[float]]:
        value = cls._to_builtin(cls._unwrap_embedding_output(embeddings))
        if not cls._is_sequence(value):
            raise RuntimeError("Embedding model returned a non-sequence value.")

        rows = list(value)
        if not rows:
            return []

        first = cls._to_builtin(cls._unwrap_embedding_output(rows[0]))
        if not cls._is_sequence(first):
            return [
                cls._embedding_vector(
                    rows,
                    sanitize_nonfinite=sanitize_nonfinite,
                )
            ]
        return [
            cls._embedding_vector(
                row,
                sanitize_nonfinite=sanitize_nonfinite,
            )
            for row in rows
        ]

    @classmethod
    def _embedding_vector(
        cls,
        embedding: object,
        sanitize_nonfinite: bool = False,
    ) -> list[float]:
        value = cls._to_builtin(cls._unwrap_embedding_output(embedding))
        if not cls._is_sequence(value):
            raise RuntimeError("Embedding model returned a non-vector row.")

        items = list(value)
        if len(items) == 1:
            nested = cls._to_builtin(cls._unwrap_embedding_output(items[0]))
            if cls._is_sequence(nested):
                return cls._embedding_vector(
                    nested,
                    sanitize_nonfinite=sanitize_nonfinite,
                )

        vector: list[float] = []
        for item in items:
            item = cls._to_builtin(cls._unwrap_embedding_output(item))
            if cls._is_sequence(item):
                raise RuntimeError("Embedding model returned a nested vector row.")
            numeric_item = float(item)
            if not math.isfinite(numeric_item):
                if not sanitize_nonfinite:
                    raise _NonFiniteEmbeddingError(
                        "Embedding model returned a non-finite value."
                    )
                numeric_item = 0.0
            vector.append(numeric_item)
        if sanitize_nonfinite:
            norm = math.sqrt(sum(item * item for item in vector))
            if norm > 0:
                vector = [item / norm for item in vector]
        return vector

    @staticmethod
    def _unwrap_embedding_output(value: object) -> object:
        if not isinstance(value, dict):
            return value

        for key in (
            "dense_vecs",
            "dense",
            "sentence_embedding",
            "embeddings",
            "embedding",
            "vectors",
            "vector",
        ):
            if key in value:
                return value[key]

        if len(value) == 1:
            return next(iter(value.values()))
        raise RuntimeError(
            "Embedding model returned a mapping without a dense vector field."
        )

    @staticmethod
    def _to_builtin(value: object) -> object:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        )

    @staticmethod
    def _encode_kwargs(
        task: str | None,
        prompt_name: str | None = None,
    ) -> dict[str, object]:
        task = (task or "").strip()
        prompt_name = (prompt_name or "").strip()
        if task == "retrieval.query" and not prompt_name:
            return {"task": "retrieval", "prompt_name": "query"}
        if task == "retrieval.passage" and not prompt_name:
            return {"task": "retrieval", "prompt_name": "document"}

        kwargs: dict[str, object] = {}
        if task:
            kwargs["task"] = task
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        return kwargs

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

    def rebuild_bm25_index(self, *, expected_index_version: str = "") -> int:
        """Synchronously rebuild and publish one complete in-memory BM25S index."""
        with self._bm25_lock:
            signature = self._docs_signature()
            if expected_index_version:
                expected_prefix = f"s3-manifest:{expected_index_version}"
                if not signature or signature[0][0] != expected_prefix:
                    raise RuntimeError(
                        "Active S3 manifest does not match the expected BM25 version: "
                        f"expected={expected_index_version} signature={signature[:1]}."
                    )
            rebuilt = self._build_bm25_index(signature)
            if self._docs_signature() != signature:
                raise RuntimeError(
                    "Document manifest changed while the BM25S index was rebuilding."
                )
            self._bm25_index = rebuilt
            self._bm25_candidate = None
            return len(rebuilt.documents)

    def prepare_bm25_candidate(self, index_version: str) -> int:
        if self.config.docs_source != "s3":
            raise RuntimeError("Versioned BM25 candidates require DOCS_SOURCE=s3.")
        manifest = self._s3_document_store().load_version_manifest(index_version)
        if not manifest:
            raise RuntimeError(f"S3 index manifest is missing: {index_version}")
        if str(manifest.get("index_version", "")) != index_version:
            raise RuntimeError("S3 index manifest version does not match its key.")

        try:
            import bm25s
        except ImportError as exc:
            raise RuntimeError(
                "BM25S is required for keyword recall. Install requirements.api.txt."
            ) from exc

        signature = self._s3_docs_signature(manifest)
        candidate = self._build_bm25_index_from_s3_manifest(
            signature,
            bm25s,
            manifest=manifest,
        )
        latest_manifest = self._s3_document_store().load_version_manifest(
            index_version
        )
        if not latest_manifest or self._s3_docs_signature(latest_manifest) != signature:
            raise RuntimeError("S3 index manifest changed during BM25S preparation.")
        with self._bm25_lock:
            if (
                self._bm25_candidate is not None
                and self._bm25_candidate[0] != index_version
            ):
                raise RuntimeError(
                    "Another BM25S candidate is already prepared: "
                    f"{self._bm25_candidate[0]}."
                )
            self._bm25_candidate = (index_version, candidate)
        return len(candidate.documents)

    def activate_bm25_candidate(self, index_version: str) -> int:
        expected_prefix = f"s3-manifest:{index_version}"
        with self._bm25_lock:
            if (
                self._bm25_index is not None
                and self._bm25_index.signature
                and self._bm25_index.signature[0][0] == expected_prefix
            ):
                return len(self._bm25_index.documents)
            if self._bm25_candidate is None:
                raise RuntimeError("No prepared BM25S candidate is available.")
            candidate_version, candidate = self._bm25_candidate
            if candidate_version != index_version:
                raise RuntimeError(
                    "Prepared BM25S candidate version mismatch: "
                    f"expected={index_version} actual={candidate_version}."
                )
            self._bm25_index = candidate
            self._bm25_candidate = None
            return len(candidate.documents)

    def discard_bm25_candidate(self, index_version: str) -> None:
        with self._bm25_lock:
            if self._bm25_candidate and self._bm25_candidate[0] == index_version:
                self._bm25_candidate = None

    def active_bm25_index_version(self) -> str:
        with self._bm25_lock:
            if self._bm25_index is None or not self._bm25_index.signature:
                return ""
            marker = self._bm25_index.signature[0][0]
        prefix = "s3-manifest:"
        return marker[len(prefix) :] if marker.startswith(prefix) else ""

    def prepared_bm25_candidate_version(self) -> str:
        with self._bm25_lock:
            return self._bm25_candidate[0] if self._bm25_candidate else ""

    def _build_bm25_index(
        self,
        signature: tuple[tuple[str, int, int], ...],
    ) -> _BM25Index:
        try:
            import bm25s
        except ImportError as exc:
            raise RuntimeError(
                "BM25S is required for keyword recall. Install requirements.api.txt."
            ) from exc

        if self.config.docs_source == "s3":
            try:
                return self._build_bm25_index_from_s3_manifest(signature, bm25s)
            except Exception:
                logger.exception(
                    "Failed to build BM25S index from active S3 manifest; "
                    "falling back to Qdrant payload scroll."
                )
                return self._build_bm25_index_from_qdrant(signature, bm25s)

        documents: list[SearchResult] = []
        tokenized_documents: list[list[str]] = []

        for file_path in sorted(self.config.docs_dir.rglob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_markdown(
                text,
                tokenizer=self.chunk_tokenizer,
                body_target_tokens=self.config.chunk_body_target_tokens,
                body_max_tokens=self.config.chunk_body_max_tokens,
                overlap_target_tokens=self.config.chunk_overlap_target_tokens,
                overlap_max_tokens=self.config.chunk_overlap_max_tokens,
            )
            source = self._display_source(file_path)
            for idx, chunk in enumerate(chunks):
                tokens = _bm25_tokens(chunk.embedding_text)
                if not tokens:
                    continue

                documents.append(
                    SearchResult(
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
                    )
                )
                tokenized_documents.append(tokens)

        retriever = None
        if documents:
            retriever = bm25s.BM25(corpus=documents)
            retriever.index(tokenized_documents, show_progress=False)

        logger.info(
            "Built BM25S index from %s chunks under %s.",
            len(documents),
            self.config.docs_dir,
        )
        return _BM25Index(
            signature=signature,
            retriever=retriever,
            documents=tuple(documents),
        )

    def _build_bm25_index_from_qdrant(
        self,
        signature: tuple[tuple[str, int, int], ...],
        bm25s_module: object,
    ) -> _BM25Index:
        documents: list[SearchResult] = []
        tokenized_documents: list[list[str]] = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                text = str(payload.get("text", ""))
                if not text:
                    continue

                tokens = _bm25_tokens(self._bm25_payload_text(payload))
                if not tokens:
                    continue

                documents.append(
                    SearchResult(
                        text=text,
                        source=self._public_source(str(payload.get("source", ""))),
                        chunk_id=int(payload.get("chunk_id", -1)),
                        score=0.0,
                        content_type=str(payload.get("content_type", "text")),
                        h1=str(payload.get("h1", "")),
                        h2=str(payload.get("h2", "")),
                        h3=str(payload.get("h3", "")),
                        headings=tuple(payload.get("headings", ())),
                        start_line=int(payload.get("start_line", 0)),
                        end_line=int(payload.get("end_line", 0)),
                        retrieval_source="bm25",
                    )
                )
                tokenized_documents.append(tokens)

            if offset is None:
                break

        retriever = None
        if documents:
            retriever = bm25s_module.BM25(corpus=documents)
            retriever.index(tokenized_documents, show_progress=False)

        logger.info(
            "Built BM25S index from %s Qdrant chunks in %s.",
            len(documents),
            self.config.collection_name,
        )
        return _BM25Index(
            signature=signature,
            retriever=retriever,
            documents=tuple(documents),
        )

    def _build_bm25_index_from_s3_manifest(
        self,
        signature: tuple[tuple[str, int, int], ...],
        bm25s_module: object,
        *,
        manifest: dict[str, object] | None = None,
    ) -> _BM25Index:
        records = self._s3_active_records(manifest)
        documents: list[SearchResult] = []
        tokenized_documents: list[list[str]] = []

        document_store = self._s3_document_store()
        for record in records:
            text = document_store.read_markdown(record)
            chunks = chunk_markdown(
                text,
                tokenizer=self.chunk_tokenizer,
                body_target_tokens=self.config.chunk_body_target_tokens,
                body_max_tokens=self.config.chunk_body_max_tokens,
                overlap_target_tokens=self.config.chunk_overlap_target_tokens,
                overlap_max_tokens=self.config.chunk_overlap_max_tokens,
            )
            for idx, chunk in enumerate(chunks):
                tokens = _bm25_tokens(chunk.embedding_text)
                if not tokens:
                    continue

                documents.append(
                    SearchResult(
                        text=chunk.text,
                        source=record.source,
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
                    )
                )
                tokenized_documents.append(tokens)

        retriever = None
        if documents:
            retriever = bm25s_module.BM25(corpus=documents)
            retriever.index(tokenized_documents, show_progress=False)

        logger.info(
            "Built BM25S index from %s S3 manifest chunks in %s.",
            len(documents),
            self.config.docs_s3_bucket,
        )
        return _BM25Index(
            signature=signature,
            retriever=retriever,
            documents=tuple(documents),
        )

    def _docs_signature(self) -> tuple[tuple[str, int, int], ...]:
        if self.config.docs_source == "s3":
            try:
                return self._s3_docs_signature()
            except Exception:
                logger.exception(
                    "Failed to inspect active S3 document manifest for BM25."
                )
                return self._qdrant_docs_signature()

        docs_dir = self.config.docs_dir
        if not docs_dir.exists():
            return ()

        signature: list[tuple[str, int, int]] = []
        for file_path in sorted(docs_dir.rglob("*.md")):
            stat = file_path.stat()
            signature.append((str(file_path.resolve()), stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def _qdrant_docs_signature(self) -> tuple[tuple[str, int, int], ...]:
        try:
            active_collection = self._active_collection_name(self.config.collection_name)
            info = self.client.get_collection(collection_name=self.config.collection_name)
        except Exception:
            logger.exception(
                "Failed to inspect active Qdrant document collection for BM25."
            )
            return ()

        points_count = int(getattr(info, "points_count", 0) or 0)
        vectors_count = int(getattr(info, "vectors_count", 0) or 0)
        return ((f"qdrant:{active_collection}", points_count, vectors_count),)

    def _s3_document_store(self) -> object:
        if self._document_store is None:
            from app.s3_documents import S3DocumentStore

            self._document_store = S3DocumentStore(self.config)
        return self._document_store

    def _s3_active_records(
        self,
        manifest: dict[str, object] | None = None,
    ) -> list[object]:
        from app.s3_documents import S3DocumentRecord

        manifest = manifest or self._s3_document_store().load_active_manifest()
        if not manifest:
            raise RuntimeError("No active S3 document manifest is available.")
        docs = manifest.get("documents", [])
        if not isinstance(docs, list):
            raise RuntimeError("Active S3 document manifest has invalid documents.")

        records: list[object] = []
        for item in docs:
            if not isinstance(item, dict):
                continue
            source_doc_id = str(item.get("source_doc_id", ""))
            key = str(item.get("key", ""))
            source = str(item.get("source", ""))
            if not source_doc_id or not key or not source:
                continue
            records.append(
                S3DocumentRecord(
                    source_doc_id=source_doc_id,
                    key=key,
                    source=source,
                    version_id=str(item.get("version_id", "")),
                    etag=str(item.get("etag", "")),
                    size=int(item.get("size", 0) or 0),
                    last_modified=str(item.get("last_modified", "")),
                    content_hash=str(item.get("content_hash", "")),
                )
            )
        return records

    def _s3_docs_signature(
        self,
        manifest: dict[str, object] | None = None,
    ) -> tuple[tuple[str, int, int], ...]:
        manifest = manifest or self._s3_document_store().load_active_manifest() or {}
        records = self._s3_active_records(manifest)
        manifest_identity = "|".join(
            str(manifest.get(key, ""))
            for key in (
                "index_version",
                "chunking_version",
                "chunk_tokenizer_model",
                "chunk_tokenizer_trust_remote_code",
                "chunk_body_target_tokens",
                "chunk_body_max_tokens",
                "chunk_overlap_target_tokens",
                "chunk_overlap_max_tokens",
            )
        )
        manifest_digest = int(
            hashlib.sha256(manifest_identity.encode("utf-8")).hexdigest()[:12],
            16,
        )
        signature: list[tuple[str, int, int]] = [
            (
                f"s3-manifest:{manifest.get('index_version', '')}",
                len(records),
                manifest_digest,
            )
        ]
        for record in records:
            version = (
                getattr(record, "version_id", "")
                or getattr(record, "etag", "")
                or getattr(record, "content_hash", "")
            )
            identity = "|".join(
                (
                    str(getattr(record, "source_doc_id", "")),
                    str(getattr(record, "key", "")),
                    str(version),
                    str(getattr(record, "content_hash", "")),
                )
            )
            digest = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12], 16)
            signature.append(
                (
                    f"s3:{getattr(record, 'key', '')}",
                    int(getattr(record, "size", 0) or 0),
                    digest,
                )
            )
        return tuple(signature)

    def _active_collection_name(self, collection_or_alias: str) -> str:
        try:
            aliases = self.client.get_aliases().aliases
        except Exception:
            return collection_or_alias

        for alias in aliases:
            if alias.alias_name == collection_or_alias:
                return alias.collection_name
        return collection_or_alias

    @staticmethod
    def _bm25_payload_text(payload: dict[str, object]) -> str:
        parts = []
        headings = payload.get("headings") or ()
        if isinstance(headings, Sequence) and not isinstance(headings, (str, bytes)):
            heading_text = " > ".join(str(heading) for heading in headings if heading)
            if heading_text:
                parts.append(f"Headings: {heading_text}")
        content_type = str(payload.get("content_type", "text"))
        parts.append(f"Content type: {content_type}")
        parts.append(str(payload.get("text", "")))
        return "\n\n".join(part for part in parts if part).strip()

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
