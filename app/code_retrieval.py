from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from qdrant_client.http.exceptions import UnexpectedResponse

from app.code_indexer import CodeEmbedder, CodeRepository, discover_code_repositories
from app.config import Settings, settings


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_QUERY_SYNONYMS = {
    "安装": ["install", "setup", "build", "package", "pip", "cmake"],
    "构建": ["build", "cmake", "compile", "make"],
    "编译": ["build", "compile", "cmake", "make"],
    "训练": ["train", "fit", "learner", "booster"],
    "预测": ["predict", "inplace_predict", "prediction"],
    "数据": ["data", "dmatrix", "matrix", "adapter"],
    "处理": ["process", "handle", "adapter", "batch"],
    "加载": ["load", "read", "deserialize"],
    "保存": ["save", "write", "serialize"],
    "参数": ["param", "config", "configure"],
    "模型": ["model", "booster", "learner"],
    "调用": ["call", "invoke"],
    "分布式": ["distributed", "rabit", "collective", "allreduce"],
    "显卡": ["gpu", "cuda", "device"],
}


@dataclass(frozen=True)
class CodeFileHit:
    file_id: UUID
    repository_id: str
    repository_name: str
    source_root: str
    path: str
    language: str
    score: float


@dataclass(frozen=True)
class CodeSearchHit:
    function_id: UUID
    file_id: UUID
    repository_id: str
    repository_name: str
    source_root: str
    path: str
    language: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str
    snippet: str
    start_line: int
    end_line: int
    score: float
    vector_score: float
    file_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class CodeSearchOutcome:
    query: str
    files: list[CodeFileHit]
    functions: list[CodeSearchHit]


class CodeIndexUnavailable(RuntimeError):
    pass


class CodeRetrieval:
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

    def search(
        self,
        query: str,
        *,
        file_top_k: int | None = None,
        function_top_k: int | None = None,
        final_top_k: int | None = None,
        repository_ids: list[str] | None = None,
    ) -> CodeSearchOutcome:
        selected_repository_ids = self._selected_repository_ids(repository_ids)
        self._ensure_index_available()

        file_limit = max(1, file_top_k or self.config.code_search_file_top_k)
        function_limit = max(
            1,
            function_top_k or self.config.code_search_function_top_k,
        )
        final_limit = max(1, final_top_k or self.config.code_search_final_top_k)

        query_vector = self.embedder.embed([query])[0]
        file_hits = self._search_files(
            query_vector,
            file_limit,
            selected_repository_ids,
        )
        if not file_hits:
            return CodeSearchOutcome(query=query, files=[], functions=[])

        file_scores = {str(hit.file_id): hit.score for hit in file_hits}
        function_hits = self._search_functions(
            query_vector,
            list(file_scores),
            function_limit,
        )
        repository_function_hits = self._search_functions_by_repositories(
            query_vector,
            selected_repository_ids,
            max(function_limit, final_limit * 10),
        )
        text_function_hits = self._search_functions_by_text(
            query,
            selected_repository_ids,
            max(function_limit, final_limit * 10),
        )
        function_hits = _merge_function_hits(
            function_hits,
            repository_function_hits,
            text_function_hits,
        )
        hydrated = self._hydrate_functions(function_hits, file_scores)
        ranked = _rank_code_hits(query, hydrated)
        return CodeSearchOutcome(
            query=query,
            files=file_hits,
            functions=ranked[:final_limit],
        )

    def repositories(self) -> list[tuple[CodeRepository, int]]:
        counts = self.repository_file_counts()
        return [
            (repository, counts.get(repository.id, 0))
            for repository in discover_code_repositories(self.config)
        ]

    def _selected_repository_ids(self, repository_ids: list[str] | None) -> list[str]:
        selected = _clean_repository_ids(repository_ids)
        if selected:
            return selected
        return [
            repository.id
            for repository in discover_code_repositories(self.config)
        ]

    def repository_file_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT repository_id, COUNT(*) AS file_count
                FROM code_files
                GROUP BY repository_id
                """
            ).fetchall()
        return {
            str(row["repository_id"]): int(row["file_count"])
            for row in rows
        }

    def _ensure_index_available(self) -> None:
        try:
            collection_names = {
                collection.name
                for collection in self.client.get_collections().collections
            }
        except Exception:
            raise

        missing = [
            collection
            for collection in (
                self.config.code_files_collection,
                self.config.code_functions_collection,
            )
            if collection not in collection_names
        ]
        if missing:
            joined = ", ".join(missing)
            raise CodeIndexUnavailable(
                "Code search index is not ready. Missing Qdrant collection(s): "
                f"{joined}. Open Code mode, select the repository, and run Index."
            )

    def _search_files(
        self,
        query_vector: list[float],
        limit: int,
        repository_ids: list[str],
    ) -> list[CodeFileHit]:
        hits = self._query_points(
            self.config.code_files_collection,
            query_vector,
            limit,
            query_filter=_repository_filter(repository_ids),
        )
        files: list[CodeFileHit] = []
        for hit in hits:
            payload = hit.payload or {}
            file_id = payload.get("file_id")
            if not file_id:
                continue
            files.append(
                CodeFileHit(
                    file_id=UUID(str(file_id)),
                    repository_id=str(payload.get("repository_id", "")),
                    repository_name=str(payload.get("repository_name", "")),
                    source_root=str(payload.get("source_root", "")),
                    path=str(payload.get("path", "")),
                    language=str(payload.get("language", "")),
                    score=float(hit.score),
                )
            )
        return files

    def _search_functions(
        self,
        query_vector: list[float],
        file_ids: list[str],
        limit: int,
    ) -> list[tuple[UUID, float]]:
        if not file_ids:
            return []
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="file_id",
                    match=MatchAny(any=file_ids),
                )
            ]
        )
        hits = self._query_points(
            self.config.code_functions_collection,
            query_vector,
            limit,
            query_filter=query_filter,
        )
        function_hits: list[tuple[UUID, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            function_id = payload.get("function_id")
            if function_id:
                function_hits.append((UUID(str(function_id)), float(hit.score)))
        return function_hits

    def _search_functions_by_repositories(
        self,
        query_vector: list[float],
        repository_ids: list[str],
        limit: int,
    ) -> list[tuple[UUID, float]]:
        hits = self._query_points(
            self.config.code_functions_collection,
            query_vector,
            limit,
            query_filter=_repository_filter(repository_ids),
        )
        function_hits: list[tuple[UUID, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            function_id = payload.get("function_id")
            if function_id:
                function_hits.append((UUID(str(function_id)), float(hit.score)))
        return function_hits

    def _search_functions_by_text(
        self,
        query: str,
        repository_ids: list[str],
        limit: int,
    ) -> list[tuple[UUID, float]]:
        tokens = _query_tokens(query)
        if not tokens:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id,
                       f.qualified_name,
                       f.signature,
                       f.docstring,
                       cf.path
                FROM code_functions f
                JOIN code_files cf ON cf.id = f.file_id
                WHERE cf.repository_id = ANY(%s)
                """,
                (repository_ids,),
            ).fetchall()

        scored: list[tuple[UUID, float]] = []
        for row in rows:
            score = _text_match_score(
                qualified_name=str(row["qualified_name"] or ""),
                signature=str(row["signature"] or ""),
                docstring=str(row["docstring"] or ""),
                path=str(row["path"] or ""),
                tokens=tokens,
            )
            if score > 0:
                scored.append((row["id"], 0.75 + score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def _query_points(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        *,
        query_filter: Filter | None = None,
    ) -> list[object]:
        if hasattr(self.client, "query_points"):
            kwargs: dict[str, object] = {
                "collection_name": collection_name,
                "query": query_vector,
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            try:
                return list(self.client.query_points(**kwargs).points)
            except UnexpectedResponse as exc:
                if _is_missing_collection_error(exc):
                    raise CodeIndexUnavailable(
                        "Code search index is not ready. Missing Qdrant collection "
                        f"`{collection_name}`. Open Code mode, select the repository, "
                        "and run Index."
                    ) from exc
                raise

        kwargs = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        try:
            return list(self.client.search(**kwargs))
        except UnexpectedResponse as exc:
            if _is_missing_collection_error(exc):
                raise CodeIndexUnavailable(
                    "Code search index is not ready. Missing Qdrant collection "
                    f"`{collection_name}`. Open Code mode, select the repository, "
                    "and run Index."
                ) from exc
            raise

    def _hydrate_functions(
        self,
        hits: list[tuple[UUID, float]],
        file_scores: dict[str, float],
    ) -> list[CodeSearchHit]:
        if not hits:
            return []

        ids = [function_id for function_id, _ in hits]
        score_by_id = {function_id: score for function_id, score in hits}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id,
                       f.file_id,
                       cf.repository_id,
                       cf.repository_name,
                       cf.source_root,
                       cf.path,
                       cf.language,
                       f.name,
                       f.qualified_name,
                       f.kind,
                       f.signature,
                       f.body,
                       f.docstring,
                       f.start_line,
                       f.end_line
                FROM code_functions f
                JOIN code_files cf ON cf.id = f.file_id
                WHERE f.id = ANY(%s)
                """,
                (ids,),
            ).fetchall()

        by_id = {row["id"]: row for row in rows}
        results: list[CodeSearchHit] = []
        for function_id in ids:
            row = by_id.get(function_id)
            if row is None:
                continue
            vector_score = float(score_by_id[function_id])
            file_id = row["file_id"]
            results.append(
                CodeSearchHit(
                    function_id=row["id"],
                    file_id=file_id,
                    repository_id=row["repository_id"],
                    repository_name=row["repository_name"],
                    source_root=row["source_root"],
                    path=row["path"],
                    language=row["language"],
                    name=row["name"],
                    qualified_name=row["qualified_name"],
                    kind=row["kind"],
                    signature=row["signature"],
                    docstring=row["docstring"] or "",
                    snippet=_limit_text(
                        row["body"],
                        self.config.code_payload_snippet_chars,
                    ),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    score=vector_score,
                    vector_score=vector_score,
                    file_score=file_scores.get(str(file_id)),
                )
            )
        return results

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self.config.database_url,
            row_factory=dict_row,
            connect_timeout=self.config.database_connect_timeout_seconds,
        )


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _clean_repository_ids(repository_ids: list[str] | None) -> list[str]:
    if not repository_ids:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for repository_id in repository_ids:
        value = repository_id.strip()
        if value and value not in seen:
            cleaned.append(value)
            seen.add(value)
    return cleaned


def _repository_filter(repository_ids: list[str]) -> Filter | None:
    if not repository_ids:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="repository_id",
                match=MatchAny(any=repository_ids),
            )
        ]
    )


def _merge_function_hits(
    *groups: list[tuple[UUID, float]],
) -> list[tuple[UUID, float]]:
    scores: dict[UUID, float] = {}
    order: list[UUID] = []
    for group in groups:
        for function_id, score in group:
            if function_id not in scores:
                order.append(function_id)
                scores[function_id] = score
            else:
                scores[function_id] = max(scores[function_id], score)
    return sorted(
        ((function_id, scores[function_id]) for function_id in order),
        key=lambda item: item[1],
        reverse=True,
    )


def _rank_code_hits(query: str, hits: list[CodeSearchHit]) -> list[CodeSearchHit]:
    tokens = _query_tokens(query)
    ranked = [
        replace(hit, score=hit.vector_score + _code_lexical_boost(hit, tokens))
        for hit in hits
    ]
    return sorted(
        ranked,
        key=lambda hit: (hit.score, hit.vector_score, -_path_penalty_rank(hit.path)),
        reverse=True,
    )


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        normalized = token.lower().strip()
        if len(normalized) < 2 or normalized in seen:
            return
        tokens.append(normalized)
        seen.add(normalized)

    for token in _TOKEN_RE.findall(query.lower()):
        add(token)
        for part in re.findall(r"[a-z]+|\d+", token.replace("_", " ")):
            add(part)
    for needle, replacements in _QUERY_SYNONYMS.items():
        if needle in query:
            for replacement in replacements:
                add(replacement)
    return tokens


def _code_lexical_boost(hit: CodeSearchHit, tokens: list[str]) -> float:
    if not tokens:
        return -_path_penalty(hit.path)

    name = hit.qualified_name.lower()
    signature = hit.signature.lower()
    path = hit.path.lower()
    docstring = hit.docstring.lower()

    boost = 0.0
    for token in tokens:
        if token in name:
            boost += 0.09
        if token in signature:
            boost += 0.04
        if token in path:
            boost += 0.03
        if docstring and token in docstring:
            boost += 0.02

    if "/src/" in path or "/include/" in path or "/python-package/xgboost/" in path:
        boost += 0.025
    return boost - _path_penalty(hit.path)


def _text_match_score(
    *,
    qualified_name: str,
    signature: str,
    docstring: str,
    path: str,
    tokens: list[str],
) -> float:
    haystacks = {
        "name": qualified_name.lower(),
        "signature": signature.lower(),
        "docstring": docstring.lower(),
        "path": path.lower(),
    }
    score = 0.0
    for token in tokens:
        if token in haystacks["name"]:
            score += 0.16
        if token in haystacks["signature"]:
            score += 0.08
        if token in haystacks["path"]:
            score += 0.06
        if token in haystacks["docstring"]:
            score += 0.04
    if score:
        if "/src/" in haystacks["path"] or "/include/" in haystacks["path"]:
            score += 0.04
        if "/python-package/xgboost/" in haystacks["path"]:
            score += 0.03
    return max(0.0, score - _path_penalty(path))


def _path_penalty(path: str) -> float:
    normalized = f"/{path.lower()}"
    if "/tests/" in normalized or "/test/" in normalized or "/test_" in normalized:
        return 0.12
    return 0.0


def _path_penalty_rank(path: str) -> int:
    return 1 if _path_penalty(path) else 0


def _is_missing_collection_error(exc: UnexpectedResponse) -> bool:
    text = str(exc).lower()
    return "not found" in text and "collection" in text
