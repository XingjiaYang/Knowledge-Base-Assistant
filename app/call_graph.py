from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import networkx as nx
import psycopg
from psycopg.rows import dict_row

from app.code_indexer import discover_code_repositories
from app.config import Settings, settings


@dataclass(frozen=True)
class CallEdge:
    caller_id: UUID
    caller_name: str
    callee_name: str
    file_id: UUID
    call_line: int


class CallGraphStore:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def get_callers(
        self,
        function_name: str,
        repository_ids: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        graph = self._load_graph(repository_ids)
        targets = self._matching_nodes(graph, function_name)
        callers: dict[str, dict[str, object]] = {}
        for target in targets:
            for caller in graph.predecessors(target):
                callers[caller] = self._node_payload(graph, caller)
        return sorted(callers.values(), key=lambda item: str(item["id"]))

    def get_call_chain(
        self,
        function_name: str,
        depth: int | None = None,
        repository_ids: Sequence[str] | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        graph = self._load_graph(repository_ids)
        max_depth = max(1, depth or self.config.code_call_graph_depth)
        roots = self._matching_nodes(graph, function_name)
        if not roots:
            return {"nodes": [], "edges": []}

        visited: set[str] = set(roots)
        queue: deque[tuple[str, int]] = deque((root, 0) for root in roots)
        edges: dict[tuple[str, str], dict[str, object]] = {}

        while queue:
            node, node_depth = queue.popleft()
            if node_depth >= max_depth:
                continue
            for callee in graph.successors(node):
                edge_data = graph.get_edge_data(node, callee) or {}
                edges[(node, callee)] = self._edge_payload(node, callee, edge_data)
                if callee not in visited:
                    visited.add(callee)
                    queue.append((callee, node_depth + 1))

        return {
            "nodes": [
                self._node_payload(graph, node)
                for node in sorted(visited)
            ],
            "edges": [
                edges[key]
                for key in sorted(edges)
            ],
        }

    def _load_graph(self, repository_ids: Sequence[str] | None = None) -> nx.DiGraph:
        graph = nx.DiGraph()
        selected_repository_ids = self._selected_repository_ids(repository_ids)
        repository_clause = ""
        params: tuple[object, ...] = ()
        if selected_repository_ids:
            repository_clause = "WHERE cf.repository_id = ANY(%s)"
            params = (selected_repository_ids,)
        with self._connect() as conn:
            functions = conn.execute(
                f"""
                SELECT f.id,
                       f.name,
                       f.qualified_name,
                       f.kind,
                       f.start_line,
                       f.end_line,
                       cf.repository_id,
                       cf.repository_name,
                       cf.path
                FROM code_functions f
                JOIN code_files cf ON cf.id = f.file_id
                {repository_clause}
                """,
                params,
            ).fetchall()
            edges = conn.execute(
                f"""
                SELECT e.caller_name,
                       e.callee_name,
                       e.call_line,
                       cf.repository_id,
                       cf.repository_name,
                       cf.path
                FROM code_call_edges e
                JOIN code_files cf ON cf.id = e.file_id
                {repository_clause}
                """,
                params,
            ).fetchall()

        for row in functions:
            node_id = _node_id(row["repository_id"], row["qualified_name"])
            graph.add_node(
                node_id,
                id=str(row["id"]),
                label=row["qualified_name"],
                name=row["name"],
                kind=row["kind"],
                repository_id=row["repository_id"],
                repository_name=row["repository_name"],
                path=row["path"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                indexed=True,
            )

        function_index = _build_function_index(graph)
        for row in edges:
            repository_id = row["repository_id"]
            caller = _node_id(repository_id, row["caller_name"])
            if caller not in graph:
                continue
            callee = _resolve_callee_node(
                function_index,
                repository_id=repository_id,
                caller_name=row["caller_name"],
                callee_name=row["callee_name"],
                path=row["path"],
            )
            if not callee:
                continue
            if graph.has_edge(caller, callee):
                data = graph[caller][callee]
                data.setdefault("lines", []).append(int(row["call_line"]))
            else:
                graph.add_edge(
                    caller,
                    callee,
                    path=row["path"],
                    lines=[int(row["call_line"])],
                )
        return graph

    def _selected_repository_ids(
        self,
        repository_ids: Sequence[str] | None,
    ) -> list[str]:
        selected = _clean_repository_ids(repository_ids)
        if selected:
            return selected
        return [
            repository.id
            for repository in discover_code_repositories(self.config)
        ]

    @staticmethod
    def _matching_nodes(graph: nx.DiGraph, function_name: str) -> list[str]:
        needle = function_name.strip()
        if not needle:
            return []
        if needle in graph:
            return [needle]

        suffixes = (f".{needle}", f"::{needle}")
        matches = [
            node
            for node, data in graph.nodes(data=True)
            if str(data.get("label", node)) == needle
            or data.get("name") == needle
            or str(data.get("label", node)).endswith(suffixes)
            or str(data.get("label", node)).split(".")[-1].split("::")[-1] == needle
        ]
        return sorted(set(matches))

    @staticmethod
    def _node_payload(graph: nx.DiGraph, node: str) -> dict[str, object]:
        data = graph.nodes[node]
        return {
            "id": node,
            "label": data.get("label", node),
            "name": data.get("name", node),
            "kind": data.get("kind", "external"),
            "path": data.get("path", ""),
            "repository_id": data.get("repository_id", ""),
            "repository_name": data.get("repository_name", ""),
            "start_line": data.get("start_line", 0),
            "end_line": data.get("end_line", 0),
            "indexed": bool(data.get("indexed", False)),
        }

    @staticmethod
    def _edge_payload(
        source: str,
        target: str,
        data: dict[str, object],
    ) -> dict[str, object]:
        lines = sorted({int(line) for line in data.get("lines", [])})
        return {
            "source": source,
            "target": target,
            "path": data.get("path", ""),
            "lines": lines,
        }

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self.config.database_url,
            row_factory=dict_row,
            connect_timeout=self.config.database_connect_timeout_seconds,
        )


def _node_id(repository_id: str, qualified_name: str) -> str:
    return f"{repository_id}:{qualified_name}" if repository_id else qualified_name


def _build_function_index(graph: nx.DiGraph) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for node, data in graph.nodes(data=True):
        if not data.get("indexed"):
            continue
        repository_id = str(data.get("repository_id", ""))
        label = str(data.get("label", node))
        name = str(data.get("name", "")) or _tail_name(label)
        path = str(data.get("path", ""))
        bucket = index.setdefault(
            repository_id,
            {
                "qualified": {},
                "name": {},
                "path_name": {},
                "suffix": {},
                "path_by_node": {},
            },
        )
        qualified = bucket["qualified"]
        name_index = bucket["name"]
        path_name = bucket["path_name"]
        suffix = bucket["suffix"]
        path_by_node = bucket["path_by_node"]

        qualified[label] = node
        name_index.setdefault(name, []).append(node)
        path_name.setdefault((path, name), []).append(node)
        path_by_node[node] = path
        for label_suffix in _qualified_suffixes(label):
            suffix.setdefault(label_suffix, []).append(node)
    return index


def _resolve_callee_node(
    index: dict[str, dict[str, object]],
    *,
    repository_id: str,
    caller_name: str,
    callee_name: str,
    path: str,
) -> str | None:
    bucket = index.get(repository_id)
    if not bucket:
        return None

    qualified = bucket["qualified"]
    suffix = bucket["suffix"]
    name_index = bucket["name"]
    path_name = bucket["path_name"]
    path_by_node = bucket["path_by_node"]

    for candidate in _call_name_candidates(callee_name, caller_name):
        if candidate in qualified:
            return str(qualified[candidate])
        resolved = _pick_node(suffix.get(candidate, []), path_by_node, path)
        if resolved:
            return resolved

        tail = _tail_name(candidate)
        resolved = _pick_node(path_name.get((path, tail), []), path_by_node, path)
        if resolved:
            return resolved
        resolved = _pick_node(name_index.get(tail, []), path_by_node, path)
        if resolved:
            return resolved
    return None


def _pick_node(
    nodes: Sequence[object],
    path_by_node: dict[object, str],
    path: str,
) -> str | None:
    if not nodes:
        return None
    same_path = [node for node in nodes if path_by_node.get(node) == path]
    if len(same_path) == 1:
        return str(same_path[0])
    if len(nodes) == 1:
        return str(nodes[0])
    return None


def _call_name_candidates(callee_name: str, caller_name: str) -> list[str]:
    raw = _strip_template_args(str(callee_name).strip())
    if not raw:
        return []

    normalized = raw.replace("->", ".")
    candidates = [normalized]

    python_scope = caller_name.rsplit(".", 1)[0] if "." in caller_name else ""
    cpp_scope = caller_name.rsplit("::", 1)[0] if "::" in caller_name else ""
    if python_scope and normalized.startswith("self."):
        candidates.append(f"{python_scope}.{normalized.removeprefix('self.')}")
    if cpp_scope and normalized.startswith("this."):
        candidates.append(f"{cpp_scope}::{normalized.removeprefix('this.')}")

    tail = _tail_name(normalized)
    if tail != normalized:
        candidates.append(tail)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _qualified_suffixes(qualified_name: str) -> list[str]:
    suffixes = {qualified_name}
    for separator in ("::", "."):
        parts = qualified_name.split(separator)
        if len(parts) > 1:
            for index in range(1, len(parts)):
                suffixes.add(separator.join(parts[index:]))
    suffixes.add(_tail_name(qualified_name))
    return sorted(suffixes)


def _tail_name(name: str) -> str:
    return name.split(".")[-1].split("::")[-1]


def _strip_template_args(text: str) -> str:
    output: list[str] = []
    depth = 0
    for char in text:
        if char == "<":
            depth += 1
            continue
        if char == ">" and depth:
            depth -= 1
            continue
        if depth == 0:
            output.append(char)
    return "".join(output).strip()


def _clean_repository_ids(repository_ids: Sequence[str] | None) -> list[str]:
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


def extract_call_edges(functions: Sequence[object]) -> list[CallEdge]:
    edges: list[CallEdge] = []
    seen: set[tuple[UUID, str, int]] = set()
    for function in functions:
        if getattr(function, "kind", "") == "class":
            continue
        for callee_name, call_line in _extract_function_calls(function):
            key = (function.id, callee_name, call_line)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                CallEdge(
                    caller_id=function.id,
                    caller_name=function.qualified_name,
                    callee_name=callee_name,
                    file_id=function.file_id,
                    call_line=call_line,
                )
            )
    return edges


def _extract_function_calls(function: object) -> list[tuple[str, int]]:
    language = getattr(function, "language", "")
    body = getattr(function, "body", "")
    if not body:
        return []

    parser = _get_parser(language)
    source = body.encode("utf-8", errors="replace")
    tree = parser.parse(source)
    calls: list[tuple[str, int]] = []

    def visit(node: object) -> None:
        node_type = getattr(node, "type", "")
        callee = ""
        if language == "python" and node_type == "call":
            callee = _python_call_name(node, source)
        elif language == "cpp" and node_type == "call_expression":
            callee = _cpp_call_name(node, source)

        if callee:
            call_line = int(getattr(node, "start_point")[0]) + int(
                getattr(function, "start_line", 1)
            )
            calls.append((callee, call_line))

        for child in getattr(node, "children", []) or []:
            visit(child)

    visit(tree.root_node)
    return calls


def _python_call_name(node: object, source: bytes) -> str:
    function_node = _child_by_field(node, "function")
    if function_node is None:
        return ""
    return _normalized_call_text(function_node, source)


def _cpp_call_name(node: object, source: bytes) -> str:
    function_node = _child_by_field(node, "function")
    if function_node is None:
        return ""
    return _normalized_call_text(function_node, source)


def _normalized_call_text(node: object, source: bytes) -> str:
    text = _node_text(node, source).strip()
    if not text:
        return ""
    return " ".join(text.split())


def _get_parser(language: str) -> object:
    try:
        from tree_sitter_language_pack import get_parser
    except Exception as exc:
        raise RuntimeError(
            "tree-sitter-language-pack is required for call graph extraction."
        ) from exc

    if language == "python":
        return get_parser("python")
    if language == "cpp":
        return get_parser("cpp")
    raise ValueError(f"Unsupported code language: {language}")


def _child_by_field(node: object | None, field_name: str) -> object | None:
    if node is None or not hasattr(node, "child_by_field_name"):
        return None
    return node.child_by_field_name(field_name)


def _node_text(node: object | None, source: bytes) -> str:
    if node is None:
        return ""
    start = int(getattr(node, "start_byte"))
    end = int(getattr(node, "end_byte"))
    return source[start:end].decode("utf-8", errors="replace")
