#!/usr/bin/env python3
"""Serial XGBoost code-search smoke test: 20 questions via /code/search."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import UUID

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
USERNAME = sys.argv[2] if len(sys.argv) > 2 else "admin"
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else "123456"

RESULTS_FILE = Path("xgboost_20_results_v3.json")

QUESTIONS: list[str] = [
    "What does xgboost.train do and what are its main parameters?",
    "How does XGBClassifier.fit work internally? Which C++ function does it call?",
    "What is DMatrix and how is it constructed from a numpy array?",
    "Where is the binary:logistic objective implemented in XGBoost?",
    "What does Booster.predict do and how does it handle different output types?",
    "How does XGBoost handle missing values in DMatrix?",
    "What is the role of Learner::UpdateOneIter in the C++ core?",
    "How does XGBoost implement GPU training? Look for CUDA-related code.",
    "What is XGBoosterCreate and where is it defined?",
    "How does the rank:ndcg objective work?",
    "What is the purpose of the GradientBooster class?",
    "How does XGBoost serialize and load models? Find save_model and load_model.",
    "What does XGBoosterEvalOneIter do?",
    "How is the eval_metric parameter processed in XGBoost?",
    "Find the implementation of the AFT (Accelerated Failure Time) objective.",
    "What is QuantileSketch used for in XGBoost?",
    "How does distributed training with rabit work in XGBoost?",
    "What does XGBoosterDumpModel return?",
    "Find the implementation of XGBoosterSetParam.",
    "How does XGBoost handle early stopping? Find early_stopping_rounds logic.",
]


def _request(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode()
        try:
            return json.loads(body_text)
        except json.JSONDecodeError:
            return {"error": body_text, "status_code": exc.code}


def login() -> str:
    resp = _request("POST", "/auth/login", {"username": USERNAME, "password": PASSWORD})
    if "token" not in resp:
        raise RuntimeError(f"Login failed: {resp}")
    return str(resp["token"])


def create_session(token: str) -> UUID:
    resp = _request("POST", "/sessions", headers={"Authorization": f"Bearer {token}"})
    return UUID(resp["id"])


def code_search(token: str, session_id: UUID, query: str) -> dict:
    payload = {
        "query": query,
        "session_id": str(session_id),
        "repository_ids": ["xgboost"],
        "final_top_k": 5,
    }
    return _request("POST", "/code/search", payload, {"Authorization": f"Bearer {token}"})


def main() -> None:
    print(f"Base URL: {BASE_URL}")
    token = login()
    print(f"Logged in as {USERNAME}")
    session_id = create_session(token)
    print(f"Session: {session_id}")

    results: list[dict] = []
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{index}/20] {question}")
        start = time.time()
        try:
            resp = code_search(token, session_id, question)
        except Exception as exc:
            resp = {"error": str(exc)}
        elapsed = time.time() - start
        summary = {
            "index": index,
            "question": question,
            "elapsed_seconds": round(elapsed, 2),
            "status_code": 200 if "answer" in resp else 500,
            "answer": resp.get("answer", ""),
            "function_count": len(resp.get("functions", [])),
            "file_count": len(resp.get("files", [])),
            "graph_node_count": len(resp.get("graph", {}).get("nodes", [])),
            "graph_edge_count": len(resp.get("graph", {}).get("edges", [])),
            "functions": [
                {
                    "qualified_name": f.get("qualified_name"),
                    "path": f.get("path"),
                    "score": f.get("score"),
                }
                for f in resp.get("functions", [])[:5]
            ],
            "files": [
                {"path": f.get("path"), "score": f.get("score")}
                for f in resp.get("files", [])[:5]
            ],
            "graph": {
                "nodes": [
                    {"id": n.get("id"), "label": n.get("label")}
                    for n in resp.get("graph", {}).get("nodes", [])[:10]
                ],
                "edges": [
                    {"source": e.get("source"), "target": e.get("target")}
                    for e in resp.get("graph", {}).get("edges", [])[:10]
                ],
            },
            "error": resp.get("error", ""),
        }
        results.append(summary)
        print(f"  -> {summary['function_count']} functions, {summary['file_count']} files, "
              f"graph {summary['graph_node_count']} nodes/{summary['graph_edge_count']} edges, "
              f"{summary['elapsed_seconds']}s")
        if summary["error"]:
            print(f"  -> ERROR: {summary['error']}")
        else:
            print(f"  -> answer preview: {summary['answer'][:120].replace(chr(10), ' ')}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
