from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_QUESTIONS = (
    "巨大历史机遇和巨大历史鲫鱼是什么梗？",
    "这份知识库里餐厅的主要业务风险是什么？",
    "请用中文总结巨大历史机遇相关材料。",
    "What does the knowledge base say about the restaurant case?",
)

METRICS = (
    ("end_to_end_ms", "HTTP end-to-end"),
    ("total_ms", "server pipeline"),
    ("history_ms", "history"),
    ("intent_ms", "intent router"),
    ("retrieval_ms", "retrieval total"),
    ("recall_ms", "recall total"),
    ("bm25_ms", "BM25 recall"),
    ("embedding_ms", "query embedding"),
    ("qdrant_ms", "Qdrant vector DB"),
    ("vector_ms", "vector recall total"),
    ("rrf_ms", "RRF fusion"),
    ("reranker_ms", "reranker"),
    ("llm_ms", "LLM total"),
    ("llm_estimated_tps", "LLM estimated TPS"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a localhost HTTP latency benchmark against /rag using one "
            "new chat session per request."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--question", default="")
    parser.add_argument(
        "--rag-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force RAG for all benchmark requests. Enabled by default.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-top-k", type=int, default=200)
    parser.add_argument("--recall-top-k", type=int, default=200)
    parser.add_argument("--rrf-top-k", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--output",
        default="/tmp/kba_latency_bench.json",
        help="JSON output path for per-request records and aggregate summary.",
    )
    return parser.parse_args()


async def login(client: httpx.AsyncClient, username: str, password: str) -> str:
    response = await client.post(
        "/auth/login",
        json={"username": username, "password": password},
    )
    response.raise_for_status()
    return str(response.json()["token"])


async def run_request(
    index: int,
    client: httpx.AsyncClient,
    token: str,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = args.question or DEFAULT_QUESTIONS[index % len(DEFAULT_QUESTIONS)]
    payload = {
        "question": question,
        "top_k": args.top_k,
        "bm25_top_k": args.bm25_top_k,
        "recall_top_k": args.recall_top_k,
        "rrf_top_k": args.rrf_top_k,
        "rag_only": args.rag_only,
    }
    headers = {"Authorization": f"Bearer {token}"}

    async with semaphore:
        started_at = time.perf_counter()
        response: httpx.Response | None = None
        try:
            response = await client.post("/rag", json=payload, headers=headers)
            end_to_end_ms = (time.perf_counter() - started_at) * 1000
            response.raise_for_status()
            data = response.json()
            timings = data.get("timings") or {}
            return {
                "index": index,
                "ok": True,
                "status_code": response.status_code,
                "end_to_end_ms": end_to_end_ms,
                "session_id": data.get("session_id"),
                "route": data.get("route", ""),
                "used_rag": data.get("used_rag", False),
                "context_count": len(data.get("contexts") or []),
                "retrieval_degraded": data.get("retrieval_degraded", False),
                "qdrant_degraded": data.get("qdrant_degraded", False),
                "reranker_degraded": data.get("reranker_degraded", False),
                "degradation_reason": data.get("degradation_reason", ""),
                "answer_chars": len(data.get("answer") or ""),
                "timings": timings,
            }
        except Exception as exc:
            return {
                "index": index,
                "ok": False,
                "status_code": response.status_code if response else None,
                "end_to_end_ms": (time.perf_counter() - started_at) * 1000,
                "error": repr(exc),
                "response_text": response.text[:1000] if response else "",
                "timings": {},
            }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "avg": statistics.fmean(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def metric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if key == "end_to_end_ms":
            value = record.get("end_to_end_ms")
        else:
            if not record.get("ok"):
                continue
            value = (record.get("timings") or {}).get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("ok")]
    errors = [record for record in records if not record.get("ok")]
    metric_summary = {
        key: summarize(metric_values(records, key))
        for key, _label in METRICS
    }
    degraded = sum(
        1 for record in ok_records if record.get("retrieval_degraded")
    )
    return {
        "request_count": len(records),
        "ok_count": len(ok_records),
        "error_count": len(errors),
        "degraded_count": degraded,
        "metrics": metric_summary,
        "ttft_note": (
            "llm_ttft_ms is null because /rag is currently non-streaming; "
            "measure TTFT after adding a streaming endpoint or provider timing."
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "Requests: "
        f"ok={summary['ok_count']} "
        f"errors={summary['error_count']} "
        f"degraded={summary['degraded_count']} "
        f"total={summary['request_count']}"
    )
    print(summary["ttft_note"])
    print()
    print(
        f"{'Metric':<22} {'count':>5} {'avg':>10} {'p50':>10} "
        f"{'p95':>10} {'p99':>10} {'min':>10} {'max':>10}"
    )
    for key, label in METRICS:
        stats = summary["metrics"].get(key) or {}
        if not stats.get("count"):
            continue
        print(
            f"{label:<22} {stats['count']:>5} "
            f"{stats['avg']:>10.1f} {stats['p50']:>10.1f} "
            f"{stats['p95']:>10.1f} {stats['p99']:>10.1f} "
            f"{stats['min']:>10.1f} {stats['max']:>10.1f}"
        )


def benchmark_output(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    final: bool,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "sessions": args.sessions,
        "concurrency": args.concurrency,
        "rag_only": args.rag_only,
        "top_k": args.top_k,
        "bm25_top_k": args.bm25_top_k,
        "recall_top_k": args.recall_top_k,
        "rrf_top_k": args.rrf_top_k,
        "final": final,
        "summary": build_summary(records),
        "records": records,
    }


def write_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main_async() -> int:
    args = parse_args()
    if args.sessions <= 0:
        raise SystemExit("--sessions must be greater than 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be greater than 0")

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=max(args.concurrency + 4, 10),
        max_keepalive_connections=max(args.concurrency, 10),
    )
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        token = await login(client, args.username, args.password)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            run_request(index, client, token, semaphore, args)
            for index in range(args.sessions)
        ]
        records: list[dict[str, Any]] = []
        output_path = Path(args.output)
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            records.append(await task)
            if args.progress_every > 0 and completed % args.progress_every == 0:
                summary = build_summary(records)
                print(
                    f"progress {completed}/{args.sessions}: "
                    f"ok={summary['ok_count']} "
                    f"errors={summary['error_count']} "
                    f"degraded={summary['degraded_count']}",
                    flush=True,
                )
                write_output(
                    output_path,
                    benchmark_output(args, records, final=False),
                )

    output_path = Path(args.output)
    output = benchmark_output(args, records, final=True)
    summary = output["summary"]
    write_output(output_path, output)
    print_summary(summary)
    print()
    print(f"Wrote {output_path}")
    return 1 if summary["error_count"] else 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
