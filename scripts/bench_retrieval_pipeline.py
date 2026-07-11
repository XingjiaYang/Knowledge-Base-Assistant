from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
import sys
from threading import Event, Lock
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.model_actors import (  # noqa: E402
    get_embedding_actor,
    get_reranker_actors,
    ray_get,
)
from app.rag import RAGPipeline  # noqa: E402


DEFAULT_QUERIES = (
    "巨大历史机遇和巨大历史鲫鱼是什么梗？",
    "这份知识库里的餐厅有哪些主要业务风险？",
    "总结巨大历史机遇相关材料的时间线。",
    "What does the knowledge base say about the restaurant case?",
    "餐厅菜单、定价和客户评价之间有什么关系？",
    "朱剑秋和永哥直播事件的背景是什么？",
    "文档中对公司财务情况有哪些分析？",
    "How does the FAQ describe the business and its customers?",
)

TIMING_FIELDS = (
    "retrieval_ms",
    "recall_ms",
    "bm25_ms",
    "vector_ms",
    "embedding_ms",
    "qdrant_ms",
    "rrf_ms",
    "reranker_ms",
)


@dataclass(frozen=True)
class RetrievalRecord:
    index: int
    ok: bool
    service_ms: float
    queue_ms: float
    end_to_end_ms: float
    context_count: int = 0
    reranked_context_count: int = 0
    retrieval_degraded: bool = False
    embedding_degraded: bool = False
    qdrant_degraded: bool = False
    reranker_degraded: bool = False
    degradation_reason: str = ""
    timings: dict[str, float] | None = None
    top_source: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BM25 + embedding + Qdrant + RRF + reranker without "
            "intent routing, LLM generation, HTTP, or Session persistence."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fixed-rate", "closed-loop"),
        default="closed-loop",
    )
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-top-k", type=int, default=100)
    parser.add_argument("--recall-top-k", type=int, default=100)
    parser.add_argument("--rrf-top-k", type=int, default=64)
    parser.add_argument(
        "--fail-on-degradation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument(
        "--output",
        default="/tmp/kba_retrieval_benchmark.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "rate": args.rate,
        "concurrency": args.concurrency,
        "workers": args.workers,
        "duration": args.duration,
        "top-k": args.top_k,
        "bm25-top-k": args.bm25_top_k,
        "recall-top-k": args.recall_top_k,
        "rrf-top-k": args.rrf_top_k,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"--{name} must be greater than 0")
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be greater than or equal to 0")
    if args.max_requests < 0:
        raise SystemExit("--max-requests must be greater than or equal to 0")
    if args.mode == "closed-loop" and args.workers < args.concurrency:
        raise SystemExit("--workers must be at least --concurrency in closed-loop mode")


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


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "avg": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def linear_slope(samples: list[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    x_mean = statistics.fmean(x for x, _y in samples)
    y_mean = statistics.fmean(y for _x, y in samples)
    denominator = sum((x - x_mean) ** 2 for x, _y in samples)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in samples) / denominator


def execute_retrieval(
    pipeline: RAGPipeline,
    index: int,
    target_at: float,
    args: argparse.Namespace,
) -> RetrievalRecord:
    started_at = time.perf_counter()
    query = DEFAULT_QUERIES[index % len(DEFAULT_QUERIES)]
    try:
        outcome = pipeline._retrieve_contexts(  # noqa: SLF001 - benchmark target
            query,
            top_k=args.top_k,
            recall_top_k=args.recall_top_k,
            bm25_top_k=args.bm25_top_k,
            rrf_top_k=args.rrf_top_k,
        )
        finished_at = time.perf_counter()
        timings = {
            field: float(getattr(outcome.timings, field))
            for field in TIMING_FIELDS
        }
        contexts = outcome.contexts
        reranked_count = sum(
            context.rerank_score is not None for context in contexts
        )
        return RetrievalRecord(
            index=index,
            ok=bool(contexts) and reranked_count == len(contexts),
            service_ms=(finished_at - started_at) * 1000,
            queue_ms=max(0.0, (started_at - target_at) * 1000),
            end_to_end_ms=max(0.0, (finished_at - target_at) * 1000),
            context_count=len(contexts),
            reranked_context_count=reranked_count,
            retrieval_degraded=outcome.retrieval_degraded,
            embedding_degraded=outcome.embedding_degraded,
            qdrant_degraded=outcome.qdrant_degraded,
            reranker_degraded=outcome.reranker_degraded,
            degradation_reason=outcome.degradation_reason,
            timings=timings,
            top_source=contexts[0].source if contexts else "",
            error=(
                "Retrieval returned no contexts or contexts without rerank scores."
                if not contexts or reranked_count != len(contexts)
                else ""
            ),
        )
    except Exception as exc:
        finished_at = time.perf_counter()
        return RetrievalRecord(
            index=index,
            ok=False,
            service_ms=(finished_at - started_at) * 1000,
            queue_ms=max(0.0, (started_at - target_at) * 1000),
            end_to_end_ms=max(0.0, (finished_at - target_at) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


def run_fixed_rate(
    executor: ThreadPoolExecutor,
    pipeline: RAGPipeline,
    args: argparse.Namespace,
) -> tuple[list[RetrievalRecord], float, float]:
    request_count = max(1, int(args.rate * args.duration))
    if args.max_requests > 0:
        request_count = min(request_count, args.max_requests)
    phase_started = time.perf_counter()
    futures: list[Future[RetrievalRecord]] = []
    for index in range(request_count):
        target_at = phase_started + index / args.rate
        sleep_seconds = target_at - time.perf_counter()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        futures.append(
            executor.submit(execute_retrieval, pipeline, index, target_at, args)
        )
    records = [future.result() for future in as_completed(futures)]
    return sorted(records, key=lambda item: item.index), phase_started, time.perf_counter()


def run_closed_loop(
    executor: ThreadPoolExecutor,
    pipeline: RAGPipeline,
    args: argparse.Namespace,
) -> tuple[list[RetrievalRecord], float, float]:
    phase_started = time.perf_counter()
    deadline = phase_started + args.duration
    stop_event = Event()
    index_lock = Lock()
    next_index = 0

    def worker() -> list[RetrievalRecord]:
        nonlocal next_index
        worker_records: list[RetrievalRecord] = []
        while not stop_event.is_set() and time.perf_counter() < deadline:
            with index_lock:
                if args.max_requests > 0 and next_index >= args.max_requests:
                    stop_event.set()
                    break
                index = next_index
                next_index += 1
            target_at = time.perf_counter()
            record = execute_retrieval(pipeline, index, target_at, args)
            worker_records.append(record)
            if not record.ok:
                stop_event.set()
        return worker_records

    futures = [executor.submit(worker) for _ in range(args.concurrency)]
    records: list[RetrievalRecord] = []
    for future in as_completed(futures):
        records.extend(future.result())
    return sorted(records, key=lambda item: item.index), phase_started, time.perf_counter()


def probe_actors() -> dict[str, Any]:
    embedding_actor = get_embedding_actor(settings)
    if embedding_actor is None:
        raise RuntimeError("Ray Embedding actor is unavailable.")
    embedding_health = ray_get(embedding_actor.health.remote(), settings)
    reranker_health: dict[str, Any] = {}
    actors = get_reranker_actors(settings, retry_unavailable=True)
    if not actors:
        raise RuntimeError("Ray Reranker actors are unavailable.")
    for name, actor in actors:
        reranker_health[name] = ray_get(actor.health.remote(), settings)
    return {
        "embedding": embedding_health,
        "rerankers": reranker_health,
    }


def embedding_batch_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float | int | bool]:
    before_stats = before.get("embedding", {}).get("dynamic_batching", {})
    after_stats = after.get("embedding", {}).get("dynamic_batching", {})
    batch_count = max(
        0,
        int(after_stats.get("batch_count", 0))
        - int(before_stats.get("batch_count", 0)),
    )
    request_count = max(
        0,
        int(after_stats.get("request_count", 0))
        - int(before_stats.get("request_count", 0)),
    )
    return {
        "enabled": bool(after_stats.get("enabled", False)),
        "batch_count": batch_count,
        "request_count": request_count,
        "average_batch_size": request_count / batch_count if batch_count else 0.0,
        "configured_max_size": int(after_stats.get("max_size", 0)),
        "lifetime_max_observed_batch_size": int(
            after_stats.get("max_observed_batch_size", 0)
        ),
        "final_queue_depth": int(after_stats.get("queue_depth", 0)),
    }


def metric(records: list[RetrievalRecord], field: str) -> dict[str, float | int]:
    values: list[float] = []
    for record in records:
        if not record.ok:
            continue
        if field in {"service_ms", "queue_ms", "end_to_end_ms"}:
            values.append(float(getattr(record, field)))
        elif record.timings is not None:
            values.append(float(record.timings[field]))
    return summarize(values)


def print_metric(label: str, values: dict[str, float | int]) -> None:
    if not values.get("count"):
        return
    print(
        f"{label:<18} count={values['count']} avg={values['avg']:.3f}ms "
        f"p50={values['p50']:.3f}ms p95={values['p95']:.3f}ms "
        f"p99={values['p99']:.3f}ms max={values['max']:.3f}ms"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    validate_args(args)
    pipeline = RAGPipeline(settings)

    try:
        actor_health_before = probe_actors()
        warmup_records = [
            execute_retrieval(pipeline, -index - 1, time.perf_counter(), args)
            for index in range(args.warmup_requests)
        ]
        if any(
            not record.ok or record.retrieval_degraded
            for record in warmup_records
        ):
            raise RuntimeError("Retrieval warmup failed or degraded.")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            if args.mode == "fixed-rate":
                records, phase_started, phase_finished = run_fixed_rate(
                    executor,
                    pipeline,
                    args,
                )
            else:
                records, phase_started, phase_finished = run_closed_loop(
                    executor,
                    pipeline,
                    args,
                )

        elapsed_seconds = max(0.000001, phase_finished - phase_started)
        successful = [record for record in records if record.ok]
        errors = [record for record in records if not record.ok]
        degraded = [
            record
            for record in successful
            if record.retrieval_degraded
            or record.embedding_degraded
            or record.qdrant_degraded
            or record.reranker_degraded
        ]
        metrics = {
            field: metric(records, field)
            for field in (
                "service_ms",
                "queue_ms",
                "end_to_end_ms",
                *TIMING_FIELDS,
            )
        }
        top_sources: dict[str, int] = {}
        for record in successful:
            top_sources[record.top_source] = top_sources.get(record.top_source, 0) + 1
        queue_slope = 0.0
        service_slope = 0.0
        if args.mode == "fixed-rate":
            queue_slope = linear_slope(
                [
                    (record.index / args.rate, record.queue_ms)
                    for record in successful
                ]
            )
            service_slope = linear_slope(
                [
                    (record.index / args.rate, record.service_ms)
                    for record in successful
                ]
            )
        actor_health_after = probe_actors()
        batch_delta = embedding_batch_delta(
            actor_health_before,
            actor_health_after,
        )

        summary: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "mode": args.mode,
                "offered_qps": args.rate if args.mode == "fixed-rate" else None,
                "concurrency": args.concurrency,
                "workers": args.workers,
                "duration_seconds": args.duration,
                "warmup_requests": args.warmup_requests,
                "max_requests": args.max_requests,
                "top_k": args.top_k,
                "bm25_top_k": args.bm25_top_k,
                "recall_top_k": args.recall_top_k,
                "rrf_top_k": args.rrf_top_k,
                "query_count": len(DEFAULT_QUERIES),
            },
            "requests": {
                "total": len(records),
                "success": len(successful),
                "errors": len(errors),
                "degraded": len(degraded),
                "actual_completion_qps": len(successful) / elapsed_seconds,
                "phase_elapsed_seconds": elapsed_seconds,
                "queue_delay_growth_ms_per_second": queue_slope,
                "service_latency_growth_ms_per_second": service_slope,
                "metrics": metrics,
            },
            "actors": {
                "before": actor_health_before,
                "after": actor_health_after,
                "embedding_batch_delta": batch_delta,
            },
            "results": {
                "top_source_counts": top_sources,
                "all_contexts_reranked": all(
                    record.context_count == record.reranked_context_count
                    for record in successful
                ),
            },
            "errors": [asdict(record) for record in errors[:20]],
            "degradations": [asdict(record) for record in degraded[:20]],
        }
        if args.include_records:
            summary["records"] = [asdict(record) for record in records]

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("Retrieval pipeline benchmark")
        if args.mode == "fixed-rate":
            print(
                f"mode=fixed-rate offered={args.rate:.3f} QPS "
                f"duration={elapsed_seconds:.3f}s"
            )
        else:
            print(
                f"mode=closed-loop concurrency={args.concurrency} "
                f"duration={elapsed_seconds:.3f}s"
            )
        print(
            f"requests={len(records)} success={len(successful)} "
            f"errors={len(errors)} degraded={len(degraded)} "
            f"actual_completion_qps={summary['requests']['actual_completion_qps']:.3f}"
        )
        if args.mode == "fixed-rate":
            print(
                f"queue_slope={queue_slope:.3f}ms/s "
                f"service_slope={service_slope:.3f}ms/s"
            )
        for field in ("service_ms", "queue_ms", *TIMING_FIELDS):
            print_metric(field, metrics[field])
        print(f"actors={json.dumps(actor_health_after, ensure_ascii=False)}")
        print(f"embedding_batch_delta={json.dumps(batch_delta)}")
        print(f"output={output_path}")

        failed = bool(errors)
        if args.fail_on_degradation:
            failed = failed or bool(degraded)
        if failed:
            raise SystemExit(1)
    finally:
        pipeline.vector_store.close()


if __name__ == "__main__":
    main()
