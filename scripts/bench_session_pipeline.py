from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from threading import Event, Lock, Thread
import time
from typing import Any
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.redis_session_store import RedisSessionStore  # noqa: E402
from app.session_store import SessionStore  # noqa: E402
from app.vector_store import SearchResult  # noqa: E402


@dataclass(frozen=True)
class WriteRecord:
    index: int
    ok: bool
    service_ms: float
    queue_ms: float
    end_to_end_ms: float
    error: str = ""


class BacklogSampler:
    def __init__(
        self,
        store: RedisSessionStore,
        interval_seconds: float,
        *,
        abort_event: Event | None = None,
        max_backlog: int = 0,
    ) -> None:
        self.store = store
        self.interval_seconds = interval_seconds
        self.abort_event = abort_event
        self.max_backlog = max_backlog
        self.samples: list[tuple[float, int]] = []
        self.errors: list[str] = []
        self.abort_reason = ""
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4))

    def _run(self) -> None:
        started = time.perf_counter()
        while not self._stop.is_set():
            try:
                backlog = int(self.store.client.xlen(self.store.archive_stream_key))
                self.samples.append((time.perf_counter() - started, backlog))
                if (
                    self.max_backlog > 0
                    and backlog >= self.max_backlog
                    and self.abort_event is not None
                ):
                    self.abort_reason = (
                        f"stream backlog reached safety limit {self.max_backlog}"
                    )
                    self.abort_event.set()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Redis -> Stream -> Archiver -> PostgreSQL sessions."
    )
    parser.add_argument(
        "--mode",
        choices=("fixed-rate", "closed-loop"),
        default="fixed-rate",
    )
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--session-count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--user-chars", type=int, default=128)
    parser.add_argument("--assistant-chars", type=int, default=1024)
    parser.add_argument("--context-count", type=int, default=5)
    parser.add_argument("--context-chars", type=int, default=2000)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--drain-timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=100000,
        help="Safety cap per phase; 0 disables the cap.",
    )
    parser.add_argument(
        "--max-backlog",
        type=int,
        default=10000,
        help="Stop closed-loop writes when Stream XLEN reaches this value; 0 disables.",
    )
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--output", default="/tmp/kba_session_benchmark.json")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "rate": args.rate,
        "concurrency": args.concurrency,
        "duration": args.duration,
        "session-count": args.session_count,
        "workers": args.workers,
        "user-chars": args.user_chars,
        "assistant-chars": args.assistant_chars,
        "context-count": args.context_count,
        "context-chars": args.context_chars,
        "sample-interval": args.sample_interval,
        "drain-timeout": args.drain_timeout,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise SystemExit(f"--{name} must be greater than 0")
    if args.warmup_seconds < 0:
        raise SystemExit("--warmup-seconds must be greater than or equal to 0")
    if args.max_requests < 0:
        raise SystemExit("--max-requests must be greater than or equal to 0")
    if args.max_backlog < 0:
        raise SystemExit("--max-backlog must be greater than or equal to 0")


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


def linear_slope(samples: list[tuple[float, int]]) -> float:
    if len(samples) < 2:
        return 0.0
    x_mean = statistics.fmean(sample[0] for sample in samples)
    y_mean = statistics.fmean(sample[1] for sample in samples)
    denominator = sum((x - x_mean) ** 2 for x, _y in samples)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in samples) / denominator


def fixed_text(prefix: str, length: int) -> str:
    repeats = (length // max(1, len(prefix))) + 1
    return (prefix * repeats)[:length]


def make_contexts(count: int, chars: int) -> list[SearchResult]:
    return [
        SearchResult(
            text=fixed_text(f"context-{index}-", chars),
            source=f"benchmark-{index}.md",
            chunk_id=index,
            score=1.0 / (index + 1),
            rerank_score=0.9 - index * 0.01,
            vector_score=0.8 - index * 0.01,
            bm25_score=10.0 - index,
            rrf_score=0.03 - index * 0.001,
            retrieval_source="hybrid",
            h1="Session benchmark",
            h2="Synthetic context",
            headings=("Session benchmark", "Synthetic context"),
            start_line=index * 10 + 1,
            end_line=index * 10 + 10,
        )
        for index in range(count)
    ]


def find_user_id(pg_store: SessionStore) -> UUID:
    with pg_store._connect() as conn:  # noqa: SLF001 - operational benchmark
        row = conn.execute(
            "SELECT id FROM app_users ORDER BY created_at LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("At least one application user is required.")
    return UUID(str(row["id"]))


def message_count(pg_store: SessionStore, session_ids: list[UUID]) -> int:
    with pg_store._connect() as conn:  # noqa: SLF001 - operational benchmark
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE session_id = ANY(%s)",
            (session_ids,),
        ).fetchone()
    return int(row["count"])


def unique_event_count(pg_store: SessionStore, session_ids: list[UUID]) -> int:
    with pg_store._connect() as conn:  # noqa: SLF001 - operational benchmark
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT event_id) AS count
            FROM chat_messages
            WHERE session_id = ANY(%s) AND event_id IS NOT NULL
            """,
            (session_ids,),
        ).fetchone()
    return int(row["count"])


def pending_count(store: RedisSessionStore, session_ids: list[UUID]) -> int:
    pipe = store.client.pipeline(transaction=False)
    for session_id in session_ids:
        pipe.hlen(store._pending_key(session_id))  # noqa: SLF001
    return sum(int(value) for value in pipe.execute())


def write_exchange(
    store: RedisSessionStore,
    session_id: UUID,
    contexts: list[SearchResult],
    user_content: str,
    assistant_content: str,
    index: int,
    target_at: float,
) -> WriteRecord:
    started_at = time.perf_counter()
    try:
        store.append_exchange(
            session_id,
            f"{index}: {user_content}"[: len(user_content)],
            f"{index}: {assistant_content}"[: len(assistant_content)],
            contexts=contexts,
            used_rag=True,
            route="rag_only",
            route_reason="Session pipeline benchmark.",
            conversation_summary="",
        )
        finished_at = time.perf_counter()
        return WriteRecord(
            index=index,
            ok=True,
            service_ms=(finished_at - started_at) * 1000,
            queue_ms=max(0.0, (started_at - target_at) * 1000),
            end_to_end_ms=max(0.0, (finished_at - target_at) * 1000),
        )
    except Exception as exc:
        finished_at = time.perf_counter()
        return WriteRecord(
            index=index,
            ok=False,
            service_ms=(finished_at - started_at) * 1000,
            queue_ms=max(0.0, (started_at - target_at) * 1000),
            end_to_end_ms=max(0.0, (finished_at - target_at) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


def run_fixed_rate_phase(
    executor: ThreadPoolExecutor,
    store: RedisSessionStore,
    session_ids: list[UUID],
    contexts: list[SearchResult],
    user_content: str,
    assistant_content: str,
    *,
    rate: float,
    duration: float,
    start_index: int,
    max_requests: int,
    abort_event: Event,
) -> tuple[list[WriteRecord], float, float]:
    request_count = max(1, int(rate * duration))
    if max_requests > 0:
        request_count = min(request_count, max_requests)
    phase_started = time.perf_counter()
    futures: list[Future[WriteRecord]] = []
    for offset in range(request_count):
        if abort_event.is_set():
            break
        target_at = phase_started + offset / rate
        sleep_seconds = target_at - time.perf_counter()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        index = start_index + offset
        session_id = session_ids[index % len(session_ids)]
        futures.append(
            executor.submit(
                write_exchange,
                store,
                session_id,
                contexts,
                user_content,
                assistant_content,
                index,
                target_at,
            )
        )

    records = [future.result() for future in as_completed(futures)]
    phase_finished = max(
        phase_started,
        phase_started + max((record.end_to_end_ms for record in records), default=0) / 1000,
        time.perf_counter(),
    )
    return sorted(records, key=lambda item: item.index), phase_started, phase_finished


def run_closed_loop_phase(
    executor: ThreadPoolExecutor,
    store: RedisSessionStore,
    session_ids: list[UUID],
    contexts: list[SearchResult],
    user_content: str,
    assistant_content: str,
    *,
    concurrency: int,
    duration: float,
    start_index: int,
    max_requests: int,
    abort_event: Event,
) -> tuple[list[WriteRecord], float, float]:
    phase_started = time.perf_counter()
    deadline = phase_started + duration
    index_lock = Lock()
    next_index = start_index

    def worker() -> list[WriteRecord]:
        nonlocal next_index
        worker_records: list[WriteRecord] = []
        while not abort_event.is_set() and time.perf_counter() < deadline:
            with index_lock:
                completed = next_index - start_index
                if max_requests > 0 and completed >= max_requests:
                    break
                index = next_index
                next_index += 1
            session_id = session_ids[index % len(session_ids)]
            target_at = time.perf_counter()
            worker_records.append(
                write_exchange(
                    store,
                    session_id,
                    contexts,
                    user_content,
                    assistant_content,
                    index,
                    target_at,
                )
            )
        return worker_records

    futures = [executor.submit(worker) for _ in range(concurrency)]
    records: list[WriteRecord] = []
    for future in as_completed(futures):
        records.extend(future.result())
    phase_finished = time.perf_counter()
    return sorted(records, key=lambda item: item.index), phase_started, phase_finished


def wait_for_archive(
    store: RedisSessionStore,
    pg_store: SessionStore,
    session_ids: list[UUID],
    expected_messages: int | None,
    timeout_seconds: float,
) -> tuple[float, int, int, int]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    current_messages = 0
    current_pending = 0
    current_stream = 0
    while time.perf_counter() < deadline:
        current_messages = message_count(pg_store, session_ids)
        current_pending = pending_count(store, session_ids)
        current_stream = int(store.client.xlen(store.archive_stream_key))
        if (
            (expected_messages is None or current_messages == expected_messages)
            and current_pending == 0
            and current_stream == 0
        ):
            return (
                (time.perf_counter() - started) * 1000,
                current_messages,
                current_pending,
                current_stream,
            )
        time.sleep(0.05)
    raise TimeoutError(
        "Archive did not drain before timeout: "
        f"messages={current_messages}/{expected_messages or 'unknown'} "
        f"pending={current_pending} stream={current_stream}"
    )


def cleanup_benchmark_sessions(
    store: RedisSessionStore,
    pg_store: SessionStore,
    user_id: UUID,
    session_ids: list[UUID],
) -> None:
    if not session_ids:
        return

    session_id_strings = {str(session_id) for session_id in session_ids}
    stream_entries = store.client.xrange(store.archive_stream_key)
    pipe = store.client.pipeline(transaction=False)
    for stream_id, fields in stream_entries:
        if str(fields.get("session_id", "")) in session_id_strings:
            pipe.xack(
                store.archive_stream_key,
                store.config.redis_archive_group,
                stream_id,
            )
            pipe.xdel(store.archive_stream_key, stream_id)
    for session_id in session_ids:
        pipe.delete(
            store._meta_key(session_id),  # noqa: SLF001
            store._messages_key(session_id),  # noqa: SLF001
            store._pending_key(session_id),  # noqa: SLF001
            store._lock_key(session_id),  # noqa: SLF001
        )
    pipe.zrem(store._user_sessions_key(user_id), *session_id_strings)  # noqa: SLF001
    pipe.execute()

    with pg_store._connect() as conn:  # noqa: SLF001 - operational benchmark
        conn.execute(
            "DELETE FROM chat_sessions WHERE id = ANY(%s)",
            (session_ids,),
        )


def metric_summary(records: list[WriteRecord], field: str) -> dict[str, float | int]:
    return summarize([float(getattr(record, field)) for record in records if record.ok])


def print_metric(label: str, metric: dict[str, float | int]) -> None:
    print(
        f"{label:<20} count={metric['count']} "
        f"avg={metric['avg']:.3f}ms "
        f"p50={metric['p50']:.3f}ms "
        f"p95={metric['p95']:.3f}ms "
        f"p99={metric['p99']:.3f}ms "
        f"max={metric['max']:.3f}ms"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not settings.redis_session_enabled:
        raise RuntimeError("REDIS_SESSION_ENABLED must be enabled.")

    pg_store = SessionStore(settings)
    pg_store.init_db()
    store = RedisSessionStore(settings, pg_store=pg_store)
    store.init_cache()
    if int(store.client.xlen(store.archive_stream_key)) != 0:
        raise RuntimeError("Archive Stream must be empty before the benchmark.")

    run_id = uuid4().hex[:8]
    user_id = find_user_id(pg_store)
    session_ids: list[UUID] = []
    contexts = make_contexts(args.context_count, args.context_chars)
    user_content = fixed_text("session-benchmark-user-", args.user_chars)
    assistant_content = fixed_text(
        "session-benchmark-assistant-",
        args.assistant_chars,
    )
    cleanup_errors: list[str] = []

    try:
        executor_workers = max(
            args.workers,
            args.concurrency if args.mode == "closed-loop" else 1,
        )
        pool_limit = int(store.client.connection_pool.max_connections)
        if executor_workers > pool_limit:
            raise ValueError(
                f"Requested {executor_workers} workers exceeds the Redis client "
                f"connection pool limit of {pool_limit}."
            )
        for index in range(args.session_count):
            session = store.create_chat_session(
                user_id,
                f"Session benchmark {run_id} {index}",
            )
            session_ids.append(session.id)

        with ThreadPoolExecutor(max_workers=executor_workers) as executor:
            warmup_records: list[WriteRecord] = []
            next_index = 0
            if args.warmup_seconds > 0:
                warmup_abort = Event()
                warmup_sampler = BacklogSampler(
                    store,
                    args.sample_interval,
                    abort_event=warmup_abort,
                    max_backlog=args.max_backlog,
                )
                warmup_sampler.start()
                try:
                    if args.mode == "closed-loop":
                        warmup_records, _warmup_start, _warmup_finish = (
                            run_closed_loop_phase(
                                executor,
                                store,
                                session_ids,
                                contexts,
                                user_content,
                                assistant_content,
                                concurrency=args.concurrency,
                                duration=args.warmup_seconds,
                                start_index=next_index,
                                max_requests=args.max_requests,
                                abort_event=warmup_abort,
                            )
                        )
                    else:
                        warmup_records, _warmup_start, _warmup_finish = (
                            run_fixed_rate_phase(
                                executor,
                                store,
                                session_ids,
                                contexts,
                                user_content,
                                assistant_content,
                                rate=args.rate,
                                duration=args.warmup_seconds,
                                start_index=next_index,
                                max_requests=args.max_requests,
                                abort_event=warmup_abort,
                            )
                        )
                finally:
                    warmup_sampler.stop()
                warmup_errors = [record for record in warmup_records if not record.ok]
                if warmup_errors:
                    raise RuntimeError(
                        f"Warmup failed with {len(warmup_errors)} write errors."
                    )
                if warmup_sampler.abort_reason:
                    raise RuntimeError(
                        f"Warmup stopped: {warmup_sampler.abort_reason}."
                    )
                next_index += len(warmup_records)
                wait_for_archive(
                    store,
                    pg_store,
                    session_ids,
                    expected_messages=len(warmup_records) * 2,
                    timeout_seconds=args.drain_timeout,
                )

            baseline_messages = message_count(pg_store, session_ids)
            baseline_stream = int(store.client.xlen(store.archive_stream_key))
            phase_abort = Event()
            sampler = BacklogSampler(
                store,
                args.sample_interval,
                abort_event=phase_abort,
                max_backlog=args.max_backlog,
            )
            sampler.start()
            try:
                if args.mode == "closed-loop":
                    records, phase_started, phase_finished = run_closed_loop_phase(
                        executor,
                        store,
                        session_ids,
                        contexts,
                        user_content,
                        assistant_content,
                        concurrency=args.concurrency,
                        duration=args.duration,
                        start_index=next_index,
                        max_requests=args.max_requests,
                        abort_event=phase_abort,
                    )
                else:
                    records, phase_started, phase_finished = run_fixed_rate_phase(
                        executor,
                        store,
                        session_ids,
                        contexts,
                        user_content,
                        assistant_content,
                        rate=args.rate,
                        duration=args.duration,
                        start_index=next_index,
                        max_requests=args.max_requests,
                        abort_event=phase_abort,
                    )
                success_count = sum(1 for record in records if record.ok)
                errors = [record for record in records if not record.ok]
                messages_at_phase_end = message_count(pg_store, session_ids)
                pending_at_phase_end = pending_count(store, session_ids)
                stream_at_phase_end = int(
                    store.client.xlen(store.archive_stream_key)
                )
                expected_messages = baseline_messages + success_count * 2
                drain_ms, final_messages, final_pending, final_stream = wait_for_archive(
                    store,
                    pg_store,
                    session_ids,
                    expected_messages if not errors else None,
                    args.drain_timeout,
                )
            finally:
                sampler.stop()

        completion_seconds = max(0.000001, phase_finished - phase_started)
        archive_window_seconds = completion_seconds + drain_ms / 1000
        archived_events = (final_messages - baseline_messages) // 2
        unique_events = unique_event_count(pg_store, session_ids)
        expected_total_events = len(warmup_records) + success_count
        committed_total_events = final_messages // 2
        ambiguous_committed_events = max(
            0,
            committed_total_events - expected_total_events,
        )
        backlog_values = [value for _at, value in sampler.samples]
        load_backlog_samples = [
            sample
            for sample in sampler.samples
            if sample[0] <= completion_seconds
        ]
        archived_during_load = (messages_at_phase_end - baseline_messages) // 2
        consistency = {
            "message_count_matches": final_messages == expected_total_events * 2,
            "event_ids_unique": unique_events == expected_total_events * 2,
            "pending_and_stream_drained": final_pending == 0 and final_stream == 0,
        }
        consistency_ok = all(consistency.values())
        summary: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "configuration": {
                "mode": args.mode,
                "offered_qps": args.rate,
                "concurrency": args.concurrency,
                "duration_seconds": args.duration,
                "warmup_seconds": args.warmup_seconds,
                "session_count": args.session_count,
                "workers": args.workers,
                "user_chars": args.user_chars,
                "assistant_chars": args.assistant_chars,
                "context_count": args.context_count,
                "context_chars": args.context_chars,
                "max_requests": args.max_requests,
                "max_backlog": args.max_backlog,
            },
            "requests": {
                "total": len(records),
                "success": success_count,
                "errors": len(errors),
                "actual_completion_qps": success_count / completion_seconds,
                "phase_elapsed_seconds": completion_seconds,
                "service_latency_ms": metric_summary(records, "service_ms"),
                "queue_delay_ms": metric_summary(records, "queue_ms"),
                "scheduled_end_to_end_ms": metric_summary(
                    records,
                    "end_to_end_ms",
                ),
            },
            "archive": {
                "baseline_messages": baseline_messages,
                "final_messages": final_messages,
                "steady_archived_events": archived_events,
                "archived_during_load": archived_during_load,
                "during_load_events_per_second": archived_during_load
                / completion_seconds,
                "observed_events_per_second": archived_events
                / archive_window_seconds,
                "drain_ms_after_last_completion": drain_ms,
                "stream_start": baseline_stream,
                "stream_end_of_load": stream_at_phase_end,
                "pending_end_of_load": pending_at_phase_end,
                "stream_peak": max(backlog_values, default=baseline_stream),
                "stream_average": statistics.fmean(backlog_values)
                if backlog_values
                else float(baseline_stream),
                "stream_growth_per_second": linear_slope(load_backlog_samples),
                "stream_final": final_stream,
                "pending_final": final_pending,
                "sampler_errors": sampler.errors,
                "safety_stop": sampler.abort_reason,
            },
            "consistency": {
                "expected_total_events": expected_total_events,
                "committed_total_events": committed_total_events,
                "ambiguous_committed_events": ambiguous_committed_events,
                "unique_message_event_ids": unique_events,
                "expected_unique_message_event_ids": expected_total_events * 2,
                **consistency,
            },
            "errors": [record.__dict__ for record in errors[:20]],
        }
        if args.include_records:
            summary["records"] = [record.__dict__ for record in records]

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("Session pipeline benchmark")
        if args.mode == "closed-loop":
            print(
                f"mode=closed-loop concurrency={args.concurrency} "
                f"duration={completion_seconds:.3f}s requests={len(records)} "
                f"success={success_count} errors={len(errors)}"
            )
        else:
            print(
                f"mode=fixed-rate offered={args.rate:.2f} QPS "
                f"duration={completion_seconds:.3f}s requests={len(records)} "
                f"success={success_count} errors={len(errors)}"
            )
        print(
            f"actual_completion_qps="
            f"{summary['requests']['actual_completion_qps']:.3f}"
        )
        print_metric("service latency", summary["requests"]["service_latency_ms"])
        print_metric("queue delay", summary["requests"]["queue_delay_ms"])
        print_metric(
            "scheduled e2e",
            summary["requests"]["scheduled_end_to_end_ms"],
        )
        print(
            "archive "
            f"events={archived_events} "
            f"throughput={summary['archive']['observed_events_per_second']:.3f}/s "
            f"drain={drain_ms:.3f}ms "
            f"stream_end_load={stream_at_phase_end} "
            f"stream_peak={summary['archive']['stream_peak']} "
            f"stream_avg={summary['archive']['stream_average']:.3f} "
            f"stream_slope={summary['archive']['stream_growth_per_second']:.3f}/s "
            f"stream_final={final_stream} pending_final={final_pending}"
        )
        print(
            "consistency "
            f"messages={final_messages}/{expected_total_events * 2} "
            f"unique_event_ids={unique_events}/{expected_total_events * 2} "
            f"ok={consistency_ok}"
        )
        print(f"output={output_path}")
        if sampler.abort_reason:
            print(f"safety_stop={sampler.abort_reason}")

        if errors or sampler.errors or not consistency_ok or sampler.abort_reason:
            raise SystemExit(1)
    finally:
        try:
            cleanup_benchmark_sessions(store, pg_store, user_id, session_ids)
        except Exception as exc:
            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        store.close()
        if cleanup_errors:
            print("Cleanup errors:")
            for error in cleanup_errors:
                print(error)


if __name__ == "__main__":
    main()
