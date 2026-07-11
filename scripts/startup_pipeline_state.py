from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from uuid import uuid4


def begin_run(
    run_id_file: Path,
    ready_file: Path,
    failed_file: Path | None = None,
) -> str:
    run_id = str(uuid4())
    _atomic_write(run_id_file, run_id)
    ready_file.unlink(missing_ok=True)
    if failed_file is not None:
        failed_file.unlink(missing_ok=True)
    return run_id


def mark_ready(
    run_id_file: Path,
    ready_file: Path,
    failed_file: Path | None = None,
) -> str:
    run_id = _read_value(run_id_file)
    if not run_id:
        raise RuntimeError(f"Startup run id is missing: {run_id_file}")
    _atomic_write(ready_file, run_id)
    if failed_file is not None:
        failed_file.unlink(missing_ok=True)
    return run_id


def mark_failed(run_id_file: Path, failed_file: Path) -> str:
    run_id = _read_value(run_id_file)
    if not run_id:
        raise RuntimeError(f"Startup run id is missing: {run_id_file}")
    _atomic_write(failed_file, run_id)
    return run_id


def wait_until_ready(
    run_id_file: Path,
    ready_file: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.25,
    failed_file: Path | None = None,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_run_id = ""
    last_ready_id = ""
    while True:
        last_run_id = _read_value(run_id_file)
        last_ready_id = _read_value(ready_file)
        if last_run_id and last_ready_id == last_run_id:
            return last_run_id
        if failed_file is not None and _read_value(failed_file) == last_run_id:
            raise RuntimeError(
                f"Document startup pipeline failed for run {last_run_id}."
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for document startup pipeline: "
                f"run_id={last_run_id!r} ready_id={last_ready_id!r}"
            )
        time.sleep(max(0.01, poll_seconds))


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{value}\n", encoding="utf-8")
    temporary.replace(path)


def _read_value(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("begin", "ready", "fail", "wait"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--run-id-file", type=Path, required=True)
        if command in {"begin", "ready", "wait"}:
            command_parser.add_argument("--ready-file", type=Path, required=True)
        if command in {"begin", "ready", "fail", "wait"}:
            command_parser.add_argument("--failed-file", type=Path)
        if command == "wait":
            command_parser.add_argument(
                "--timeout-seconds",
                type=float,
                default=1800.0,
            )

    args = parser.parse_args()
    if args.command == "begin":
        run_id = begin_run(args.run_id_file, args.ready_file, args.failed_file)
        print(f"Document startup run started: {run_id}")
        return
    if args.command == "ready":
        run_id = mark_ready(args.run_id_file, args.ready_file, args.failed_file)
        print(f"Document startup run ready: {run_id}")
        return
    if args.command == "fail":
        if args.failed_file is None:
            parser.error("fail requires --failed-file")
        run_id = mark_failed(args.run_id_file, args.failed_file)
        print(f"Document startup run failed: {run_id}")
        return

    run_id = wait_until_ready(
        args.run_id_file,
        args.ready_file,
        timeout_seconds=args.timeout_seconds,
        failed_file=args.failed_file,
    )
    print(f"Document startup run observed ready: {run_id}")


if __name__ == "__main__":
    main()
